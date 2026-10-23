"""HTTP / WebSocket 介面。

與前一版的差異，逐項對應 README §1.4 列出的問題：

  - 不接受客戶端指定輸出路徑。前一版的 output 欄位讓任何連得上的人
    都能指定寫檔位置（server 綁 0.0.0.0），是任意檔案寫入。
  - 直接回傳音訊 bytes，不是伺服器端路徑字串。前一版跨機器呼叫
    只會拿到一個對方檔案系統上的路徑。
  - 錯誤回正確的 HTTP status code，不是一律 200 加 str(e)。
  - 用 lifespan 而非已棄用的 @app.on_event("startup")。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import pronunciation, wavio
from .engine import Engine, PreflightError, preflight
from .settings import ALLOWED_LANGS, Settings, load

log = logging.getLogger("gsv.api")

PCM_MEDIA_TYPE = "audio/L16"

_state: dict[str, Any] = {"engine": None, "settings": None}

# 切換模型 / 參考音訊要序列化：兩個併發的切換會互相覆蓋，而且會讓
# 兩份模型同時存在。
_switch_lock = threading.Lock()


def engine() -> Engine:
    eng = _state.get("engine")
    if eng is None:
        raise HTTPException(status_code=503, detail="引擎尚未就緒")
    return eng


# ---------------------------------------------------------------- 請求模型


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="要合成的文字")
    text_lang: str | None = Field(None, description="zh（中英混合）或 en")
    profile: str | None = Field(None, description="quality 或 low；省略則用設定檔的值")
    format: str = Field("wav", description="wav 或 pcm")


class ModelRequest(BaseModel):
    version: str | None = Field(None, description="v2Pro / v2ProPlus / v4")
    t2s: str | None = Field(None, description="GPT (s1) 權重路徑")
    vits: str | None = Field(None, description="SoVITS (s2) 權重路徑")
    device: str | None = Field(None, description="auto / cpu / cuda / cuda:0 …")
    is_half: bool | None = Field(None, description="fp16。CPU 上會被 upstream 強制關閉")


class DictRequest(BaseModel):
    lang: str = Field(..., description="en 或 zh")
    word: str = Field(..., min_length=1)
    # 英文是 ARPAbet 音素（["W","EY1","V"]），中文是拼音（["dang4"]）
    value: list[str] = Field(..., min_length=1)


class LookupRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ReferenceRequest(BaseModel):
    audio: str | None = None
    text: str | None = None
    lang: str | None = None


def _check_lang(lang: str | None) -> None:
    if lang is not None and lang not in ALLOWED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"text_lang 只接受 {sorted(ALLOWED_LANGS)}（本專案只含中英文前端），收到 {lang!r}",
        )


def _check_profile(profile: str | None) -> None:
    if profile is None:
        return
    if profile not in ("quality", "low"):
        raise HTTPException(status_code=400, detail="profile 只接受 quality 或 low")
    if profile == "low":
        settings: Settings = _state["settings"]
        # server 本身就設成 low 的話，那是擁有者的選擇，不再攔。
        if settings.latency.profile != "low" and not settings.latency.allow_low_profile:
            raise HTTPException(
                status_code=403,
                detail=(
                    "本 server 不接受單一請求指定 profile=low。"
                    "low 會啟用 streaming_mode，upstream 自評為 Medium quality。"
                    "要開放請在 config.yaml 設 latency.allow_low_profile: true。"
                ),
            )


# ---------------------------------------------------------------- 生命週期


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    _state["settings"] = settings
    log.info("載入模型中（version=%s, device=%s）…", settings.version, settings.device)
    # 建構與暖機都是同步且耗時的，放到執行緒避免卡住 event loop
    _state["engine"] = await asyncio.to_thread(Engine, settings)
    log.info("就緒：%s", _state["engine"].status())
    yield
    _state["engine"] = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load()
    app = FastAPI(
        title="GPT-SoVITS 推論 server",
        description="純推論、低延遲。主力 v2Pro。",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # ------------------------------------------------------------ 合成

    def _collect(text: str, text_lang: str | None, profile: str | None) -> tuple[int, bytes, float]:
        eng = engine()
        start = time.perf_counter()
        sr = None
        parts: list[np.ndarray] = []
        ttfa: float | None = None
        for chunk_sr, chunk in eng.synthesize(text, text_lang=text_lang, profile=profile):
            if ttfa is None:
                ttfa = time.perf_counter() - start
            sr = chunk_sr
            parts.append(chunk)
        if sr is None or not parts:
            raise HTTPException(status_code=500, detail="推論沒有產生任何音訊")
        return sr, np.concatenate(parts).tobytes(), ttfa or 0.0

    @app.post("/tts", summary="合成並回傳完整音檔")
    def tts(req: TTSRequest) -> Response:
        _check_lang(req.text_lang)
        _check_profile(req.profile)
        if req.format not in ("wav", "pcm"):
            raise HTTPException(status_code=400, detail="format 只接受 wav 或 pcm")
        drop_cursor = pronunciation.drop_cursor()
        try:
            sr, pcm, ttfa = _collect(req.text, req.text_lang, req.profile)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("合成失敗")
            raise HTTPException(status_code=500, detail=f"合成失敗: {type(exc).__name__}") from exc

        headers = {
            "X-Sample-Rate": str(sr),
            "X-TTFA-Ms": f"{ttfa * 1000:.1f}",
            "X-Audio-Seconds": f"{len(pcm) / 2 / sr:.3f}",
        }
        # upstream 的 replace_phs() 把不認識的音素靜默丟掉（只 print 一行）。
        # 那在 server 上等於看不見，所以這裡一定要回報出來。
        dropped = pronunciation.drops_since(drop_cursor)
        if dropped:
            headers["X-Dropped-Phones"] = " ".join(dropped)
        if req.format == "pcm":
            return Response(content=pcm, media_type=f"{PCM_MEDIA_TYPE};rate={sr};channels=1", headers=headers)
        return Response(content=wavio.wrap(sr, pcm), media_type="audio/wav", headers=headers)

    @app.get("/tts/stream", summary="串流 raw PCM（s16le 單聲道）")
    def tts_stream(
        text: str = Query(..., min_length=1),
        text_lang: str | None = Query(None),
        profile: str | None = Query(None),
    ) -> StreamingResponse:
        _check_lang(text_lang)
        _check_profile(profile)
        eng = engine()

        # 取樣率必須放在標頭裡，而標頭在產生第一塊音訊之前就要送出，
        # 所以從設定推導而非等第一塊資料。
        sr = eng.status()["sampling_rate"]
        chunk_samples = _state["settings"].server.pcm_chunk_samples

        def body() -> Iterator[bytes]:
            buf = bytearray()
            for _, chunk in eng.synthesize(text, text_lang=text_lang, profile=profile):
                buf.extend(chunk.tobytes())
                # 切成固定大小送出，只影響網路傳輸，不影響合成
                while len(buf) >= chunk_samples * 2:
                    yield bytes(buf[: chunk_samples * 2])
                    del buf[: chunk_samples * 2]
            if buf:
                yield bytes(buf)

        # 串流的標頭在產生第一塊音訊之前就送出，所以這裡還不知道會不會丟音素。
        # 用 trailer 不可靠（很多客戶端不讀），所以改成：丟掉的音素記在
        # /metrics 的 dropped_phones 裡，由控制台輪詢顯示。
        return StreamingResponse(
            body(),
            media_type=f"{PCM_MEDIA_TYPE};rate={sr};channels=1",
            headers={"X-Sample-Rate": str(sr), "X-Audio-Format": "s16le-mono"},
        )

    @app.websocket("/tts/ws")
    async def tts_ws(ws: WebSocket) -> None:
        """收 JSON {text, text_lang?, profile?}，回二進位 PCM 區塊，最後回一則 JSON 統計。"""
        await ws.accept()
        try:
            while True:
                payload = json.loads(await ws.receive_text())
                text = (payload.get("text") or "").strip()
                if not text:
                    await ws.send_json({"error": "text 不可為空"})
                    continue
                lang, profile = payload.get("text_lang"), payload.get("profile")
                if lang is not None and lang not in ALLOWED_LANGS:
                    await ws.send_json({"error": f"text_lang 只接受 {sorted(ALLOWED_LANGS)}"})
                    continue
                try:
                    _check_profile(profile)
                except HTTPException as exc:
                    await ws.send_json({"error": exc.detail})
                    continue

                eng = engine()
                sr = eng.status()["sampling_rate"]
                await ws.send_json({"event": "begin", "sample_rate": sr, "format": "s16le-mono"})
                start = time.perf_counter()
                ttfa: float | None = None
                total = 0

                # synthesize 是同步產生器。整批搬到執行緒再送會喪失串流的
                # 即時性，所以在執行緒裡每取出一塊就立刻丟進 queue，
                # 由 event loop 端即時送出。
                queue: asyncio.Queue[bytes | None] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def pump() -> None:
                    try:
                        for _, chunk in eng.synthesize(text, text_lang=lang, profile=profile):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk.tobytes())
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                task = asyncio.create_task(asyncio.to_thread(pump))
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if ttfa is None:
                        ttfa = time.perf_counter() - start
                    total += len(item)
                    await ws.send_bytes(item)
                await task

                await ws.send_json(
                    {
                        "event": "end",
                        "ttfa_ms": round((ttfa or 0.0) * 1000, 1),
                        "total_ms": round((time.perf_counter() - start) * 1000, 1),
                        "audio_seconds": round(total / 2 / sr, 3),
                    }
                )
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("WebSocket 合成失敗")
            try:
                await ws.send_json({"error": f"{type(exc).__name__}"})
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ 控制台

    CONSOLE = Path(__file__).resolve().parent / "console.html"

    @app.get("/", include_in_schema=False)
    def console() -> Response:
        if not CONSOLE.is_file():
            return HTMLResponse("<h1>找不到 console.html</h1>", status_code=500)
        # 單一 HTML 檔、不依賴任何 CDN —— 必須能在離線的機器上操作本機 server
        return FileResponse(CONSOLE, media_type="text/html; charset=utf-8")

    @app.get("/candidates", summary="掃描可用的模型與參考音訊")
    def candidates() -> dict[str, Any]:
        """列出 pretrained_models/ 底下可用的候選。

        SoVITS 的版本是讀「檔頭前兩個 byte」判定的（process_ckpt.py:100），
        很便宜。GPT 的 .ckpt 不載入，只列檔名與大小 —— 載一次要好幾百 MB。
        """
        import wave

        from process_ckpt import head2version

        # 這些是支援模型，不是角色模型，不該出現在選單裡
        SUPPORT = ("/sv/", "/fast_langdetect/", "/chinese-hubert-base/",
                   "/chinese-roberta-wwm-ext-large/", "/g2pw-chinese/")

        settings: Settings = _state["settings"]
        root = Path(settings.resolve("GPT_SoVITS/pretrained_models"))
        sovits: list[dict[str, Any]] = []
        gpt: list[dict[str, Any]] = []
        refs: list[dict[str, Any]] = []
        if not root.is_dir():
            return {"root": str(root), "sovits": [], "gpt": [], "references": []}

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = os.path.relpath(path, Path(settings.resolve(".")))
            if any(k in "/" + rel.replace(os.sep, "/") for k in SUPPORT):
                continue
            size_mb = round(path.stat().st_size / 1e6, 1)
            suffix = path.suffix.lower()
            if suffix == ".pth":
                entry: dict[str, Any] = {"path": rel, "size_mb": size_mb}
                try:
                    with open(path, "rb") as f:
                        head = f.read(2)
                    if head in head2version:
                        tv, mv, lora = head2version[head]
                        entry.update({"version": mv, "text_version": tv, "is_lora": bool(lora),
                                      "header": head.decode("latin-1")})
                    else:
                        # 檔頭是 PK 的是舊格式，版本靠檔案大小猜（process_ckpt.py:111-126）
                        entry.update({"version": None, "header": head.decode("latin-1", "replace")})
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                sovits.append(entry)
            elif suffix == ".ckpt":
                gpt.append({"path": rel, "size_mb": size_mb})
            elif suffix == ".wav":
                entry = {"path": rel, "size_mb": size_mb}
                try:
                    with wave.open(str(path)) as w:
                        sr, n = w.getframerate(), w.getnframes()
                        entry.update({"sample_rate": sr, "channels": w.getnchannels(),
                                      "seconds": round(n / sr, 2) if sr else None})
                        # TTS.py:816 要求換算成 16kHz 後樣本數在 48000~160000
                        entry["usable"] = bool(sr and 3.0 <= n / sr <= 10.0)
                except Exception:  # noqa: BLE001
                    # 非 PCM（例如 float32）的 wav，wave 模組讀不了
                    entry.update({"sample_rate": None, "usable": None,
                                  "note": "非 PCM 格式，wave 模組無法讀取；add_voice.py 可以轉換"})
                refs.append(entry)

        return {"root": os.path.relpath(root, Path(settings.resolve("."))),
                "sovits": sovits, "gpt": gpt, "references": refs}

    @app.get("/runtime", summary="執行環境資訊")
    def runtime() -> dict[str, Any]:
        """控制台的「環境」區塊。這些是偵測來的，不是設定來的。"""
        import platform

        import torch

        info: dict[str, Any] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [],
            "cpu_threads": torch.get_num_threads(),
            "platform": f"{platform.system()} {platform.machine()}",
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                    "capability": f"{props.major}.{props.minor}",
                    # fp16 要 sm 5.3 以上才划算；V100 是 7.0
                    "fp16_ok": (props.major, props.minor) >= (5, 3),
                })
        try:
            import torchaudio

            info["torchaudio"] = torchaudio.__version__
        except Exception:  # noqa: BLE001
            info["torchaudio"] = None
        return info

    # ------------------------------------------------------------ 狀態

    @app.get("/health", summary="狀態")
    def health() -> JSONResponse:
        eng = _state.get("engine")
        if eng is None:
            return JSONResponse({"ready": False}, status_code=503)
        return JSONResponse({"ready": True, **eng.status()})

    @app.get("/metrics", summary="延遲量測（TTFA 為主要指標）")
    def metrics() -> dict[str, Any]:
        return engine().metrics()

    # ------------------------------------------------------------ 切換

    @app.post("/model", summary="切換模型 / 版本（會重新載入並暖機）")
    def switch_model(req: ModelRequest) -> dict[str, Any]:
        import copy

        settings: Settings = _state["settings"]
        # 在動到任何東西之前先把新設定在副本上組起來並驗證完。
        # 驗證不過就連舊引擎都不碰 —— 一個打錯字的請求不該讓 server 癱掉。
        candidate = copy.deepcopy(settings)
        if req.version is not None:
            candidate.version = req.version
        if req.t2s is not None:
            candidate.models.t2s = req.t2s
        if req.vits is not None:
            candidate.models.vits = req.vits
        if req.device is not None:
            candidate.device = req.device
        if req.is_half is not None:
            candidate.is_half = req.is_half
        try:
            candidate.validate()
            preflight(candidate)
        except (ValueError, PreflightError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        log.info("切換模型：%s", {"version": candidate.version, "vits": candidate.models.vits})
        with _switch_lock:
            # 先把舊的釋放掉再載入新的。若先建新的再換掉舊的，兩份模型會同時
            # 在 GPU 上、VRAM 短暫加倍 —— V100 上可能直接 OOM。代價是這段
            # 時間內請求會收到 503，這比 OOM 好。
            #
            # 另外：prompt_cache 的 prompt_semantic 是用舊 vits 模型的
            # extract_latent 算出來的（TTS.py:830），換模型後必須重算。
            # 重建 Engine 會重跑暖機，所以自動處理掉。
            previous = copy.deepcopy(settings)
            old = _state.get("engine")
            _state["engine"] = None
            if old is not None:
                old.release()
            try:
                engine_new = Engine(candidate)
            except Exception as exc:  # noqa: BLE001
                log.exception("切換模型失敗，嘗試還原先前的設定")
                # preflight 已經過了還失敗，代表是載入本身的問題。
                # 盡力把原本的狀態救回來，不要把 server 留在沒有引擎的狀態。
                try:
                    _state["engine"] = Engine(previous)
                    _state["settings"] = previous  # 與引擎持有的是同一物件
                    raise HTTPException(
                        status_code=500,
                        detail=f"切換失敗（已還原為先前的模型）: {type(exc).__name__}: {exc}",
                    ) from exc
                except HTTPException:
                    raise
                except Exception as restore_exc:  # noqa: BLE001
                    log.exception("還原也失敗")
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"切換失敗且無法還原: {type(exc).__name__}: {exc}；"
                            f"還原錯誤: {type(restore_exc).__name__}: {restore_exc}"
                        ),
                    ) from exc
            _state["engine"] = engine_new
            # 關鍵：讓 _state["settings"] 就是引擎持有的那一份物件。
            #
            # Engine(candidate) 持有的是 candidate，若這裡不換掉
            # _state["settings"]，兩者就會變成不同物件 —— 之後 /reference
            # 改的是沒人讀的那一份，set_reference() 仍用舊路徑重新暖機並
            # 回 200，切換靜默失效。
            _state["settings"] = candidate
        return _state["engine"].status()

    @app.post("/reference", summary="切換參考音訊（會重新暖機）")
    def switch_reference(req: ReferenceRequest) -> dict[str, Any]:
        import copy
        import os

        if req.lang is not None and req.lang not in ALLOWED_LANGS:
            raise HTTPException(status_code=400, detail=f"lang 只接受 {sorted(ALLOWED_LANGS)}")

        eng = engine()
        # 一定要改「引擎持有的那一份」設定，不是 _state["settings"] ——
        # 兩者在 /model 之後可能已經是不同物件。
        live = eng.settings

        # 先在副本上驗證，失敗就什麼都不動
        candidate = copy.deepcopy(live.reference)
        if req.audio is not None:
            candidate.audio = req.audio
        if req.text is not None:
            candidate.text = req.text
        if req.lang is not None:
            candidate.lang = req.lang
        if not candidate.text.strip():
            raise HTTPException(status_code=400, detail="reference.text 不可為空")
        full = live.resolve(candidate.audio)
        if not os.path.isfile(full):
            raise HTTPException(status_code=400, detail=f"找不到參考音訊: {candidate.audio}")

        previous = copy.deepcopy(live.reference)
        try:
            with _switch_lock:
                live.reference = candidate
                eng.set_reference()
        except (PreflightError, FileNotFoundError) as exc:
            live.reference = previous
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("切換參考音訊失敗，還原先前的參考")
            live.reference = previous
            try:
                eng.set_reference()
            except Exception:  # noqa: BLE001
                log.exception("還原參考音訊也失敗")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return eng.status()

    # ---- 字典 ----
    #
    # 為什麼需要這組端點：英文的 english.py:322 對「不在字典裡、長度 <= 3」的
    # 詞一律逐字母拼讀（wav 唸成 W-A-V），中文的 g2pw 遇到人名與專有名詞會猜錯
    # 聲調。兩邊都要能當場改、當場生效，而不是每次都得改檔案重啟。

    @app.get("/dict", summary="列出目前生效的自訂發音")
    def dict_list() -> dict[str, Any]:
        entries = pronunciation.load()
        return {
            "path": str(pronunciation.DEFAULT_PATH),
            "entries": entries,
            "counts": {lang: len(entries.get(lang, {})) for lang in pronunciation.LANGS},
            "recent_dropped_phones": pronunciation.recent_drops(),
        }

    @app.post("/dict/lookup", summary="查一段文字現在實際會怎麼唸")
    def dict_lookup(req: LookupRequest) -> dict[str, Any]:
        try:
            return pronunciation.lookup(req.text)
        except Exception as exc:  # noqa: BLE001
            log.exception("查詢發音失敗")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.put("/dict", summary="新增或修改一筆自訂發音（立刻生效，不必重啟）")
    def dict_put(req: DictRequest) -> dict[str, Any]:
        eng = engine()
        try:
            with _switch_lock:
                entries = pronunciation.upsert(req.lang, req.word, req.value)
                eng.custom_pronunciations = entries
        except pronunciation.PronunciationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("寫入自訂發音失敗")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        # 中文的 key 會被正規化成簡體（管線裡 text_normalize 做繁簡轉換），
        # 所以要回報實際存進去的是哪個詞 —— 使用者打繁體時才不會以為沒生效。
        stored = pronunciation.normalize_word(req.lang, req.word)
        return {
            "entries": entries,
            "stored_key": stored,
            "normalized": stored != req.word.strip(),
            "lookup": pronunciation.lookup(req.word, entries),
        }

    @app.delete("/dict", summary="刪掉一筆自訂發音並還原成 upstream 原本的讀音")
    def dict_delete(
        lang: str = Query(..., description="en 或 zh"),
        word: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        eng = engine()
        try:
            with _switch_lock:
                entries = pronunciation.remove(lang, word)
                eng.custom_pronunciations = entries
        except pronunciation.PronunciationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"entries": entries, "lookup": pronunciation.lookup(word, entries)}

    return app
