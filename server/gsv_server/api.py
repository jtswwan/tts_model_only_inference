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
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import wavio
from .engine import Engine
from .settings import ALLOWED_LANGS, Settings, load

log = logging.getLogger("gsv.api")

PCM_MEDIA_TYPE = "audio/L16"

_state: dict[str, Any] = {"engine": None, "settings": None}


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
    if profile is not None and profile not in ("quality", "low"):
        raise HTTPException(status_code=400, detail="profile 只接受 quality 或 low")


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
                if profile is not None and profile not in ("quality", "low"):
                    await ws.send_json({"error": "profile 只接受 quality 或 low"})
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
        settings: Settings = _state["settings"]
        if req.version is not None:
            settings.version = req.version
        if req.t2s is not None:
            settings.models.t2s = req.t2s
        if req.vits is not None:
            settings.models.vits = req.vits
        try:
            settings.validate()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        log.info("切換模型：%s", {"version": settings.version, "vits": settings.models.vits})
        try:
            _state["engine"] = Engine(settings)
        except Exception as exc:  # noqa: BLE001
            log.exception("切換模型失敗")
            _state["engine"] = None
            raise HTTPException(status_code=500, detail=f"切換失敗: {type(exc).__name__}: {exc}") from exc
        return _state["engine"].status()

    @app.post("/reference", summary="切換參考音訊（會重新暖機）")
    def switch_reference(req: ReferenceRequest) -> dict[str, Any]:
        settings: Settings = _state["settings"]
        if req.lang is not None and req.lang not in ALLOWED_LANGS:
            raise HTTPException(status_code=400, detail=f"lang 只接受 {sorted(ALLOWED_LANGS)}")
        if req.audio is not None:
            settings.reference.audio = req.audio
        if req.text is not None:
            settings.reference.text = req.text
        if req.lang is not None:
            settings.reference.lang = req.lang
        try:
            engine().set_reference()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("切換參考音訊失敗")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return engine().status()

    return app
