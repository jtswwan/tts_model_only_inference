"""推論引擎：包裝 upstream 的 TTS，負責暖機、快取與延遲量測。

本模組不改 upstream 的推論程式碼。它做三件事：

  1. 用正確的形式建構 TTS（見 _build_tts_config 關於 "custom" 鍵的說明）
  2. 啟動時把所有能預先算完的東西算完（暖機），讓請求路徑上只剩必要計算
  3. 量測延遲，並以 TTFA 為主要指標
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Iterator

import numpy as np

from . import bootstrap

# upstream 在模組層級就固定了 os.getcwd()（TTS.py:13），因此 setup() 必須
# 在 import 之前完成。
bootstrap.setup()

from . import bert_trim  # noqa: E402
from .settings import Settings  # noqa: E402

from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

log = logging.getLogger("gsv.engine")


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_tts_config(settings: Settings) -> TTS_Config:
    """建構 TTS_Config。

    **必須包在 "custom" 鍵底下。** TTS.py:304-306 的邏輯是：

        configs_ = deepcopy(self.default_configs)
        configs_.update(configs)
        self.configs = configs_.get("custom", configs_["v2"])

    平鋪傳入的鍵會被 update 放進最外層，然後因為沒有 "custom" 鍵而
    整組被忽略，退回 v2 的預設值 —— 會安靜地載入錯誤的模型。
    """
    device = _resolve_device(settings.device)
    return TTS_Config(
        {
            "custom": {
                "device": device,
                "is_half": settings.is_half,
                "version": settings.version,
                "t2s_weights_path": settings.resolve(settings.models.t2s),
                "vits_weights_path": settings.resolve(settings.models.vits),
                "bert_base_path": settings.resolve(settings.models.bert),
                "cnhuhbert_base_path": settings.resolve(settings.models.cnhubert),
            }
        }
    )


class _PhoneBertCache:
    """文字前端結果的 LRU 快取。

    同一段文字在同一語言與版本下，音素序列與 BERT 特徵是確定性的，
    所以快取命中與重算在數值上相同（等價優化，README §4 類別一）。

    回傳時複製一份：呼叫端雖然目前只做 torch.cat 等不會就地修改的操作，
    但快取的生命週期比單次請求長，不該把可變物件的所有權交出去。
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def wrap(self, fn):
        def wrapped(text: str, language: str, version: str, final: bool = False):
            if self.maxsize <= 0:
                return fn(text, language, version, final)
            key = (text, language, version)
            with self._lock:
                hit = self._data.get(key)
                if hit is not None:
                    self._data.move_to_end(key)
                    self.hits += 1
            if hit is not None:
                phones, bert, norm_text = hit
                return list(phones), bert.clone(), norm_text
            phones, bert, norm_text = fn(text, language, version, final)
            with self._lock:
                self.misses += 1
                self._data[key] = (list(phones), bert.clone(), norm_text)
                while len(self._data) > self.maxsize:
                    self._data.popitem(last=False)
            return phones, bert, norm_text

        return wrapped

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else None,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class Engine:
    """序列化的推論引擎。

    GPU 操作一律序列化（單一鎖）。理由是低延遲的前提是「可預測」的延遲：
    多個請求在同一張 GPU 上爭搶會讓每一個都變慢且抖動，排隊反而可預測。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._ttfa: deque[float] = deque(maxlen=128)
        self._total: deque[float] = deque(maxlen=128)
        self.warm = False
        self.trim_stats: dict[str, Any] = {}
        self.trim_verified = False
        self.quality_notes: list[str] = []

        t0 = time.perf_counter()
        self.config = _build_tts_config(settings)
        self.tts = TTS(self.config)
        self.load_seconds = time.perf_counter() - t0

        if settings.latency.trim_bert:
            # 剪裁前先留下原模型對探針文字的輸出。取法與 TextPreprocessor
            # 實際使用的那條路徑完全相同，剪完後逐位元比對。
            #
            # 這個自檢是刻意設計成「不通過就拒絕啟動」的：剪裁不等價會默默
            # 影響音質，而音質問題不會在任何 log 或 metric 上出現。寧可開不
            # 起來也不要悄悄變差。
            baseline = bert_trim.probe(
                self.tts.bert_model, self.tts.bert_tokenizer, self.config.device
            )
            wrapper, stats = bert_trim.trim(self.tts.bert_model)
            after = bert_trim.probe(wrapper, self.tts.bert_tokenizer, self.config.device)
            bert_trim.assert_equivalent(baseline, after)
            self.trim_verified = True

            self.tts.bert_model = wrapper
            self.tts.text_preprocessor.bert_model = wrapper
            self.trim_stats = stats
            log.info(
                "BERT 已剪裁：%d → %d 層，省下 %s 個參數（%.2f%%）；"
                "%d 段探針文字的輸出逐位元相同",
                stats["num_layers_original"],
                stats["num_layers_kept"],
                f"{stats['params_removed']:,}",
                stats["removed_pct"],
                len(baseline),
            )
            del baseline, after

        self.text_cache = _PhoneBertCache(settings.latency.text_cache_size)
        pre = self.tts.text_preprocessor
        pre.get_phones_and_bert = self.text_cache.wrap(pre.get_phones_and_bert)

        self._warn_quality_deviations()
        self.warmup()

    def _warn_quality_deviations(self) -> None:
        """把任何會影響輸出的設定偏離大聲講出來。

        這些都不是錯誤，是使用者可以選的。但它們會改變音訊，所以不該安靜
        生效 —— 音質問題不會出現在任何 metric 上，只能靠事先講明。
        """
        s = self.settings
        lat, samp = s.latency, s.sampling
        notes: list[str] = []

        if lat.profile != "quality":
            notes.append(
                "latency.profile=low：啟用 streaming_mode，upstream 在 "
                "TTS.py:1027 自評為 Medium quality，且會強制 parallel_infer=False "
                "使 AR 改走 infer_panel_naive（另一條程式碼路徑）"
            )
        if lat.fixed_length_chunk:
            notes.append("latency.fixed_length_chunk=true：upstream 在 TTS.py:1033 自評為 lower quality")
        if lat.text_split_method != "cut1":
            notes.append(
                f"latency.text_split_method={lat.text_split_method}（upstream 預設 cut1）："
                "切句粒度改變送進 GPT 的文本，會改變韻律"
            )
        if samp.speed_factor != 1.0:
            notes.append(
                f"sampling.speed_factor={samp.speed_factor}：TTS.py:1099 會因此強制關閉 "
                "split_bucket，並對輸出音訊重採樣"
            )
        if (samp.top_k, samp.top_p, samp.temperature) != (5, 1.0, 1.0):
            notes.append(
                f"取樣參數 top_k={samp.top_k} top_p={samp.top_p} "
                f"temperature={samp.temperature}（upstream 執行時預設 5 / 1 / 1）"
            )
        if not lat.trim_bert:
            # 這個方向是變慢而非變差，列出來只是說明為何沒有剪裁
            log.info("latency.trim_bert=false：BERT 未剪裁（不影響音質，只是較慢）")

        self.quality_notes = notes
        if notes:
            log.warning("以下設定會改變輸出，相對 upstream 預設：")
            for n in notes:
                log.warning("  - %s", n)
        else:
            log.info("所有會影響輸出的設定都與 upstream 預設一致")

    # ---- 暖機 ----

    def warmup(self) -> dict[str, float]:
        """把所有能預先算完的東西算完。

        參考音訊是固定的，它只衍生四樣東西（見 README §5.3）：
        prompt_semantic（需要 CNHubert）、refer_spec、sv_emb（需要 SV 模型）、
        以及 prompt 文本的 phones 與 bert_features。全部在這裡算完填進
        tts.prompt_cache 之後，請求路徑上就不再碰 CNHubert 與 SV 模型，
        也不再有任何音檔 I/O。
        """
        ref = self.settings.reference
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        self.tts.set_ref_audio(self.settings.resolve(ref.audio))
        timings["set_ref_audio"] = time.perf_counter() - t0

        if self.settings.latency.warmup_dummy_run:
            # 第一次真實推論會付掉 CUDA kernel、cuDNN autotune、onnxruntime
            # session 初始化、jieba 詞典載入等一次性成本。在這裡付掉，
            # 避免第一個真實請求出現延遲尖峰。
            t0 = time.perf_counter()
            for _ in self._run(self._build_inputs("暖機。", warmup=True)):
                pass
            timings["dummy_run"] = time.perf_counter() - t0

        self.warm = True
        log.info("暖機完成：%s", {k: round(v, 3) for k, v in timings.items()})
        return timings

    def set_reference(self) -> dict[str, float]:
        """換掉參考音訊並重新暖機。

        參考音訊變更會讓 prompt_cache 失效（TTS.py:1133 比對 ref_audio_path），
        prompt 文本變更則由 TTS.py:1158-1165 在下一次 run 時處理。這裡主動
        重算，讓失效的成本付在切換時而不是下一個請求上。
        """
        import os

        path = self.settings.resolve(self.settings.reference.audio)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到參考音訊: {path}")
        self.warm = False
        # prompt 文本可能跟著換，對應的前端快取一併清掉
        self.text_cache.clear()
        return self.warmup()

    # ---- 推論 ----

    def _build_inputs(
        self,
        text: str,
        *,
        text_lang: str | None = None,
        profile: str | None = None,
        warmup: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        s = self.settings
        lat = s.latency
        samp = s.sampling
        profile = profile or lat.profile

        inputs: dict[str, Any] = {
            "text": text,
            "text_lang": text_lang or s.text_lang,
            # 參考音訊路徑仍要傳，但 prompt_cache 已在暖機時填好，
            # TTS.py:1133 比對路徑相同就不會重算（見 README §5.3）。
            "ref_audio_path": s.resolve(s.reference.audio),
            "prompt_text": s.reference.text,
            "prompt_lang": s.reference.lang,
            "top_k": samp.top_k,
            "top_p": samp.top_p,
            "temperature": samp.temperature,
            "repetition_penalty": samp.repetition_penalty,
            "speed_factor": samp.speed_factor,
            "seed": samp.seed,
            "text_split_method": lat.text_split_method,
            "fragment_interval": lat.fragment_interval,
            "batch_size": 1,
        }

        if profile == "quality":
            # upstream 在 TTS.py:1026 把 return_fragment 標為 "Best Quality"。
            # 保持 parallel_infer=True，使 AR 走 infer_panel_batch_infer
            # （TTS.py:1066-1067）—— 與 upstream 預設同一條路徑。
            inputs["return_fragment"] = True
            inputs["streaming_mode"] = False
            inputs["parallel_infer"] = True
        else:
            # upstream 在 TTS.py:1027 把 streaming_mode 標為 "Medium quality"。
            # 它會強制 parallel_infer=False（TTS.py:1079-1081），AR 改走
            # infer_panel_naive，是另一條程式碼路徑。
            inputs["return_fragment"] = False
            inputs["streaming_mode"] = True
            inputs["parallel_infer"] = False
            inputs["overlap_length"] = lat.overlap_length
            inputs["min_chunk_length"] = lat.min_chunk_length
            inputs["fixed_length_chunk"] = lat.fixed_length_chunk

        if warmup:
            inputs["text_split_method"] = "cut0"  # 不切，暖機只要跑通一次

        inputs.update(overrides)
        return inputs

    def _run(self, inputs: dict[str, Any]) -> Iterator[tuple[int, np.ndarray]]:
        yield from self.tts.run(inputs)

    def synthesize(
        self,
        text: str,
        *,
        text_lang: str | None = None,
        profile: str | None = None,
        **overrides: Any,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """逐段產出 (sample_rate, int16 波形)。

        鎖會在整個產生過程中持有，讓同時間只有一個請求在用 GPU。
        """
        if not text or not text.strip():
            raise ValueError("text 不可為空")

        inputs = self._build_inputs(text, text_lang=text_lang, profile=profile, **overrides)

        with self._lock:
            start = time.perf_counter()
            first_at: float | None = None
            try:
                for sr, chunk in self._run(inputs):
                    if first_at is None:
                        first_at = time.perf_counter() - start
                        self._ttfa.append(first_at)
                    yield sr, chunk
            finally:
                self._total.append(time.perf_counter() - start)

    def stop(self) -> None:
        """要求 upstream 在下一個檢查點中止目前的推論（TTS.stop）。"""
        self.tts.stop()

    # ---- 狀態 ----

    def metrics(self) -> dict[str, Any]:
        def summary(samples: deque[float]) -> dict[str, Any] | None:
            if not samples:
                return None
            ordered = sorted(samples)
            return {
                "count": len(ordered),
                "last_ms": round(samples[-1] * 1000, 1),
                "median_ms": round(ordered[len(ordered) // 2] * 1000, 1),
                "min_ms": round(ordered[0] * 1000, 1),
                "max_ms": round(ordered[-1] * 1000, 1),
            }

        return {
            "ttfa": summary(self._ttfa),
            "total": summary(self._total),
            "text_cache": self.text_cache.stats(),
            "bert_trim": self.trim_stats or None,
            "model_load_seconds": round(self.load_seconds, 2),
        }

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "warm": self.warm,
            "version": self.config.version,
            "device": str(self.config.device),
            "is_half": self.config.is_half,
            "use_vocoder": self.config.use_vocoder,
            "sampling_rate": self.tts.configs.sampling_rate
            if not self.config.use_vocoder
            else self.tts.vocoder_configs["sr"],
            "latency_profile": s.latency.profile,
            "text_split_method": s.latency.text_split_method,
            "reference_audio": s.reference.audio,
            "text_lang": s.text_lang,
            # 音質狀態。clean 為 true 表示所有會影響輸出的設定都與 upstream
            # 預設一致，且 BERT 剪裁已通過逐位元等價自檢。
            "quality": {
                "clean": not self.quality_notes
                and (self.trim_verified or not s.latency.trim_bert),
                "bert_trim_verified": self.trim_verified,
                "bert_trim_enabled": s.latency.trim_bert,
                "allow_low_profile": s.latency.allow_low_profile,
                "deviations": self.quality_notes,
            },
        }
