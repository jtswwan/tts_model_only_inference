"""把中英切換處多出來的停頓剪掉。

為什麼會有那個停頓
------------------
`TextPreprocessor.py:212-222` 對非中文片段的 BERT 特徵給全零，而且每個片段的
BERT 是各自獨立算再 `torch.cat` 接起來的。所以每個語言邊界上，條件訊號都是
從真實的 1024 維特徵掉到全零再跳回來 —— 模型在那裡會猶豫。

實測 `這個 server 用 v2Pro 模型`：語言邊界處有一段 60 ms 的低能量區，峰值
RMS 0.0097（不是數位靜音，是模型自己產生的）。句子中間沒有任何標點，那個
停頓就是多餘的。

這不是能從上游修的東西（見 README：模型就是用這個管線訓練的，改成餵真實
BERT 特徵會造成 train/inference 不一致）。所以改成在輸出端處理。

為什麼不影響 TTFA
-----------------
關鍵是**只扣住靜音，不扣住語音**：

- 第一個語音出現之前，所有東西原樣直通 —— 所以第一個 byte 的時間完全不變。
- 之後遇到低能量的窗就先存著不送；等語音回來時，最多只送 `max_gap_ms` 那麼多
  存起來的靜音，再送語音。

扣住的是「客戶端本來就只會播成無聲」的樣本，所以聽感上不會慢，而且不需要
任何前瞻緩衝 —— 串流一樣適用。

什麼時候才會啟用
----------------
只在「有中英邊界」**且**「中間沒有任何停頓標點」時啟用。有標點時那些停頓是
應該存在的，而且沒辦法可靠地把音訊上的 gap 對應回文字位置，所以寧可不動。
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

log = logging.getLogger("gsv.gaptrim")

# 會產生合理停頓的標點。有這些就不啟用 —— 那些停頓是該有的。
PAUSE_PUNCT = set("，。！？；：、,.!?;:\n…—")


def boundaries(text: str) -> list[str]:
    """回傳句中語言邊界的型別，依出現順序。

    `"zh2en"`（中文後面接英文）或 `"en2zh"`（英文後面接中文）。

    為什麼要分：中文講到一半要切進英文時，那個停頓是模型在猶豫，不該有；
    但講完一個英文詞再回到中文，稍微頓一下是自然的。所以只消 zh2en。

    句中有停頓標點時回空的 —— 那些停頓是該有的，而且沒辦法把音訊上的
    gap 可靠地對應回文字位置。
    """
    from text.LangSegmenter import LangSegmenter

    stripped = text.strip()
    if any(ch in PAUSE_PUNCT for ch in stripped[:-1]):
        # 句尾的標點不算（那是結尾，不是中間）
        return []
    try:
        segs = LangSegmenter.getTexts(stripped)
    except Exception:  # noqa: BLE001
        return []
    # 非 en 一律視為 zh —— 與 TextPreprocessor.py:166-168 的處理一致，
    # 並且合併相鄰的同類片段（否則會數出不存在的邊界）
    kinds: list[str] = []
    for seg in segs:
        k = "en" if seg["lang"] == "en" else "zh"
        if not kinds or kinds[-1] != k:
            kinds.append(k)
    return [f"{a}2{b}" for a, b in zip(kinds, kinds[1:])]


def has_language_switch_without_punct(text: str) -> bool:
    return bool(boundaries(text))


class GapLimiter:
    """串流安全的靜音上限器。

    每次 `feed()` 餵一塊 int16 樣本，產出處理過的樣本。結束時要呼叫 `flush()`。
    """

    def __init__(
        self,
        sample_rate: int,
        max_gap_ms: int = 120,
        win_ms: int = 10,
        rel_threshold: float = 0.04,
        abs_floor: float = 60.0,
        fade_ms: float = 2.0,
        kinds: list[str] | None = None,
        min_gap_ms: int = 150,
    ) -> None:
        # 句中邊界的型別，依順序。只有 zh2en 的停頓會被消掉。
        # 音訊上的 gap 和文字上的邊界是「依序對應」的近似 —— 模型不一定在
        # 每個邊界都停，所以 gap 數通常少於邊界數。對不上的就不動。
        self.kinds = list(kinds or [])
        self.gap_index = 0
        # 只有超過這個長度的靜音才算「一個語言邊界上的停頓」。
        # 音節之間本來就有很多幾十毫秒的低能量空隙，把那些也算進來的話，
        # 邊界索引會被吃光，真正的停頓反而對應不到正確的型別。
        self.min_gap = int(sample_rate * min_gap_ms / 1000)
        self.sr = sample_rate
        self.win = max(1, int(sample_rate * win_ms / 1000))
        self.max_gap = int(sample_rate * max_gap_ms / 1000)
        self.rel = rel_threshold
        self.abs_floor = abs_floor
        self.fade = max(1, int(sample_rate * fade_ms / 1000))

        self._carry = np.empty(0, dtype=np.int16)
        self._pending: list[np.ndarray] = []
        self._started = False
        self._peak = 0.0
        self.trimmed_samples = 0

    def _rms(self, win: np.ndarray) -> float:
        x = win.astype(np.float32)
        return float(np.sqrt((x * x).mean())) if len(x) else 0.0

    def _threshold(self) -> float:
        return max(self.abs_floor, self._peak * self.rel)

    def _release_pending(self) -> list[np.ndarray]:
        """送出存起來的靜音，但最多只送 max_gap 那麼多（保留靠近語音的尾段）。"""
        if not self._pending:
            return []
        buf = np.concatenate(self._pending)
        self._pending.clear()

        if len(buf) < self.min_gap:
            # 音節之間的小空隙，不算邊界停頓，也不消耗邊界索引
            return [buf]

        kind = self.kinds[self.gap_index] if self.gap_index < len(self.kinds) else None
        self.gap_index += 1
        if kind != "zh2en":
            # 英文接中文（或對應不到邊界）的停頓保持原樣
            return [buf]

        if len(buf) <= self.max_gap:
            return [buf]
        self.trimmed_samples += len(buf) - self.max_gap
        kept = buf[-self.max_gap:].copy()
        # 切口兩側都是低能量，爆音風險很小，但還是做一個短淡入當保險
        n = min(self.fade, len(kept))
        if n > 1:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            kept[:n] = (kept[:n].astype(np.float32) * ramp).astype(np.int16)
        return [kept]

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16)
        data = np.concatenate([self._carry, chunk]) if len(self._carry) else chunk
        n_win = len(data) // self.win
        self._carry = data[n_win * self.win:].copy()

        out: list[np.ndarray] = []
        for i in range(n_win):
            w = data[i * self.win:(i + 1) * self.win]
            rms = self._rms(w)
            self._peak = max(self._peak, rms)

            if not self._started:
                # 第一個語音出現之前原樣直通 —— TTFA 一個 byte 都不動
                out.append(w)
                if rms >= self._threshold():
                    self._started = True
                continue

            if rms < self._threshold():
                self._pending.append(w)
            else:
                out.extend(self._release_pending())
                out.append(w)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int16)

    def flush(self) -> np.ndarray:
        """結尾的靜音原樣送出 —— 使用者要修的是「中間」的停頓。"""
        out = list(self._pending)
        self._pending.clear()
        if len(self._carry):
            out.append(self._carry)
            self._carry = np.empty(0, dtype=np.int16)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int16)


def limit_stream(
    chunks: Iterator[tuple[int, np.ndarray]],
    max_gap_ms: int,
    kinds: list[str] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """包住 synthesize() 的產生器，把 zh2en 邊界上過長的靜音縮到上限。"""
    limiter: GapLimiter | None = None
    sr = 0
    for sr, chunk in chunks:
        if limiter is None:
            limiter = GapLimiter(sr, max_gap_ms=max_gap_ms, kinds=kinds)
        out = limiter.feed(chunk)
        if len(out):
            yield sr, out
    if limiter is not None:
        tail = limiter.flush()
        if len(tail):
            yield sr, tail
        if limiter.trimmed_samples:
            log.info(
                "剪掉 %.0f ms 的多餘停頓（中英邊界，句中無標點）",
                limiter.trimmed_samples / limiter.sr * 1000,
            )
