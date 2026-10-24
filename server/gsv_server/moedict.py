"""教育部《重編國語辭典修訂本》查詢。

為什麼需要
----------
整個中文前端是**普通話**的：g2pw 用大陸語料訓練、`text/g2pw/polyphonic.rep`
的 45,046 筆是大陸讀音、`chinese2.text_normalize()` 還會把繁體轉簡體。
所以台／陸讀音不同的詞，一律以大陸讀音輸出。

實測 `scripts/tw_check.py --scan-upstream`：那 45,046 筆裡有 **2,703 筆**
與教育部不同。

授權
----
辭典正文著作權屬教育部，JSON 轉換出自 g0v 萌典，採
**CC BY-ND 3.0 TW**：可原封不動重製散布、可商用，但**不得改作**
（含節錄重組）。

所以這裡：**不修改那個檔案，也不從它產生任何衍生資料檔**，只在需要時
載入記憶體查詢。完整標示見 `moedict/來源與授權.md`。

載入成本
--------
14 MB 的 .xz、159,777 條，解壓與建索引約數秒、記憶體數百 MB。所以是
**惰性載入** —— 只有控制台真的按下「查教育部」時才載，不碰它的人不付這筆。
"""

from __future__ import annotations

import json
import logging
import lzma
import threading
from pathlib import Path

from . import bootstrap

log = logging.getLogger("gsv.moedict")

# repo 根的 moedict/（可攜版裡是 app/moedict/）
DATA = bootstrap.VENDOR_ROOT.parent / "moedict" / "dict-revised.json.xz"
DOWNLOAD_HINT = "python3 scripts/tw_check.py --download"

_index: dict[str, list[str]] | None = None
_lock = threading.Lock()


def available() -> bool:
    return DATA.is_file()


def _to_tone3(marked: str) -> list[str] | None:
    """「dǎng àn」-> ['dang3', 'an4']。"""
    from pypinyin.contrib.tone_convert import to_tone3

    out: list[str] = []
    for syl in marked.split():
        try:
            converted = to_tone3(syl, v_to_u=False)
        except Exception:  # noqa: BLE001
            return None
        if not converted:
            return None
        if converted[-1] not in "12345":
            converted += "5"  # 輕聲
        out.append(converted)
    return out or None


def load() -> dict[str, list[str]] | None:
    """載入辭典。多執行緒安全，只載一次。檔案不在就回 None。"""
    global _index
    if _index is not None:
        return _index
    if not available():
        return None
    with _lock:
        if _index is not None:
            return _index
        log.info("載入教育部辭典 %s", DATA)
        with lzma.open(DATA) as f:
            data = json.load(f)
        idx: dict[str, list[str]] = {}
        for entry in data:
            title = entry.get("title")
            if not title:
                continue
            readings = [
                h["pinyin"] for h in entry.get("heteronyms", []) if h.get("pinyin")
            ]
            if readings:
                idx[title] = readings
        _index = idx
        log.info("教育部辭典就緒：%d 條", len(idx))
    return _index


def lookup(word: str) -> list[list[str]] | None:
    """回傳教育部的讀音（可能多個異讀，各自是拼音陣列）。查不到回 None。

    `word` 用**繁體** —— 辭典是繁體的。管線內部用簡體，所以呼叫端要自己
    拿原始的繁體字串來查，不要拿 `text_normalize()` 之後的。
    """
    idx = load()
    if not idx:
        return None
    readings = idx.get(word)
    if not readings:
        return None
    out = [r for r in (_to_tone3(x) for x in readings) if r]
    return out or None


# 辭典是繁體，但管線內部的詞是簡體（`chinese2.text_normalize()` 會做繁簡
# 轉換）。所以另外建一份「簡體 -> (繁體, 讀音)」的對照，才查得到。
# 實測 143,993 個 2-4 字詞約 3 秒，和載入辭典本身差不多，一樣惰性只做一次。
#
# 這份對照只存在記憶體裡，不寫成檔案 —— ND 不允許從辭典產生衍生資料檔。
_simplified: dict[str, tuple[str, list[list[str]]]] | None = None


def simplified_index() -> dict[str, tuple[str, list[list[str]]]] | None:
    global _simplified
    if _simplified is not None:
        return _simplified
    idx = load()
    if not idx:
        return None
    with _lock:
        if _simplified is not None:
            return _simplified
        bootstrap.require_setup()
        from text.chinese2 import text_normalize

        out: dict[str, tuple[str, list[list[str]]]] = {}
        for trad, readings in idx.items():
            if not (0 < len(trad) <= 4):
                continue
            if not all("一" <= c <= "鿿" for c in trad):
                continue
            try:
                key = text_normalize(trad)
            except Exception:  # noqa: BLE001
                continue
            if len(key) != len(trad) or key in out:
                continue
            tone3 = [r for r in (_to_tone3(x) for x in readings) if r]
            if tone3:
                out[key] = (trad, tone3)
        _simplified = out
        log.info("教育部辭典簡體對照就緒：%d 條", len(out))
    return _simplified


def lookup_any(word: str) -> tuple[str, list[list[str]]] | None:
    """用繁體或簡體都查得到。回傳 (辭典裡的繁體詞, 讀音清單)。"""
    idx = load()
    if not idx:
        return None
    readings = idx.get(word)
    if readings:
        tone3 = [r for r in (_to_tone3(x) for x in readings) if r]
        if tone3:
            return word, tone3
    simp = simplified_index()
    return simp.get(word) if simp else None
