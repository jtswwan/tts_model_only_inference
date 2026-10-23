"""自訂英文發音：覆寫 g2p_en 的查字典結果。

為什麼需要
----------
`text/english.py:322` 的 `qryword()` 對「不在 cmudict 裡、長度 <= 3」的詞
一律逐字母拼讀：

    # oov 长度小于等于 3 直接读字母
    if len(word) <= 3:

對縮寫這是對的（GPU 唸 G-P-U、PNG 唸 P-N-G、TTS 唸 T-T-S），但對剛好能當
單字唸的詞就錯了 —— `wav` 會變成 `D AH1 B AH0 L Y UW0 EY1 V IY1`，
也就是 "double-u ay vee"。

為什麼不改 upstream
-------------------
upstream 自己留了 `text/engdict-hot.rep` 這個熱修字典（裡面已經有 CHATGPT、
JSON、CONDA），`get_dict()` 會在讀完 cache 之後無條件套用它。但那個檔案在
vendored 樹裡、被 `scripts/manifest.txt` 管著，`vendor_upstream.sh` 會重新
複製，手改會被蓋掉。

所以這裡用同樣的手法，但資料放在 `server/pronunciation.rep`（本專案自己的
檔案），在暖機時合併進 `_g2p.cmu`。效果和 upstream 的熱修字典完全相同，
而且 upstream 的程式碼一個字都不用改。

為什麼一定要驗證
----------------
`replace_phs()`（english.py:111-121）對不在 `symbols` 裡的音素是**靜默丟掉**
的，只 print 一行 "ph not in symbols"。所以一個打錯的音素不會報錯，只會讓
那個字少一個音 —— 這種無聲失敗正是本專案要避免的，因此載入時就擋掉。
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import bootstrap

log = logging.getLogger("gsv.pronunciation")

# server/pronunciation.rep —— 在 vendored 樹之外，不受 vendor_upstream.sh 影響
DEFAULT_PATH = bootstrap.VENDOR_ROOT / "pronunciation.rep"


class PronunciationError(RuntimeError):
    pass


def parse(text: str, source: str = "<string>") -> dict[str, list[str]]:
    """解析 `.rep` 格式。格式錯誤就失敗，不猜。"""
    entries: dict[str, list[str]] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise PronunciationError(
                f"{source}:{lineno} 格式錯誤：至少要有「單字 音素」兩欄 —— {raw!r}"
            )
        word, phones = parts[0].lower(), parts[1:]
        if word in entries:
            raise PronunciationError(f"{source}:{lineno} 單字 {word!r} 重複定義")
        entries[word] = phones
    return entries


def validate(entries: dict[str, list[str]], source: str = "<string>") -> None:
    """音素必須都在 symbols2 裡，否則 replace_phs() 會靜默丟掉。"""
    bootstrap.require_setup()
    from text.symbols2 import symbols

    known = set(symbols)
    problems: list[str] = []
    for word, phones in entries.items():
        bad = [p for p in phones if p not in known]
        if bad:
            problems.append(f"{word}: {' '.join(bad)}")
    if problems:
        raise PronunciationError(
            f"{source} 裡有 text/symbols2.py 不認識的音素，載入會導致那些音被"
            f"靜默丟掉：\n  - " + "\n  - ".join(problems)
        )


def load(path: Path | None = None) -> dict[str, list[str]]:
    """讀取並驗證自訂發音。檔案不存在就回空的 —— 這個檔案是選用的。"""
    path = DEFAULT_PATH if path is None else Path(path)
    if not path.exists():
        return {}
    entries = parse(path.read_text(encoding="utf-8"), str(path))
    validate(entries, str(path))
    return entries


def apply(entries: dict[str, list[str]]) -> int:
    """合併進 `_g2p.cmu`，回傳套用的筆數。

    做法和 upstream 的 `hot_reload_hot()` 一樣：直接覆寫字典條目。
    `qryword()` 的第一步就是查 `self.cmu`（english.py:312），所以查得到的
    字根本不會走到那個「長度 <= 3 就逐字母」的分支。
    """
    if not entries:
        return 0
    bootstrap.require_setup()
    from text import english

    for word, phones in entries.items():
        english._g2p.cmu[word] = [phones]
    log.info("已套用 %d 筆自訂發音：%s", len(entries), ", ".join(sorted(entries)))
    return len(entries)


def load_and_apply(path: Path | None = None) -> dict[str, list[str]]:
    entries = load(path)
    apply(entries)
    return entries
