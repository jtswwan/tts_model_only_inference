"""特殊符號：讓使用者決定某個符號要發什麼音。

為什麼需要：`chinese2.py:68` 的 replace_punctuation() 把漢字與標點以外的字元
全部刪掉 —— 注音符號、表情符號、各種記號都在內。「好ㄟ」送進去只剩「好」，
而且沒有任何提示。

一個符號可以對應兩種東西：

  拼音 / 音素（`ㄏㄏ he1 he1`）
      直接把這串音素送進模型，不經過任何漢字或英文單字。所以沒有對應漢字的
      音也能用，聲母和韻母還可以自由組合（`f i1`、`EE E1`）。
      不套用變調：填幾聲就是幾聲。

  文字（`ㄟ 欸`）
      合成前把符號換成這段文字，之後照一般文字處理。

這張表存在 `server/aliases.rep`，一行一筆：`符號 內容`。檔案不存在時用內建的
預設值；第一次修改時才寫檔。可以用控制台的「特殊符號」編輯。

直接送音素是怎麼做到的
----------------------
upstream 的前端把文字切成中文片段與英文片段，每個片段各自呼叫
`TextPreprocessor.clean_text_inf()` 轉成音素（TextPreprocessor.py:175-180）。
英文片段不算 BERT 特徵，`get_bert_inf()` 直接給零（:216-220）。

所以：合成前把符號換成一個不可能出現在正常文字裡的假英文單字（佔位詞），
它會被切成英文片段；再包住 `clean_text_inf()`，看到佔位詞就直接回傳使用者
指定的音素，不去問英文的 g2p。BERT 特徵照英文片段的規則是零，和夾在中文裡的
英文單字一樣。

沒有借用英文字典（pronunciation.rep 的 [en]），也沒有改 upstream 的檔案。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable

from .bootstrap import VENDOR_ROOT

DEFAULT_PATH = VENDOR_ROOT / "aliases.rep"

# 檔案不存在時的內容
DEFAULTS: dict[str, str] = {"ㄟ": "ei1"}

HEADER = """\
# 特殊符號 —— 本專案自己的檔案，不是 vendored upstream
#
# 一行一筆：符號 內容
#   內容是帶聲調的拼音（he1 he1）或音素（f i1）-> 直接發這串音
#   內容是文字（呵呵）                         -> 合成前換成這段文字
# 符號不能含空白。可以用控制台的「特殊符號」編輯，也可以直接手改
# （改完重啟 server）。
"""

# 佔位詞：gsvsym + 小寫字母編號。全小寫 ASCII 字母，LangSegmenter 才會把它判成
# 英文、而且不會從中間拆開。
_PLACEHOLDER_RE = re.compile(r"gsvsym[a-z]+")


class AliasError(RuntimeError):
    pass


_lock = threading.Lock()
_table: dict[str, str] | None = None
# 佔位詞 -> (符號, 音素)；每次表變動時重建
_placeholders: dict[str, tuple[str, list[str]]] = {}
_symbol_to_placeholder: dict[str, str] = {}


def _check(symbol: str, text: str, source: str = "") -> None:
    where = f"{source} " if source else ""
    if not symbol:
        raise AliasError(f"{where}符號不能是空的")
    if any(ch.isspace() for ch in symbol):
        raise AliasError(f"{where}符號 {symbol!r} 不能含空白")
    if symbol.startswith("#"):
        raise AliasError(f"{where}符號不能以 # 開頭（那是註解）")
    if not text:
        raise AliasError(f"{where}{symbol!r} 的內容不能是空的")
    if symbol in text:
        raise AliasError(f"{where}{symbol!r} 的內容裡不能再出現它自己")
    if _looks_like_failed_pinyin(text):
        raise AliasError(
            f"{where}{text!r} 看起來像拼音但聲調不對。聲調要 1-5"
            f"（例如 he1、ei4；5 是輕聲）"
        )


def _looks_like_failed_pinyin(text: str) -> bool:
    """像拼音但不合格：小寫字母後面接了數字，卻解不成音素（he9、xq1）。"""
    tokens = text.split()
    return (
        bool(tokens)
        and all(re.fullmatch(r"[a-z]+\d+", t) for t in tokens)
        and phones_of(text) is None
    )


def phones_of(content: str) -> list[str] | None:
    """內容若是拼音 / 音素就回傳音素清單，是一般文字就回 None。

    判定條件：每個 token 都是帶聲調的拼音或合法音素，而且至少有一個帶聲調 ——
    否則單獨一個 `a` 也會被當成音素（它剛好是合法的聲母符號）。
    """
    tokens = content.split()
    if not tokens or not any(t[-1:].isdigit() for t in tokens):
        return None
    from text.symbols2 import symbols

    from . import pronunciation

    known = set(symbols)
    phones = pronunciation.expand_en_value(tokens)
    if all(p in known for p in phones):
        return phones
    return None


def parse(content: str, source: str = "<string>") -> dict[str, str]:
    table: dict[str, str] = {}
    for lineno, raw in enumerate(content.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        symbol, text = parts[0], (parts[1].strip() if len(parts) > 1 else "")
        _check(symbol, text, f"{source}:{lineno}")
        if symbol in table:
            raise AliasError(f"{source}:{lineno} {symbol!r} 重複定義")
        table[symbol] = text
    return table


def load(path: Path | None = None) -> dict[str, str]:
    path = DEFAULT_PATH if path is None else Path(path)
    if not path.exists():
        return dict(DEFAULTS)
    return parse(path.read_text(encoding="utf-8"), str(path))


def _save(table: dict[str, str], path: Path | None = None) -> None:
    path = DEFAULT_PATH if path is None else Path(path)
    lines = [HEADER] + [f"{symbol} {table[symbol]}" for symbol in sorted(table)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def _letters(index: int) -> str:
    """0 -> a、25 -> z、26 -> ba …（只用字母，佔位詞裡不能有數字）"""
    out = ""
    while True:
        index, rem = divmod(index, 26)
        out = chr(ord("a") + rem) + out
        if index == 0:
            return out


def _install(table: dict[str, str]) -> None:
    """換上新的表，並重建佔位詞。呼叫端要先拿 _lock。"""
    global _table
    placeholders: dict[str, tuple[str, list[str]]] = {}
    by_symbol: dict[str, str] = {}
    for index, symbol in enumerate(sorted(table)):
        phones = phones_of(table[symbol])
        if phones is not None:
            word = "gsvsym" + _letters(index)
            placeholders[word] = (symbol, phones)
            by_symbol[symbol] = word
    _table = dict(table)
    _placeholders.clear()
    _placeholders.update(placeholders)
    _symbol_to_placeholder.clear()
    _symbol_to_placeholder.update(by_symbol)


def _current() -> dict[str, str]:
    if _table is None:
        _install(load())
    return _table  # type: ignore[return-value]


def table() -> dict[str, str]:
    """目前生效的表（副本）。第一次呼叫時讀檔。"""
    with _lock:
        return dict(_current())


def describe() -> list[dict[str, Any]]:
    """給控制台用：每個符號的內容，以及它是直接發音還是換成文字。"""
    with _lock:
        current = dict(_current())
        direct = {sym: _placeholders[word][1] for sym, word in _symbol_to_placeholder.items()}
    return [
        {
            "symbol": symbol,
            "content": current[symbol],
            "kind": "phones" if symbol in direct else "text",
            "phones": direct.get(symbol),
        }
        for symbol in sorted(current)
    ]


def upsert(symbol: str, text: str) -> dict[str, str]:
    """新增或修改一筆，寫檔並立刻生效。"""
    symbol, text = symbol.strip(), " ".join(text.split())
    _check(symbol, text)
    with _lock:
        current = dict(_current())
        current[symbol] = text
        _save(current)
        _install(current)
        return dict(current)


def remove(symbol: str) -> dict[str, str]:
    symbol = symbol.strip()
    with _lock:
        current = dict(_current())
        if symbol not in current:
            raise AliasError(f"表裡沒有 {symbol!r}")
        del current[symbol]
        _save(current)
        _install(current)
        return dict(current)


def apply(text: str, *, for_lookup: bool = False) -> str:
    """把文字裡的符號換掉。長的符號先換，`ㄏㄏ` 才不會被 `ㄏ` 搶先拆開。

    直接發音的符號換成佔位詞（前後補空白，免得和相鄰的英文字黏在一起）；
    文字型的換成那段文字。

    `for_lookup=True` 給發音字典的查詢用：直接發音的符號整個拿掉 —— 它不經過
    字典，查詢結果裡出現佔位詞只會讓人困惑。
    """
    with _lock:
        current = dict(_current())
        by_symbol = dict(_symbol_to_placeholder)
    for symbol in sorted(current, key=len, reverse=True):
        if symbol not in text:
            continue
        if symbol in by_symbol:
            text = text.replace(symbol, " " if for_lookup else f" {by_symbol[symbol]} ")
        else:
            text = text.replace(symbol, current[symbol])
    return text


def wrap_clean_text_inf(original: Callable) -> Callable:
    """包住 `TextPreprocessor.clean_text_inf`，讓佔位詞直接變成指定的音素。

    只處理「英文片段裡有佔位詞」的情況，其他一律原樣交給 upstream。

    片段裡佔位詞以外的部分：
      - 有英文字母或數字的：是真的英文，交給 upstream 的英文前端
      - 只有標點的：照 `chinese2.rep_map` 轉成標點音素。交給英文前端的話，
        全形標點會被丟掉（「ㄟ，你好」的逗號就不見了）
    """

    def wrapped(text: str, language: str, version: str = "v2"):
        if language.replace("all_", "") != "en" or not _PLACEHOLDER_RE.search(text):
            return original(text, language, version)

        from text import cleaned_text_to_sequence
        from text.chinese2 import rep_map
        from text.symbols2 import punctuation

        with _lock:
            placeholders = dict(_placeholders)

        ids: list[int] = []
        norm: list[str] = []
        pos = 0
        pieces: list[tuple[str, bool]] = []
        for match in _PLACEHOLDER_RE.finditer(text):
            pieces.append((text[pos:match.start()], False))
            pieces.append((match.group(), True))
            pos = match.end()
        pieces.append((text[pos:], False))

        for piece, is_placeholder in pieces:
            if is_placeholder:
                entry = placeholders.get(piece)
                if entry is None:
                    # 表在這段文字換好佔位詞之後被改掉了。當作這個符號不存在。
                    continue
                symbol, phones = entry
                ids.extend(cleaned_text_to_sequence(phones, version))
                norm.append(symbol)
            elif re.search(r"[A-Za-z0-9]", piece):
                piece_ids, _, piece_norm = original(piece, language, version)
                ids.extend(piece_ids)
                norm.append(piece_norm)
            else:
                marks = [rep_map.get(ch, ch) for ch in piece]
                marks = [m for m in marks if m in punctuation]
                if marks:
                    ids.extend(cleaned_text_to_sequence(marks, version))
                    norm.append("".join(marks))
        # 英文片段不用 word2ph（get_bert_inf 對非中文直接給零）
        return ids, None, "".join(norm)

    return wrapped
