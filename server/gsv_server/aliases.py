"""特殊符號代換：把文字裡的符號換成別的文字，再送進前端。

為什麼需要：`chinese2.py:68` 的 replace_punctuation() 把漢字與標點以外的字元
全部刪掉 —— 注音符號、表情符號、各種記號都在內。「好ㄟ」送進去只剩「好」，
而且沒有任何提示。

做法是最單純的文字取代：合成前把符號換成一段唸得出來的文字（`ㄟ` -> `欸`、
`ㄏㄏ` -> `呵呵`）。換完之後它就是普通的文字，讀音照一般流程走，想調聲調
就到發音字典改那個字（例如 `欸 ei1`）。

選代換用的字時要避開語氣詞：`tone_sandhi.py:506` 把「吧呢哈啊呐噻嘛吖嗨哦哒
额滴哩哟喽啰耶喔诶」這些字一律改成輕聲，自訂的聲調會被蓋掉。所以 ㄟ 用
「欸」而不是「誒」。

這張表存在 `server/aliases.rep`，一行一筆：`符號 代換文字`。檔案不存在時用
內建的預設值；第一次修改時才寫檔。可以用控制台的「特殊符號」編輯。
"""

from __future__ import annotations

import threading
from pathlib import Path

from .bootstrap import VENDOR_ROOT

DEFAULT_PATH = VENDOR_ROOT / "aliases.rep"

# 檔案不存在時的內容
DEFAULTS: dict[str, str] = {"ㄟ": "欸"}

HEADER = """\
# 特殊符號代換 —— 本專案自己的檔案，不是 vendored upstream
#
# 一行一筆：符號 代換文字
# 合成前會把文字裡的「符號」換成「代換文字」。符號不能含空白。
# 可以用控制台的「特殊符號」編輯，也可以直接手改（改完重啟 server）。
"""


class AliasError(RuntimeError):
    pass


_lock = threading.Lock()
_table: dict[str, str] | None = None


def _check(symbol: str, text: str, source: str = "") -> None:
    where = f"{source} " if source else ""
    if not symbol:
        raise AliasError(f"{where}符號不能是空的")
    if any(ch.isspace() for ch in symbol):
        raise AliasError(f"{where}符號 {symbol!r} 不能含空白")
    if symbol.startswith("#"):
        raise AliasError(f"{where}符號不能以 # 開頭（那是註解）")
    if not text:
        raise AliasError(f"{where}{symbol!r} 的代換文字不能是空的")
    if symbol in text:
        raise AliasError(f"{where}{symbol!r} 的代換文字裡不能再出現它自己")


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


def table() -> dict[str, str]:
    """目前生效的代換表（副本）。第一次呼叫時讀檔。"""
    global _table
    with _lock:
        if _table is None:
            _table = load()
        return dict(_table)


def upsert(symbol: str, text: str) -> dict[str, str]:
    """新增或修改一筆，寫檔並立刻生效。"""
    global _table
    symbol, text = symbol.strip(), text.strip()
    _check(symbol, text)
    with _lock:
        current = dict(_table) if _table is not None else load()
        current[symbol] = text
        _save(current)
        _table = current
        return dict(current)


def remove(symbol: str) -> dict[str, str]:
    global _table
    symbol = symbol.strip()
    with _lock:
        current = dict(_table) if _table is not None else load()
        if symbol not in current:
            raise AliasError(f"代換表裡沒有 {symbol!r}")
        del current[symbol]
        _save(current)
        _table = current
        return dict(current)


def apply(text: str) -> str:
    """把文字裡的符號換掉。長的符號先換，`ㄏㄏ` 才不會被 `ㄏ` 搶先拆開。"""
    current = table()
    for symbol in sorted(current, key=len, reverse=True):
        if symbol in text:
            text = text.replace(symbol, current[symbol])
    return text


# ---- 用拼音指定讀音 ----

# tone_sandhi.py:506 一律改成輕聲的字。拿來當代換字的話，指定的聲調會被蓋掉。
_FORCED_NEUTRAL = set("吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶")

_index: dict[str, list[str]] | None = None
_freq: dict[str, int] = {}

# 借用一個字並改它的讀音時，只借罕用字：那筆自訂讀音對所有句子都生效，
# 借到常用字（例如把「哪」改成 ne5）會把別的句子唸壞。
_BORROW_MAX_FREQ = 300


def looks_like_pinyin(text: str) -> bool:
    """整段都是「小寫字母 + 聲調數字」的音節（he1 he1）才算。"""
    tokens = text.split()
    return bool(tokens) and all(
        t[-1:] in "12345" and t[:-1].isalpha() and t[:-1].isascii() and t.islower()
        for t in tokens
    )


def looks_like_failed_pinyin(text: str) -> bool:
    """像拼音但不合格：小寫字母後面接了數字，可是不是 1-5（he9、ei0）。"""
    import re

    tokens = text.split()
    return (
        bool(tokens)
        and all(re.fullmatch(r"[a-z]+\d+", t) for t in tokens)
        and not looks_like_pinyin(text)
    )


def _char_index() -> dict[str, list[str]]:
    """拼音（帶調）-> 讀這個音的常用字，最常用的排前面。第一次用時建立。"""
    global _index
    if _index is not None:
        return _index
    from pypinyin import Style, pinyin

    try:
        import jieba_fast as jieba
    except Exception:  # noqa: BLE001
        import jieba
    jieba.initialize()
    freq = getattr(jieba.dt, "FREQ", {}) or {}
    index: dict[str, list[tuple[int, int, str]]] = {}
    for code in range(0x4E00, 0x9FA6):
        char = chr(code)
        readings = pinyin(char, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        for rank, reading in enumerate(readings):
            # 主要讀音排在破音讀法前面，同一級再照詞頻排
            index.setdefault(reading, []).append((rank, -freq.get(char, 0), char))
    _index = {k: [c for _, _, c in sorted(v)] for k, v in index.items()}
    _freq.update({c: freq.get(c, 0) for v in _index.values() for c in v})
    return _index


def resolve_pinyin(text: str, version: str) -> dict:
    """把一串拼音變成「唸出來正好是這串拼音」的漢字。

    使用者想的是「這個符號唸 he1 he1」，但代換表存的是文字。這裡替每個音節
    挑一個字，然後用**真正的前端**唸一次確認結果相符 —— 不相符的字不用。

    找不到天生就唸這個音的字時（例如 ei1），退而求其次：挑一個同音節的字，
    在發音字典裡替它加一筆自訂讀音。那會改到這個字在所有句子裡的讀音，
    所以會在回傳的 notes 裡講明。

    回傳 {"text": 漢字, "phones": 實際音素, "expected": 想要的音素,
          "exact": 是否相符, "notes": [...]}。
    """
    from text.cleaner import clean_text

    from . import pronunciation

    def speak(chars: str) -> list[str]:
        phones = clean_text(chars, "zh", version)[0]
        return [p for p in phones if p[:1].isalnum()]

    index = _char_index()
    # 已經有自訂讀音的字不借：會把使用者原本的設定蓋掉
    taken = set(pronunciation.load().get("zh", {}))
    notes: list[str] = []
    chosen: list[str] = []
    expected: list[str] = []
    for syllable in text.split():
        want = pronunciation.pinyin_to_phones(syllable)
        if want is None:
            raise AliasError(f"{syllable!r} 不是合法的拼音音節（要像 he1、ei4 這樣，聲調 1-5）")
        expected.extend(want)
        picked = None
        for char in index.get(syllable, [])[:40]:
            if char in _FORCED_NEUTRAL or char in SKIP_CARRIERS:
                continue
            if speak(char) == want:
                picked = char
                break
        if picked is None:
            # 沒有天生唸這個音的字：借一個同音節的罕用字，替它加自訂讀音
            base = syllable[:-1]
            pool = {c for tone in "12345" for c in index.get(base + tone, [])}
            pool = sorted(
                (c for c in pool if _freq.get(c, 0) <= _BORROW_MAX_FREQ),
                key=lambda c: (-_freq.get(c, 0), c),
            )
            for char in pool:
                if char in _FORCED_NEUTRAL or char in SKIP_CARRIERS:
                    continue
                if pronunciation.normalize_word("zh", char) in taken:
                    continue
                try:
                    pronunciation.upsert("zh", char, [syllable])
                except pronunciation.PronunciationError:
                    continue
                if speak(char) == want:
                    picked = char
                    notes.append(
                        f"沒有天生唸 {syllable} 的字，所以借用罕用字「{char}」並在發音字典加了一筆"
                        f"「{char} {syllable}」"
                    )
                    break
                pronunciation.remove("zh", char)
        if picked is None:
            raise AliasError(
                f"找不到能唸成 {syllable} 的字。可以改填一個讀音相近的漢字，"
                f"再到發音字典調它的聲調"
            )
        chosen.append(picked)

    result = "".join(chosen)
    phones = speak(result)
    if phones != expected:
        # 每個字單獨唸都對，連在一起不一樣：國語的連讀變調（兩個三聲相連，
        # 前一個變二聲…）。那是 upstream 照規則做的，真人也這樣唸，不去擋它。
        notes.append(
            "連在一起唸時，聲調照國語的變調規則變了（例如兩個三聲相連，前一個唸二聲）"
        )
    return {
        "text": result,
        "phones": phones,
        "expected": expected,
        "exact": phones == expected,
        "notes": notes,
    }


# 不拿來當代換字的字：一和不有自己的變調規則，讀音會隨後面的字改變
SKIP_CARRIERS = set("一不")
