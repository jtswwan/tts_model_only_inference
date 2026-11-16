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
