#!/usr/bin/env python3
"""看一個英文詞（或一整句）實際會被唸成什麼音素。

為什麼需要這支：`text/english.py:322` 對「不在字典裡、長度 <= 3」的詞一律
逐字母拼讀。那對縮寫是對的（GPU 唸 G-P-U），對剛好能當單字唸的詞就錯了
（wav 被唸成 "double-u ay vee"）。這支會直接標出哪些詞被逐字母拼讀，
不用靠耳朵猜。

    python3 scripts/check_en.py wav gpu api        # 查幾個詞
    python3 scripts/check_en.py "the wav file"     # 查一整句
    python3 scripts/check_en.py --spelled wav api  # 只列出被逐字母拼讀的

要修某個詞的發音：把它加到 server/pronunciation.rep，然後重啟 server。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from gsv_server import bootstrap  # noqa: E402

bootstrap.setup()

from gsv_server import pronunciation  # noqa: E402


def letter_by_letter(g2p_obj, word: str) -> list[str]:
    """重現 qryword() 的逐字母分支，用來判斷某個詞是不是走了那條路。"""
    phones: list[str] = []
    for ch in word.lower():
        if ch == "a":
            phones.extend(["EY1"])
        elif not ch.isalpha():
            phones.append(ch)
        else:
            phones.extend(g2p_obj.cmu[ch][0])
    return phones


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="+", help="要查的詞或句子")
    ap.add_argument("--spelled", action="store_true",
                    help="只列出被逐字母拼讀的詞")
    ap.add_argument("--no-custom", action="store_true",
                    help="不套用 server/pronunciation.rep，看 upstream 原本的結果")
    args = ap.parse_args()

    from text.english import _g2p, text_normalize

    applied: dict[str, list[str]] = {}
    if not args.no_custom:
        applied = pronunciation.load()
        pronunciation.apply(applied)
        if applied:
            print(f"已套用 server/pronunciation.rep：{', '.join(sorted(applied))}\n")

    # 把句子拆成詞；單一參數含空白時當句子處理
    words: list[str] = []
    for item in args.words:
        words.extend(text_normalize(item).split() if " " in item else [item])

    rows = []
    for w in words:
        clean = w.strip(".,!?;:\"'()")
        if not clean or not any(c.isalpha() for c in clean):
            continue
        phones = _g2p.qryword(clean)
        spelled = phones == letter_by_letter(_g2p, clean)
        custom = clean.lower() in applied
        rows.append((clean, phones, spelled, custom))

    if args.spelled:
        rows = [r for r in rows if r[2]]
        if not rows:
            print("沒有詞被逐字母拼讀。")
            return 0

    width = max((len(r[0]) for r in rows), default=4) + 2
    print(f"{'詞'.ljust(width)}{'音素'.ljust(44)}判斷")
    print("-" * (width + 58))
    for word, phones, spelled, custom in rows:
        tag = "自訂發音" if custom else ("逐字母拼讀 ←" if spelled else "整字")
        print(f"{word.ljust(width)}{' '.join(phones).ljust(44)}{tag}")

    if any(r[2] and not r[3] for r in rows):
        print()
        print("被標記的詞走了 english.py:322 的「長度 <= 3 就逐字母」分支。")
        print("如果那個詞其實該當單字唸，把它加進 server/pronunciation.rep。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
