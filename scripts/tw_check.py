#!/usr/bin/env python3
"""拿教育部國語辭典的讀音，比對這個 server 現在實際會唸什麼。

為什麼需要
----------
整個中文前端是**普通話**的：g2pw 用大陸語料訓練、`text/g2pw/polyphonic.rep`
的 45,046 筆是大陸讀音、`chinese2.text_normalize()` 還會把繁體轉成簡體。
所以台／陸讀音不同的詞，一律以大陸讀音輸出。

常見的例子（都實測過）：

    詞      server 現在      教育部
    檔案    dang3 an4        dang3 an4   （這個剛好一樣）
    垃圾    la1 ji1          le4 se4
    企業    qi3 ye4          qi4 ye4
    星期    xing1 qi1        xing1 qi2

與其憑印象猜，這支直接查**萌典**（moedict，g0v 把教育部國語辭典轉成 JSON
的開放專案）的公開 API，逐詞比對。

授權
----
辭典正文的著作權屬教育部，採「創用 CC 姓名標示-禁止改作 臺灣 3.0」。
所以這支是**逐詞查詢**，不把資料集打包進專案；查到的讀音只是拿來當你
自己判斷的依據。要離線用的話請自行評估授權。

用法
----
    python3 scripts/tw_check.py 檔案 垃圾 企業        # 查幾個詞
    python3 scripts/tw_check.py --text "我把檔案存成 wav"  # 從句子抓詞來查
    python3 scripts/tw_check.py --diff-only 垃圾 企業 資料  # 只列出不一樣的
    python3 scripts/tw_check.py --apply --diff-only 垃圾 企業  # 直接寫進字典
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))

from gsv_server import bootstrap  # noqa: E402

bootstrap.setup()

from gsv_server import pronunciation as P  # noqa: E402

DATA_URL = ("https://raw.githubusercontent.com/g0v/moedict-data/master/"
            "dict-revised.json.xz")
DATA = REPO / "moedict" / "dict-revised.json.xz"
API = "https://www.moedict.tw/uni/{}.json"
CACHE = REPO / ".moedict-cache.json"
OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def download() -> int:
    """原封不動抓下整份辭典。授權見 moedict/來源與授權.md。"""
    import urllib.request

    DATA.parent.mkdir(parents=True, exist_ok=True)
    print(f"下載 {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA)
    print(f"{OK} {DATA}（{DATA.stat().st_size / 1e6:.1f} MB）")
    print("  教育部《重編國語辭典修訂本》／g0v 萌典")
    print("  CC BY-ND 3.0 TW —— 可原封不動重製散布，不得改作")
    return 0


def load_cache() -> dict:
    """公開 API 模式用的小快取。有本機辭典時用不到。"""
    if CACHE.is_file():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


_index: dict[str, list[str]] | None = None


def local_index() -> dict[str, list[str]] | None:
    """把辭典載入記憶體。**不產生任何衍生檔案** —— ND 不允許節錄重組。"""
    global _index
    if _index is not None:
        return _index
    if not DATA.is_file():
        return None
    import lzma

    with lzma.open(DATA) as f:
        data = json.load(f)
    idx: dict[str, list[str]] = {}
    for entry in data:
        title = entry.get("title")
        if not title:
            continue
        readings = [h["pinyin"] for h in entry.get("heteronyms", []) if h.get("pinyin")]
        if readings:
            idx[title] = readings
    _index = idx
    return idx


def moedict(word: str, cache: dict, delay: float = 0.4) -> list[str] | None:
    """教育部的讀音（可能多個異讀）。有本機辭典就用本機，否則退回公開 API。"""
    idx = local_index()
    if idx is not None:
        return idx.get(word)

    if word in cache:
        return cache[word]
    url = API.format(urllib.parse.quote(word))
    try:
        time.sleep(delay)  # 公開 API，別打太兇
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
    except Exception:  # noqa: BLE001
        cache[word] = None
        return None
    out = [h["pinyin"] for h in data.get("heteronyms", []) if h.get("pinyin")]
    cache[word] = out or None
    return cache[word]


def to_tone3(marked: str) -> list[str] | None:
    """「dǎng àn」-> ['dang3', 'an4']。"""
    from pypinyin.contrib.tone_convert import to_tone3 as _t3

    out = []
    for syl in marked.split():
        try:
            converted = _t3(syl, v_to_u=False)
        except Exception:  # noqa: BLE001
            return None
        if not converted or converted[-1] not in "12345":
            converted = converted + "5"  # 輕聲
        out.append(converted)
    return out


def current_reading(word: str) -> list[str]:
    """這個 server 現在實際會唸什麼。"""
    items = P.lookup(word)["items"]
    out: list[str] = []
    for it in items:
        out.extend(it.get("pinyin") or [])
    return out


def scan_upstream(limit: int) -> int:
    """把 upstream 的多音字表整個比對教育部。

    `text/g2pw/polyphonic.rep` 是大陸讀音，45,046 筆。這裡列出其中與教育部
    不同的，讓你自己決定要不要改 —— **不會自動套用**。實測 2,703 筆不同，
    全部套用是不對的：裡面很多是教育部標準但台灣口語不這樣唸的（例如
    說服 shui4 fu2、滑稽 gu3 ji1），改了反而更怪。

    建議的做法是 `--text "你的稿子"` —— 只看實際會用到的詞。
    """
    idx = local_index()
    if idx is None:
        print(f"{BAD} 需要本機辭典，先跑 --download")
        return 1
    from text.chinese2 import text_normalize
    from text.g2pw import g2pw as g2pw_mod

    # moedict 是繁體、pp_dict 是簡體，用管線自己的 normalize 對齊
    simp: dict[str, tuple[str, list[str]]] = {}
    for trad, readings in idx.items():
        if not (1 < len(trad) <= 4) or not all("\u4e00" <= c <= "\u9fff" for c in trad):
            continue
        try:
            key = text_normalize(trad)
        except Exception:  # noqa: BLE001
            continue
        if len(key) == len(trad):
            simp.setdefault(key, (trad, readings))

    diffs = []
    for word, cn in g2pw_mod.pp_dict.items():
        hit = simp.get(word)
        if not hit:
            continue
        trad, readings = hit
        tws = [r for r in (to_tone3(x) for x in readings) if r]
        if tws and not any(t == list(cn) for t in tws):
            diffs.append((trad, list(cn), tws[0]))

    print(f"upstream polyphonic.rep {len(g2pw_mod.pp_dict):,} 筆，"
          f"其中 {len(diffs):,} 筆與教育部不同\n")
    print(f"{'詞':8s} {'upstream（大陸）':22s} {'教育部（台灣）'}")
    print("-" * 56)
    for trad, cn, tw in diffs[:limit]:
        print(f"{trad:8s} {' '.join(cn):22s} {' '.join(tw)}")
    if len(diffs) > limit:
        print(f"... 還有 {len(diffs) - limit:,} 筆（--limit 調整）")
    print(f"\n{WARN} 不建議全部套用 —— 教育部標準有些台灣口語不這樣唸"
          f"（說服 shui4 fu2、滑稽 gu3 ji1）。")
    print(f"   建議用 --text \"你的稿子\" 只處理實際會用到的詞。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="要查的詞（繁體）")
    ap.add_argument("--text", help="從這段文字抓詞來查")
    ap.add_argument("--diff-only", action="store_true", help="只列出不一樣的")
    ap.add_argument("--apply", action="store_true",
                    help="把不一樣的直接寫進 server/pronunciation.rep")
    ap.add_argument("--scan-upstream", action="store_true",
                    help="比對 upstream polyphonic.rep 的 45,046 筆大陸讀音與教育部，"
                         "列出所有不同的（不會自動套用）")
    ap.add_argument("--limit", type=int, default=40, help="--scan-upstream 列幾筆")
    ap.add_argument("--download", action="store_true",
                    help="抓整份教育部辭典到 moedict/（之後離線查，快很多）")
    args = ap.parse_args()

    if args.download:
        return download()

    if args.scan_upstream:
        return scan_upstream(args.limit)

    words = list(args.words)
    if args.text:
        import jieba_fast.posseg as psg

        from text.chinese2 import text_normalize

        for tok in psg.lcut(args.text):
            w = tok.word.strip()
            if len(w) >= 2 and all("一" <= c <= "鿿" for c in w):
                words.append(w)
    if not words:
        ap.error("給我幾個詞，或用 --text")

    if local_index() is not None:
        print(f"用本機辭典 {DATA.name}（{len(local_index()):,} 條）\n")
    else:
        print(f"{WARN} 找不到 {DATA}，改用公開 API（慢）。"
              f"建議先跑 python3 scripts/tw_check.py --download\n")

    P.apply(P.load())  # 先套用現有字典，比對的才是「現在實際的讀音」
    cache = load_cache()
    rows, diffs = [], []
    for w in dict.fromkeys(words):
        moe = moedict(w, cache)
        cur = current_reading(w)
        if not moe:
            rows.append((w, cur, None, "辭典查無此詞"))
            continue
        readings = [to_tone3(m) for m in moe]
        readings = [r for r in readings if r]
        if not readings:
            rows.append((w, cur, None, "拼音無法解析"))
            continue
        same = any(r == cur for r in readings)
        note = "" if same else "不一樣 ←"
        if len(readings) > 1:
            note += f"（辭典有 {len(readings)} 種異讀）"
        rows.append((w, cur, readings, note))
        if not same:
            diffs.append((w, readings[0]))

    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    shown = [r for r in rows if (not args.diff_only or "不一樣" in (r[3] or ""))]
    print(f"{'詞':8s} {'server 現在':20s} {'教育部':20s}")
    print("-" * 58)
    for w, cur, readings, note in shown:
        moe_s = " / ".join(" ".join(r) for r in readings) if readings else "—"
        print(f"{w:8s} {' '.join(cur):20s} {moe_s:20s} {note}")

    if not diffs:
        print(f"\n{OK} 沒有差異")
        return 0
    print(f"\n{len(diffs)} 個詞和教育部不一樣。")
    if not args.apply:
        print("加 --apply 把它們寫進字典（之後可在控制台一鍵刪除還原）。")
        return 0

    added, failed = 0, []
    for w, py in diffs:
        try:
            P.upsert("zh", w, py)
            added += 1
        except P.PronunciationError as e:
            failed.append((w, str(e).splitlines()[-1].strip()))
    print(f"{OK} 寫入 {added} 筆到 {P.DEFAULT_PATH}")
    for w, err in failed:
        print(f"  {BAD} {w}: {err[:100]}")
    print("要重啟 server 才會生效（或用 PUT /dict 逐筆加，那個不用重啟）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
