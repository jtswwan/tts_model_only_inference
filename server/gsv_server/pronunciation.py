"""自訂發音：中英共用的字典覆寫層。

為什麼需要
----------
**英文**：`text/english.py:322` 的 `qryword()` 對「不在 cmudict 裡、長度 <= 3」
的詞一律逐字母拼讀：

    # oov 长度小于等于 3 直接读字母
    if len(word) <= 3:

對縮寫這是對的（GPU 唸 G-P-U、PNG 唸 P-N-G），但對剛好能當單字唸的詞就錯了
—— `wav` 會變成 `D AH1 B AH0 L Y UW0 EY1 V IY1`，也就是 "double-u ay vee"。

**中文**：多音字由 `text/g2pw/g2pw.py:147` 的 `correct_pronunciation()` 處理，
它查 `pp_dict`。g2pw 是整句推理，大多數時候是對的，但遇到人名、專有名詞、
或罕見讀法時會猜錯（「一行」該唸 yi1 hang2 還是 yi1 xing2 要看語境）。

upstream 兩邊都留了熱修檔
--------------------------
| | 英文 | 中文 |
|---|---|---|
| upstream 熱修檔 | `text/engdict-hot.rep` | `text/g2pw/polyphonic-fix.rep` |
| 覆寫對象 | `_g2p.cmu[word]` | `g2pw.pp_dict[word]` |
| 單位 | ARPAbet 音素 | 拼音（帶聲調數字） |

但那兩個檔案都在 vendored 樹裡、被 `scripts/manifest.txt` 管著，
`vendor_upstream.sh` 會重新複製，手改會被蓋掉。

而且中文那邊還有一個坑：`g2pw/g2pw.py:117-125` 的 `get_dict()` 讀到
`polyphonic.pickle` 就直接回傳，**不會**再套用 `.rep`。所以 pickle 存在時
改 `.rep` 根本沒效果（英文的 `get_dict()` 有 `hot_reload_hot()`，中文沒有）。

所以這裡自己來：資料放 `server/pronunciation.rep`（vendored 樹之外），
在暖機時覆寫進那兩個 dict。效果和 upstream 的熱修檔相同，而且繞過了
中文那個 pickle 的坑，upstream 的程式碼一個字都不用改。

為什麼一定要驗證
----------------
`replace_phs()`（english.py:111-121）對不在 `symbols` 裡的音素是**靜默丟掉**
的，只 print 一行 "ph not in symbols"。所以一個打錯的音素不會報錯，只會讓
那個字少一個音。中文側打錯拼音則會讓 `to_finals_tone3()` 回空字串，同樣
無聲失敗。兩邊都在載入時擋掉。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import bootstrap

log = logging.getLogger("gsv.pronunciation")

# server/pronunciation.rep —— 在 vendored 樹之外，不受 vendor_upstream.sh 影響
DEFAULT_PATH = bootstrap.VENDOR_ROOT / "pronunciation.rep"

LANGS = ("en", "zh")

HEADER = """\
# 自訂發音 —— 本專案自己的檔案，不是 vendored upstream
#
# 這個檔案可以用控制台的「字典」頁編輯，也可以直接手改（改完重啟 server，
# 或用 PUT /dict 讓它立刻生效）。
#
# [en] 英文：單字 後面接大寫 ARPAbet 音素，空白分隔，重音用 0/1/2 標在母音後。
#      為什麼需要：english.py:322 對「不在字典裡、長度 <= 3」的詞一律逐字母
#      拼讀。那對 GPU / PNG 是對的，對 wav 就錯了。
#
# [zh] 中文：詞 後面接拼音，空白分隔，聲調用數字。拼音數量必須等於字數。
#      為什麼需要：g2pw 整句推理大多正確，但人名與專有名詞會猜錯。
#
# 音素 / 拼音都會在載入時驗證 —— 打錯會明確失敗，而不是像 upstream 的
# replace_phs() 那樣靜默把音丟掉。
#
# 查某個詞現在怎麼唸：python3 scripts/check_en.py <詞>
"""


class PronunciationError(RuntimeError):
    pass


# ---- 解析 ----


def parse(text: str, source: str = "<string>") -> dict[str, dict[str, list[str]]]:
    """解析分節的 `.rep`。沒有節標頭的行歸到 [en]（向後相容）。"""
    entries: dict[str, dict[str, list[str]]] = {lang: {} for lang in LANGS}
    section = "en"
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[(\w+)\]", line)
        if header:
            section = header.group(1).lower()
            if section not in LANGS:
                raise PronunciationError(
                    f"{source}:{lineno} 不認識的節 [{section}]，只能是 "
                    + " / ".join(f"[{v}]" for v in LANGS)
                )
            continue
        parts = line.split()
        if len(parts) < 2:
            raise PronunciationError(
                f"{source}:{lineno} 格式錯誤：至少要有「詞 發音」兩欄 —— {raw!r}"
            )
        word = parts[0] if section == "zh" else parts[0].lower()
        if word in entries[section]:
            raise PronunciationError(f"{source}:{lineno} [{section}] 的 {word!r} 重複定義")
        entries[section][word] = parts[1:]
    return entries


# ---- 正規化 ----


def normalize_word(lang: str, word: str) -> str:
    """把使用者輸入的詞轉成 upstream 實際會查字典的形式。

    中文管線在 `clean_text` 裡先跑 `chinese2.text_normalize()`，那會做繁簡
    轉換（實測 `檔案` -> `档案`），之後才由 jieba 斷詞、`correct_pronunciation()`
    拿著**簡體**的詞去查 `pp_dict`。

    所以字典的 key 必須是簡體。使用者打繁體卻存成繁體的話，key 永遠對不上，
    改了完全沒反應而且沒有任何錯誤 —— 正是本專案要避免的無聲失敗。
    """
    word = word.strip()
    if lang != "zh":
        return word.lower()
    try:
        bootstrap.require_setup()
        from text.chinese2 import text_normalize

        normed = "".join(
            ch for ch in text_normalize(word) if not ch.isspace()
        ).strip("".join(PUNCT_STRIP))
        return normed or word
    except Exception:  # noqa: BLE001
        return word


PUNCT_STRIP = "，。！？；：、,.!?;:"


# ---- 驗證 ----


def _validate_en(words: dict[str, list[str]], source: str) -> list[str]:
    bootstrap.require_setup()
    from text.symbols2 import symbols

    known = set(symbols)
    problems = []
    for word, phones in words.items():
        bad = [p for p in phones if p not in known]
        if bad:
            problems.append(
                f"[en] {word}: text/symbols2.py 不認識的音素 {' '.join(bad)}"
                f"（會被 replace_phs() 靜默丟掉）"
            )
    return problems


def _zh_pinyin_key(initial: str, final_with_tone: str) -> tuple[str, str] | None:
    """照抄 `chinese2.py:244-289` 把 (聲母, 帶調韻母) 轉成 pinyin_to_symbol_map 的鍵。

    不能只拿 `to_finals_tone3()` 的結果去比對 symbols2 —— upstream 中間還有
    兩層改寫，少做就會把合法的拼音誤判成錯的（實測 `niu2` 的 strict 韻母是
    `iou2`，不在 symbols2 裡，但 upstream 會先改寫成 `iu` 才查表）。
    """
    v_without_tone = final_with_tone[:-1]
    tone = final_with_tone[-1:]
    if tone not in "12345":
        return None
    pinyin = initial + v_without_tone
    if initial:
        v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
        if v_without_tone in v_rep_map:
            pinyin = initial + v_rep_map[v_without_tone]
    else:
        pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
        if pinyin in pinyin_rep_map:
            pinyin = pinyin_rep_map[pinyin]
        else:
            single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
            if pinyin[:1] in single_rep_map:
                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
    return pinyin, tone


def _validate_zh(words: dict[str, list[str]], source: str) -> list[str]:
    bootstrap.require_setup()
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

    from text.chinese2 import pinyin_to_symbol_map

    problems = []
    for word, pinyins in words.items():
        chars = [c for c in word if not c.isspace()]
        if len(pinyins) != len(chars):
            problems.append(
                f"[zh] {word}: 有 {len(chars)} 個字但給了 {len(pinyins)} 個拼音。"
                f"correct_pronunciation() 是逐字對應的，數量必須相同。"
            )
            continue
        for py in pinyins:
            # 沒標聲調的話 to_finals_tone3 會自己補輕聲（"ce" -> "e5"），
            # 使用者多半不是那個意思。寧可明確要求，不要悄悄換一個讀音。
            if py[-1:] not in "12345":
                problems.append(
                    f"[zh] {word}: 拼音 {py!r} 沒有聲調。要在後面加 1-5"
                    f"（例如 {py}1；5 是輕聲）。"
                )
                continue
            try:
                initial = to_initials(py)
                final = to_finals_tone3(py, neutral_tone_with_five=True)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"[zh] {word}: 拼音 {py!r} 無法解析（{exc}）")
                continue
            if not final:
                problems.append(f"[zh] {word}: 拼音 {py!r} 解不出韻母，應該是打錯了")
                continue
            key = _zh_pinyin_key(initial, final)
            if key is None:
                problems.append(f"[zh] {word}: 拼音 {py!r} 缺聲調（要 1-5）")
            elif key[0] not in pinyin_to_symbol_map:
                problems.append(
                    f"[zh] {word}: 拼音 {py!r} 不是合法的音節"
                    f"（chinese2.py:289 會在 {key[0]!r} 這裡 assert 失敗）"
                )
    return problems


def validate(entries: dict[str, dict[str, list[str]]], source: str = "<string>") -> None:
    problems = _validate_en(entries.get("en", {}), source)
    problems += _validate_zh(entries.get("zh", {}), source)
    if problems:
        raise PronunciationError(f"{source} 有問題：\n  - " + "\n  - ".join(problems))


# ---- 載入與套用 ----


def load(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """讀取並驗證。檔案不存在就回空的 —— 這個檔案是選用的。"""
    path = DEFAULT_PATH if path is None else Path(path)
    empty: dict[str, dict[str, list[str]]] = {lang: {} for lang in LANGS}
    if not path.exists():
        return empty
    entries = parse(path.read_text(encoding="utf-8"), str(path))
    validate(entries, str(path))
    return entries


# upstream 原本的值，第一次覆寫前存下來，這樣刪除自訂發音不必重啟
_originals: dict[str, dict[str, Any]] = {lang: {} for lang in LANGS}

_MISSING = object()


def _target_dicts() -> dict[str, dict]:
    """拿到 upstream 的兩個字典本體。必須在 text.english / g2pw import 之後。"""
    bootstrap.require_setup()
    from text import english
    from text.g2pw import g2pw as g2pw_mod

    return {"en": english._g2p.cmu, "zh": g2pw_mod.pp_dict}


def apply(entries: dict[str, dict[str, list[str]]]) -> dict[str, int]:
    """覆寫進 upstream 的字典，回傳每種語言套用的筆數。

    做法和 upstream 的 `hot_reload_hot()` / `polyphonic-fix.rep` 一樣：直接
    覆寫字典條目。`qryword()` 與 `correct_pronunciation()` 第一步都是查字典，
    查得到就不會走到各自的猜測邏輯。
    """
    counts = {lang: 0 for lang in LANGS}
    if not any(entries.get(lang) for lang in LANGS):
        return counts
    targets = _target_dicts()
    for lang in LANGS:
        words = entries.get(lang, {})
        target = targets[lang]
        for word, value in words.items():
            if word not in _originals[lang]:
                _originals[lang][word] = target.get(word, _MISSING)
            # 英文的 cmu 是 {word: [[ph, ...]]}；中文的 pp_dict 是 {word: [py, ...]}
            target[word] = [list(value)] if lang == "en" else list(value)
            counts[lang] += 1
    applied = {k: v for k, v in counts.items() if v}
    if applied:
        log.info(
            "已套用自訂發音：%s",
            "、".join(
                f"{lang} {n} 筆（{', '.join(sorted(entries[lang]))}）"
                for lang, n in applied.items()
            ),
        )
    return counts


def restore(lang: str, word: str) -> bool:
    """把某個詞還原成 upstream 原本的值，不必重啟。"""
    if lang not in LANGS:
        raise PronunciationError(f"不認識的語言 {lang!r}")
    word = normalize_word(lang, word)
    original = _originals.get(lang, {}).pop(word, _MISSING)
    target = _target_dicts()[lang]
    if original is _MISSING:
        target.pop(word, None)
    else:
        target[word] = original
    return True


# ---- 寫檔 ----


def dump(entries: dict[str, dict[str, list[str]]]) -> str:
    out = [HEADER]
    for lang in LANGS:
        words = entries.get(lang, {})
        out.append(f"\n[{lang}]")
        for word in sorted(words):
            out.append(f"{word} {' '.join(words[word])}")
    return "\n".join(out) + "\n"


def save(entries: dict[str, dict[str, list[str]]], path: Path | None = None) -> Path:
    """驗證後原子寫入。驗證不過就不寫，檔案保持可用。"""
    path = DEFAULT_PATH if path is None else Path(path)
    validate(entries, str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dump(entries), encoding="utf-8")
    tmp.replace(path)
    return path


def upsert(lang: str, word: str, value: list[str],
           path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """新增或修改一筆，寫檔並立刻生效。"""
    if lang not in LANGS:
        raise PronunciationError(f"不認識的語言 {lang!r}，只能是 {' / '.join(LANGS)}")
    if not value:
        raise PronunciationError("發音不能是空的")
    entries = load(path)
    key = normalize_word(lang, word)
    if not key:
        raise PronunciationError("詞不能是空的")
    entries[lang][key] = list(value)
    save(entries, path)
    apply({lang: {key: list(value)}, **{l: {} for l in LANGS if l != lang}})
    return entries


def remove(lang: str, word: str, path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """刪掉一筆，寫檔並還原成 upstream 原本的值。"""
    if lang not in LANGS:
        raise PronunciationError(f"不認識的語言 {lang!r}")
    entries = load(path)
    key = normalize_word(lang, word)
    if key not in entries[lang]:
        raise PronunciationError(f"[{lang}] 裡沒有 {key!r}")
    del entries[lang][key]
    save(entries, path)
    restore(lang, key)
    return entries


def load_and_apply(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
    entries = load(path)
    apply(entries)
    return entries


def flatten(entries: dict[str, dict[str, list[str]]]) -> list[str]:
    """給 /health 用的扁平清單。"""
    return [f"{lang}:{word}" for lang in LANGS for word in sorted(entries.get(lang, {}))]


# ---- 被靜默丟掉的音素 ----
#
# `replace_phs()`（english.py:111-121）對不在 `symbols` 裡的音素只 print 一行
# "ph not in symbols" 就丟掉。那在 server 上等於看不見：輸出少一個音，沒有
# 任何錯誤、沒有任何回傳欄位。這裡把它記下來，讓 API 與控制台看得到。
#
# 做法是在執行期包住 `english.replace_phs`，不改 upstream 的檔案。實際回傳值
# **一律交給原函式**，所以音訊輸出完全不變 —— 這裡只負責記錄。

import threading  # noqa: E402
from collections import deque  # noqa: E402

_drops: deque[dict[str, Any]] = deque(maxlen=200)
_drop_seq = 0
_drop_lock = threading.Lock()
_recorder_installed = False

# english.py:112 的 replace_phs 內部有 `rep_map = {"'": "-"}`，那是局部變數拿不到。
# 這裡複製那一個 key：被映射的不算「丟掉」。
_MAPPED = {"'"}


def _record_drops(phones: list[str]) -> None:
    global _drop_seq
    bootstrap.require_setup()
    from text.symbols2 import symbols

    known = set(symbols)
    dropped = [p for p in phones if p not in known and p not in _MAPPED]
    if not dropped:
        return
    with _drop_lock:
        _drop_seq += 1
        _drops.append({"seq": _drop_seq, "phones": dropped})
    log.warning(
        "有音素不在 symbols2 裡，upstream 的 replace_phs() 會把它們丟掉：%s",
        " ".join(dropped),
    )


def install_drop_recorder() -> bool:
    """包住 `english.replace_phs` 以記錄被丟掉的音素。可重複呼叫。

    回傳值一律來自原函式，所以輸出不變。
    """
    global _recorder_installed
    if _recorder_installed:
        return False
    bootstrap.require_setup()
    from text import english

    original = english.replace_phs
    if getattr(original, "_gsv_wrapped", False):
        _recorder_installed = True
        return False

    def wrapper(phs):
        try:
            _record_drops(list(phs))
        except Exception:  # noqa: BLE001
            pass  # 記錄失敗絕不能影響合成
        return original(phs)

    wrapper._gsv_wrapped = True  # type: ignore[attr-defined]
    english.replace_phs = wrapper  # type: ignore[assignment]
    _recorder_installed = True
    log.info("已安裝音素丟失記錄器（包住 english.replace_phs，不改 upstream）")
    return True


def drop_cursor() -> int:
    with _drop_lock:
        return _drop_seq


def drops_since(cursor: int) -> list[str]:
    """回傳 cursor 之後被丟掉的音素，給單一請求用。"""
    with _drop_lock:
        out: list[str] = []
        for item in _drops:
            if item["seq"] > cursor:
                out.extend(item["phones"])
        return out


def recent_drops(limit: int = 20) -> list[dict[str, Any]]:
    with _drop_lock:
        return list(_drops)[-limit:]


# ---- 查詢：某段文字現在實際會怎麼唸 ----


def _letter_by_letter(g2p_obj: Any, word: str) -> list[str]:
    """重現 `qryword()` 的逐字母分支（english.py:322-332），用來判斷是否走了那條路。"""
    phones: list[str] = []
    for ch in word.lower():
        if ch == "a":
            phones.extend(["EY1"])
        elif not ch.isalpha():
            phones.append(ch)
        else:
            phones.extend(g2p_obj.cmu[ch][0])
    return phones


def _lookup_en(text: str, custom: dict[str, list[str]]) -> list[dict[str, Any]]:
    from text.english import _g2p, text_normalize

    # 真實管線是 clean_text -> english.text_normalize -> g2p，normalize 會把
    # 數字展開（"v2Pro" 的 "2" -> "two"）。不跑它，預覽就會和實際輸出不一致。
    text = text_normalize(text)
    out = []
    for raw in text.split():
        word = raw.strip(".,!?;:\"'()[]").lower()
        if not word or not any(c.isalpha() for c in word):
            continue
        try:
            phones = _g2p.qryword(word)
        except Exception as exc:  # noqa: BLE001
            out.append({"lang": "en", "word": word, "error": str(exc)})
            continue
        spelled = phones == _letter_by_letter(_g2p, word)
        out.append({
            "lang": "en",
            "word": word,
            "phones": phones,
            "custom": word in custom,
            "letter_by_letter": spelled and word not in custom,
            "in_cmudict": len(word) > 1 and word in _g2p.cmu,
        })
    return out


def _lookup_zh(text: str, custom: dict[str, list[str]]) -> list[dict[str, Any]]:
    """逐詞回報中文讀音，做法必須和真實管線一致。

    這裡的 `text` 是**單一個 LangSegmenter 片段**，不是整句。因為
    `get_phones_and_bert()`（TextPreprocessor.py:175-180）是逐片段呼叫
    `clean_text_inf()` 的，所以 g2pw 實際上也只看到那一個片段 —— 中英混雜時
    它拿不到跨越英文的上下文。餵整句會得到和實際輸出不同的聲調。

    順序也要照抄：`clean_text` 先 `text_normalize`（會把數字展開成中文）
    再 `g2p`，而 `chinese2.py:186` 在 `_g2p` 裡才把英文字母去掉。
    """
    import jieba_fast.posseg as psg

    from pypinyin import Style

    from text.chinese2 import g2pw, text_normalize
    from text.g2pw import g2pw as g2pw_mod

    pp = g2pw_mod.pp_dict
    # clean_text 的順序是 text_normalize 再 g2p，而 text_normalize 會把數字
    # 展開成中文（"v2Pro" 的 "2" -> "二"）。先去英文再 normalize 會留下那個
    # 數字，導致 jieba 的詞和 g2pw 的拼音錯位。
    zh_text = re.sub("[a-zA-Z]+", "", text_normalize(text))
    if not zh_text.strip():
        return []
    try:
        sentence_pinyin = g2pw.lazy_pinyin(
            zh_text, neutral_tone_with_five=True, style=Style.TONE3
        )
    except Exception:  # noqa: BLE001
        sentence_pinyin = []

    out = []
    pos = 0
    for token in psg.lcut(zh_text):
        word = token.word
        n = len(word)
        pinyins = list(sentence_pinyin[pos:pos + n]) if sentence_pinyin else []
        pos += n
        if not any("\u4e00" <= c <= "\u9fff" for c in word):
            continue

        # correct_pronunciation()（g2pw.py:147-156）先查整詞，查到就直接回傳，
        # 查不到才逐字 fallback。所以整詞條目會遮蔽其中單字的條目 ——
        # 使用者改了單字卻沒反應時，就是踩到這個。
        whole_word_hit = word in pp
        if whole_word_hit:
            pinyins = list(pp[word])
        else:
            for idx, ch in enumerate(word):
                if ch in pp and idx < len(pinyins):
                    pinyins[idx] = pp[ch][0]

        out.append({
            "lang": "zh",
            "word": word,
            "pinyin": pinyins,
            "custom": word in custom,
            "in_polyphonic": whole_word_hit,
            "shadows_single_chars": word if (n > 1 and whole_word_hit) else None,
        })
    return out


def lookup(text: str, entries: dict[str, dict[str, list[str]]] | None = None
           ) -> dict[str, Any]:
    """逐詞回報「現在實際會唸成什麼」，並標出需要注意的地方。

    標出三件事：
    - `letter_by_letter`：英文走了 english.py:322 的逐字母分支
    - `shadows_single_chars`：中文這個整詞條目會遮蔽其中單字的自訂發音
    - `dropped_phones`：有音素不在 symbols2 裡，會被 replace_phs() 靜默丟掉
    """
    bootstrap.require_setup()
    from text.LangSegmenter import LangSegmenter

    entries = load() if entries is None else entries
    cursor = drop_cursor()
    # 逐片段處理，順序與 get_phones_and_bert() 相同。
    #
    # 注意「非 en 一律當成 zh」：LangSegmenter 會把中文誤判成 ja（實測
    # 「存成」就被判成 ja），而 TextPreprocessor.py:166-168 對非 en 的片段是
    # 用「使用者指定的語言」而不是偵測結果：
    #
    #     # 因无法区别中日韩文汉字,以用户输入为准
    #     langlist.append(language)
    #
    # 所以實際上那些片段都走 chinese2。這裡照做，否則被誤判的詞會從編輯器
    # 的清單裡消失 —— 使用者就沒辦法修它。
    items: list[dict[str, Any]] = []
    for seg in LangSegmenter.getTexts(text):
        if seg["lang"] == "en":
            items.extend(_lookup_en(seg["text"], entries.get("en", {})))
        else:
            items.extend(_lookup_zh(seg["text"], entries.get("zh", {})))
    return {
        "text": text,
        "items": items,
        "dropped_phones": drops_since(cursor),
    }
