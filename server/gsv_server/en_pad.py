"""拿掉 upstream 塞在「短英文片段」前面的逗號（只在中英混合的句子裡）。

問題
----
`text/cleaner.py:46-49`：

    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones

英文片段不到 4 個音素就在前面補一個逗號。用意是整段文字只有 "OK" 這種極短的
英文時，音素太少模型會唸壞，補一個逗號墊長度。

但前端是把一句話切成中文片段與英文片段分別處理的，這條規則對「夾在中文裡的
英文」一樣生效：

    等等你發IG的時候   ->   f a1 , AY1 JH IY1 d e5 …
                                 ^ 這個逗號不在原文裡

「IG」「AI」「OK」「PC」這種兩三個字母的縮寫都會中。模型看到逗號就停頓，實測
「發」和「IG」之間有 140 ms 的靜音。

做法
----
包住 `clean_text_inf()`，句子裡有中文（或特殊符號）時，把那個補上去的逗號拿掉。
整段都是英文時不動 —— 那是這條規則本來要處理的情況。

這會改變送進模型的音素，所以是選項（`latency.zh_en_no_pad_comma`），開啟時列在
`/health` 的 `quality.deviations`。
"""

from __future__ import annotations

import re
import threading
from typing import Callable

_ctx = threading.local()

# 有中文字，或有特殊符號的佔位詞（見 aliases.py）就算混合句
_MIXED_RE = re.compile(r"[一-龥]|gsvsym[a-z]+")

# 片段以這些開頭時，最前面的逗號是原文就有的，不是補的
_LEADING_PUNCT = set(",，、;；:：·/")


def wrap_get_phones_and_bert(original: Callable) -> Callable:
    """記下「目前這句是不是中英混合」，給 clean_text_inf 的包裝用。"""

    def wrapped(text: str, language: str, version: str, final: bool = False):
        previous = getattr(_ctx, "mixed", False)
        _ctx.mixed = bool(_MIXED_RE.search(text))
        try:
            return original(text, language, version, final)
        finally:
            _ctx.mixed = previous

    return wrapped


def wrap_clean_text_inf(original: Callable) -> Callable:
    def wrapped(text: str, language: str, version: str = "v2"):
        ids, word2ph, norm_text = original(text, language, version)
        if (
            not getattr(_ctx, "mixed", False)
            or language.replace("all_", "") != "en"
            or len(ids) < 2
            # upstream 只在音素不到 4 個時補；補完最多 4 個
            or len(ids) - 1 >= 4
        ):
            return ids, word2ph, norm_text

        from text import cleaned_text_to_sequence

        comma = cleaned_text_to_sequence([","], version)[0]
        if ids[0] != comma:
            return ids, word2ph, norm_text
        # 片段本身以逗號類標點開頭時，第一個逗號可能是原文的。那種情況只有在
        # 後面緊接著第二個逗號時才確定有一個是補的。
        if text.lstrip()[:1] in _LEADING_PUNCT and ids[1] != comma:
            return ids, word2ph, norm_text
        return ids[1:], word2ph, norm_text

    return wrapped
