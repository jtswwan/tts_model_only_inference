"""讓中英混雜時，中文的 BERT 看得到整句。

問題
----
`TextPreprocessor.get_phones_and_bert()` 是逐片段處理的
（`TextPreprocessor.py:175-180`）：

    for i in range(len(textlist)):
        phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
        bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
        ...
    bert = torch.cat(bert_list, dim=1)

所以被英文隔開的中文會各自成組，每組的 BERT 只看得到自己那一小段。實測
「我把檔案存成 wav 然後上傳到 server 再用 python 處理」中文被拆成 **4 組**，
`我把檔案存成` 那一組完全不知道後面還有話 —— 沒有任何一次前向看過整句，
整句的語調輪廓不可能對。純中文句子只有 1 組，BERT 看得到整句，語調就正常。

做法
----
把所有中文片段串成一個連續的中文句子，**一次**算 BERT，再把結果切回各片段
的位置；英文片段照舊用全零。英文的音素不變、位置不變，只有中文那幾段的
BERT 從「看自己」變成「看整句中文」。

這等於把 upstream 在純中文時本來就有的全句上下文，還給中英混雜的情況。

為什麼不是亂改
--------------
- 送進 BERT 的文本仍然是一個合法的中文句子（英文本來就會被
  `chinese2.py:186` 的 `re.sub("[a-zA-Z]+", "", seg)` 丟掉）。
- 純中文輸入時行為完全不變（只有一組，切不切都一樣）。
- 音素序列完全不變 —— 可以逐一比對驗證（`verify()`）。

但這確實改變了 BERT 特徵，所以是 §4 的刻意偏離，預設關閉，開啟時列在
`/health` 的 `quality.deviations`。

注意：斷詞與變調現在會跨越原本的片段邊界（`存成` 和 `然後` 變成相鄰），
那正是目的 —— 但也表示中文音素**可能**和逐片段時不同。`verify()` 會把
差異列出來，而不是假裝沒有。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("gsv.zh_context")

_installed = False
_original = None


def _merge_segments(text: str, language: str) -> tuple[list[str], list[str]]:
    """照抄 TextPreprocessor.py:158-169 的片段合併規則。"""
    from text.LangSegmenter import LangSegmenter

    textlist: list[str] = []
    langlist: list[str] = []
    for tmp in LangSegmenter.getTexts(text):
        if langlist:
            if (tmp["lang"] == "en" and langlist[-1] == "en") or (
                tmp["lang"] != "en" and langlist[-1] != "en"
            ):
                textlist[-1] += tmp["text"]
                continue
        if tmp["lang"] == "en":
            langlist.append("en")
        else:
            # 因无法区别中日韩文汉字,以用户输入为准（upstream 原註解）
            langlist.append(language)
        textlist.append(tmp["text"])
    return textlist, langlist


def build(preprocessor: Any, text: str, language: str, version: str):
    """回傳 (phones, bert, norm_text)，中文部分共用一次全句 BERT。"""
    from text import cleaned_text_to_sequence
    from text.chinese2 import g2p as zh_g2p
    from text.chinese2 import text_normalize as zh_normalize

    textlist, langlist = _merge_segments(text, language)
    zh_idx = [i for i, l in enumerate(langlist) if l != "en"]
    if len(zh_idx) < 2:
        return None  # 只有 0 或 1 組中文，切不切都一樣，交回 upstream

    # 1. 各中文片段先各自 normalize（和 clean_text 的順序一致），再串起來。
    #    先串再 normalize 會讓數字展開跨越邊界，字數對不回去。
    zh_norms = {i: zh_normalize(textlist[i]) for i in zh_idx}
    if any(not n for n in zh_norms.values()):
        return None
    norm_all = "".join(zh_norms[i] for i in zh_idx)

    # 2. 整句中文一次 g2p + 一次 BERT —— 這就是全句上下文的來源
    phones_all, word2ph_all = zh_g2p(norm_all)
    if len(word2ph_all) != len(norm_all):
        log.warning("word2ph 與文本長度不符（%d vs %d），交回 upstream",
                    len(word2ph_all), len(norm_all))
        return None
    bert_all = preprocessor.get_bert_feature(norm_all, word2ph_all).to(preprocessor.device)

    # 3. 依各片段的字數切回去
    zh_slices: dict[int, tuple[list[str], torch.Tensor]] = {}
    char_off = 0
    phone_off = 0
    for i in zh_idx:
        n_char = len(zh_norms[i])
        n_phone = sum(word2ph_all[char_off:char_off + n_char])
        zh_slices[i] = (
            phones_all[phone_off:phone_off + n_phone],
            bert_all[:, phone_off:phone_off + n_phone],
        )
        char_off += n_char
        phone_off += n_phone

    # 4. 依原順序組回去，英文照舊
    phones_list: list[list[int]] = []
    bert_list: list[torch.Tensor] = []
    norm_list: list[str] = []
    for i, lang in enumerate(langlist):
        if lang == "en":
            ph, word2ph, norm = preprocessor.clean_text_inf(textlist[i], "en", version)
            bert = preprocessor.get_bert_inf(ph, word2ph, norm, "en")
            phones_list.append(ph)
            bert_list.append(bert)
            norm_list.append(norm)
        else:
            ph_str, bert = zh_slices[i]
            phones_list.append(cleaned_text_to_sequence(ph_str, version))
            bert_list.append(bert)
            norm_list.append(zh_norms[i])

    phones = sum(phones_list, [])
    bert = torch.cat(bert_list, dim=1)
    if bert.shape[-1] != len(phones):
        log.warning("BERT 長度與音素數不符（%d vs %d），交回 upstream",
                    bert.shape[-1], len(phones))
        return None
    return phones, bert.to(preprocessor.device), "".join(norm_list)


def install() -> bool:
    """在執行期包住 `get_phones_and_bert`。不改 upstream 檔案。可重複呼叫。"""
    global _installed, _original
    if _installed:
        return False
    from TTS_infer_pack.TextPreprocessor import TextPreprocessor

    original = TextPreprocessor.get_phones_and_bert
    if getattr(original, "_gsv_wrapped", False):
        _installed = True
        return False

    def wrapper(self, text: str, language: str, version: str, final: bool = False):
        if getattr(self, "_gsv_zh_context", False):
            try:
                built = build(self, text, language, version)
            except Exception:  # noqa: BLE001
                log.exception("全句中文 BERT 失敗，交回 upstream")
                built = None
            if built is not None:
                phones, bert, norm_text = built
                # upstream 對太短的文本會補一個句點再跑一次（TextPreprocessor.py:185-186）
                if not final and len(phones) < 6:
                    return original(self, "." + text, language, version, final=True)
                return phones, bert, norm_text
        return original(self, text, language, version, final)

    wrapper._gsv_wrapped = True  # type: ignore[attr-defined]
    TextPreprocessor.get_phones_and_bert = wrapper  # type: ignore[assignment]
    _original = original
    _installed = True
    log.info("已安裝全句中文 BERT（包住 get_phones_and_bert，不改 upstream）")
    return True


def verify(preprocessor: Any, text: str, language: str, version: str) -> dict[str, Any]:
    """比對「全句 BERT」與 upstream 逐片段的差異。

    音素應該**可能**不同（斷詞與變調現在會跨越片段邊界，那正是目的），
    所以如實列出來，不假裝相同。
    """
    assert _original is not None, "尚未 install()"
    base_phones, base_bert, base_norm = _original(preprocessor, text, language, version)
    built = build(preprocessor, text, language, version)
    if built is None:
        return {"applicable": False}
    phones, bert, norm = built
    same_phones = list(phones) == list(base_phones)
    return {
        "applicable": True,
        "phones_identical": same_phones,
        "phones_upstream": len(base_phones),
        "phones_full_context": len(phones),
        "norm_upstream": base_norm,
        "norm_full_context": norm,
        "bert_max_abs_diff": (
            float((bert.float() - base_bert.float()).abs().max())
            if bert.shape == base_bert.shape else None
        ),
    }
