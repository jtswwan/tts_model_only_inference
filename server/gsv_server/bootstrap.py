"""讓 vendored 的 upstream 程式碼能在不被修改的前提下正常 import。

upstream 假設了兩件事，兩件都和工作目錄有關：

1. sys.path 同時含 repo 根與 GPT_SoVITS/
   `TTS.py` 寫 `from AR.models... import`（需要 GPT_SoVITS/ 在路徑上）
   也寫 `from tools.i18n.i18n import`（需要 repo 根在路徑上）。

2. 工作目錄就是 repo 根
   `TTS.py:13` 在模組層級執行 `now_dir = os.getcwd()`，
   `sv.py:5-6` 用 `os.getcwd()` 組出 eres2net 路徑與 SV 權重路徑，
   `text/chinese2.py:35` 寫死相對路徑 "GPT_SoVITS/text/G2PWModel"。

因此 setup() 必須在 import 任何 upstream 模組「之前」呼叫 —— 包含
TTS.py，因為它在模組層級就把 getcwd() 的值固定下來了。

修這些相對路徑的替代做法是改 upstream 的檔案。本專案選擇不改，
改用進入點負責建立它們預期的環境。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# server/ —— 內含 vendored 的 GPT_SoVITS/ 與 tools/
VENDOR_ROOT = Path(__file__).resolve().parent.parent

# 專案內的 nltk 語料目錄（repo 根的 nltk_data/，可攜版裡是 app/nltk_data/）。
# 放專案內而不是 ~/nltk_data，可攜版才能整個資料夾搬走還能離線跑英文。
NLTK_DATA = VENDOR_ROOT.parent / "nltk_data"

_done = False


def setup() -> Path:
    """建立 upstream 預期的 sys.path 與工作目錄。多次呼叫安全。"""
    global _done
    if _done:
        return VENDOR_ROOT

    if not (VENDOR_ROOT / "GPT_SoVITS").is_dir():
        raise RuntimeError(
            f"{VENDOR_ROOT} 下找不到 GPT_SoVITS/。"
            " 請先執行 scripts/vendor_upstream.sh。"
        )

    _use_local_nltk_data()
    os.chdir(VENDOR_ROOT)
    for path in (VENDOR_ROOT / "GPT_SoVITS", VENDOR_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    _done = True
    return VENDOR_ROOT


def _use_local_nltk_data() -> None:
    """讓 nltk 先找專案內的 nltk_data/。

    `text/english.py` 透過 g2p_en 需要 nltk 的 cmudict 與
    averaged_perceptron_tagger。g2p_en 在 import 時會自己 nltk.download()，
    那表示：離線機器上英文合成會失敗，而有網路時第一個英文請求會在請求路徑上
    下載語料 —— 兩者都不能接受。

    nltk 在 import 時讀 NLTK_DATA（os.pathsep 分隔），所以這裡必須在
    import nltk 之前執行。使用者已設的值保留在後面，不覆蓋。
    """
    existing = os.environ.get("NLTK_DATA", "")
    entry = str(NLTK_DATA)
    parts = [p for p in existing.split(os.pathsep) if p]
    if entry not in parts:
        os.environ["NLTK_DATA"] = os.pathsep.join([entry, *parts])


def require_setup() -> None:
    if not _done:
        raise RuntimeError("必須先呼叫 bootstrap.setup() 才能 import upstream 模組")
