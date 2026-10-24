#!/usr/bin/env python3
"""產生 Colab.ipynb。

把 notebook 寫成產生器而不是手改 JSON：.ipynb 是 JSON，手改容易弄壞，
而且 cell 內容是 list of lines，直接編輯很痛苦。

    python3 scripts/make_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/jtswwan/tts_model_only_inference"
BRANCH = "claude/blissful-hawking-safc6x"


# nbformat 的 source 是「每行含換行字元」的字串陣列。少了 \n，
# Jupyter/Colab 會把整個 cell 串成一行，程式碼直接壞掉。
def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(text)}


CELLS = [
    md(f"""
# GPT-SoVITS 純推論 server — Colab

v2Pro + 洛神。跑完就會得到一個可以操作的控制台，以及**真實的 GPU 延遲數字**。

> **先確認有 GPU**：上方選單 `執行階段` → `變更執行階段類型` → 硬體加速器選 **T4 GPU**。
> 沒有 GPU 也能跑，但即時倍率只有 0.6–0.8×（達不到實時）。

依序執行下面的 cell 即可。第一次要下載約 2.85 GB 權重。

Repo: {REPO_URL}
"""),

    md("## 1. 檢查執行環境"),
    code("""
import subprocess, sys, platform
print("Python", platform.python_version())
r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True)
if r.returncode == 0:
    print("GPU:", r.stdout.strip())
else:
    print("⚠ 沒有偵測到 GPU。執行階段 → 變更執行階段類型 → T4 GPU")
    print("  沒有 GPU 也能跑完，但達不到實時。")
"""),

    md("## 2. 取得程式碼"),
    code(f"""
import os, pathlib
REPO_DIR = pathlib.Path("/content/tts_model_only_inference")
if not REPO_DIR.exists():
    !git clone --depth 1 -b {BRANCH} {REPO_URL} {{REPO_DIR}}
else:
    print("已存在，跳過")
os.chdir(REPO_DIR)
print("工作目錄:", os.getcwd())
"""),

    md("""
## 3. 安裝依賴

Colab 已經預裝 CUDA 版的 torch，重裝要好幾分鐘又佔好幾 GB，所以先檢查能不能沿用。

有一個相容性問題要處理：**torchaudio 從 2.8/2.9 起把自己的 I/O backend 移除**，
`torchaudio.load()` 改委派給 `torchcodec`，而 `TTS.py:773` 讀參考音訊正是用它。
下面會依序試三條路，挑最省的那條：

1. Colab 現成的 torchaudio 就能讀 wav → 什麼都不用動
2. 裝 `torchcodec` 就能讀 → 只加一個套件
3. 都不行 → 才降級 torch/torchaudio 到 `<2.9`（最慢，要重裝 CUDA 版 torch）
"""),
    code("""
import subprocess, sys, importlib

def sh(*args):
    print("$", " ".join(args))
    return subprocess.run(args).returncode

def can_load_wav() -> bool:
    \"\"\"實測 torchaudio.load 能不能讀 wav —— 這才是真正要的能力。\"\"\"
    probe = '''
import torchaudio, torch, tempfile, os
sr = 32000
wav = torch.zeros(1, sr)          # 1 秒靜音
p = os.path.join(tempfile.mkdtemp(), "probe.wav")
torchaudio.save(p, wav, sr, encoding="PCM_S", bits_per_sample=16)
a, got = torchaudio.load(p)
assert got == sr and a.shape[1] == sr
print("OK")
'''
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        print("  torchaudio.load 失敗:", (r.stderr or "").strip().splitlines()[-1][:160])
    return r.returncode == 0

import torch, torchaudio
print(f"現成版本: torch {torch.__version__} / torchaudio {torchaudio.__version__}")

if can_load_wav():
    print("✓ 路徑 1：現成的 torchaudio 就能讀 wav，不動它")
else:
    print("→ 路徑 2：試裝 torchcodec")
    sh(sys.executable, "-m", "pip", "install", "-q", "torchcodec")
    if can_load_wav():
        print("✓ 路徑 2 成功")
    else:
        print("→ 路徑 3：降級 torch / torchaudio 到 <2.9（會花幾分鐘）")
        sh(sys.executable, "-m", "pip", "install", "-q",
           "torch>=2.1,<2.9", "torchaudio>=2.1,<2.9",
           "--index-url", "https://download.pytorch.org/whl/cu121")
        print("⚠ 降級後請重新啟動執行階段（執行階段 → 重新啟動），再從這個 cell 繼續")
"""),

    md("""
### 3b. 其餘套件

`requirements.txt` 裡的 torch 已經處理過了，這裡只裝其餘的。

`numpy<2.0` 是 upstream 的釘選。如果 Colab 的 numpy 是 2.x，pip 會降級它 —— 降級後**必須重新啟動執行階段**，
因為已經 import 過的 numpy 不會換掉。下面會偵測並提示。
"""),
    code("""
import numpy, subprocess, sys
before = numpy.__version__
print("numpy（安裝前）:", before)

# 跳過 torch/torchaudio（上一個 cell 已處理），其餘照 requirements.txt 裝
req = [l.strip() for l in open("requirements.txt", encoding="utf-8")
       if l.strip() and not l.startswith("#")]
req = [l for l in req if not l.split(">")[0].split("=")[0].split("<")[0].strip()
       in ("torch", "torchaudio")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *req], check=False)

after = subprocess.run([sys.executable, "-c", "import numpy;print(numpy.__version__)"],
                       capture_output=True, text=True).stdout.strip()
print("numpy（安裝後）:", after)
if after != before:
    print()
    print("⚠ numpy 版本變了，必須重新啟動執行階段才會生效：")
    print("   執行階段 → 重新啟動工作階段，然後從『2. 取得程式碼』重新執行")
else:
    print("✓ 不需要重新啟動")
"""),

    md("""
## 4. 下載模型權重（約 2.85 GB）

同時會抓英文前端要的 nltk 語料（cmudict 與 averaged_perceptron_tagger，約 20 MB）到
專案內的 `nltk_data/`。少了它，**含英文或數字的句子會在串流中途失敗**（純中文不受影響）——
因為 `text/english.py` 透過 `g2p_en` 需要這兩份語料。
"""),
    code("""
!python3 scripts/start.py --download-only
"""),

    md("""
### 4b. 教育部國語辭典（台灣讀音）

`config.yaml` 預設開了 `zh_taiwan_readings: true` —— 中文讀音改用教育部
《重編國語辭典修訂本》，蓋掉 upstream 的大陸讀音。**這一格不跑的話，
server 會警告並沿用大陸讀音**（`垃圾` 唸 lājī 而不是 lèsè）。

整份 14 MB，**原封不動**取自 [g0v 萌典](https://github.com/g0v/moedict-data)。
辭典正文著作權屬教育部，採 CC BY-ND 3.0 TW：可原封不動重製散布、可商用，
但不得改作。本專案不修改它，也不從它產生衍生資料檔，只在執行時載入記憶體查詢。
"""),
    code("""
!python3 scripts/tw_check.py --download

# 看看這個模式實際在修什麼：比對 upstream 的 45,046 筆大陸讀音與教育部
!python3 scripts/tw_check.py --scan-upstream --limit 12
"""),

    md("""
## 5. 啟動 server

在背景啟動，並等暖機完成。第一次會載入模型、跑 BERT 剪裁的等價性自檢、
再跑一次 dummy 推論把一次性成本付掉 —— GPU 上通常不到一分鐘。
"""),
    code("""
import subprocess, sys, time, json, urllib.request, pathlib, os

PORT = 8000
LOG = pathlib.Path("/content/server.log")
proc = subprocess.Popen(
    [sys.executable, "-m", "gsv_server", "--host", "127.0.0.1", "--port", str(PORT)],
    cwd="server", stdout=open(LOG, "w"), stderr=subprocess.STDOUT)

def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=4) as r:
            return json.load(r)
    except Exception:
        return None

print("暖機中", end="")
t0 = time.time()
info = None
while time.time() - t0 < 900:
    if proc.poll() is not None:
        print("\\n✗ server 結束了，log 如下：")
        print(LOG.read_text()[-3000:])
        raise SystemExit(1)
    info = health()
    if info and info.get("ready"):
        break
    print(".", end="", flush=True)
    time.sleep(5)

print(f"\\n就緒（{time.time()-t0:.0f}s）")
print(json.dumps(info, ensure_ascii=False, indent=1))
q = info["quality"]
print()
print("音質狀態：", "✓ 與 upstream 預設一致" if q["clean"] else "⚠ " + "；".join(q["deviations"]))
print("BERT 剪裁等價性：", "✓ 逐位元驗證通過" if q["bert_trim_verified"] else "未啟用")
"""),

    md("""
## 6. 打開控制台

Colab 看不到 localhost，要透過它自己的 proxy。下面會開一個新視窗。
如果被瀏覽器擋了，改用第二行的 iframe 版本。
"""),
    code("""
from google.colab import output
output.serve_kernel_port_as_window(PORT)      # 開新視窗
# output.serve_kernel_port_as_iframe(PORT, height=1400)   # 或嵌在這裡
"""),

    md("""
## 7. 量真實的 GPU 延遲

這是這個 notebook 最有價值的部分 —— **TTFA 的實測值**。

量的是客戶端收到第一個 byte 的時間，不是伺服器自己回報的數字。
「即時倍率」大於 1 表示生成快於播放，串流可以不中斷。
"""),
    code("""
import time, urllib.parse, urllib.request, statistics, pathlib
from IPython.display import Audio, display

PORT = globals().get("PORT", 8000)   # 重啟後單獨執行這個 cell 也能用

TEXTS = [
    ("短", "你好。"),
    ("中", "慢慢來也沒關係，只要沒有停下來，你其實一直都在前進。"),
    ("長", "明天的天氣預報是：早上會有一點懶洋洋的微風，適合慢慢醒來；"
           "中午陽光剛好，不刺眼，但足夠讓人想出去走走；"
           "下午可能會飄來幾朵心情很好的雲，把天空點綴得剛剛好；"
           "晚上氣溫微涼，適合聽音樂或發呆。"),
    ("中英混合", "這個 server 用 v2Pro 模型，sample rate 是 32000 Hz。"),
]

# 回傳 (TTFA, 總時間, PCM, sample rate)。
# /tts/stream 是 chunked 串流，所以 server 如果在串流「中途」出錯，HTTP 狀態碼
# 早就送出 200 了 —— 客戶端只會拿到 IncompleteRead，看不出原因。這裡把它接住，
# 改成去 server log 撈最後幾行錯誤印出來。
def measure(text):
    url = f"http://127.0.0.1:{PORT}/tts/stream?" + urllib.parse.urlencode({"text": text})
    t0 = time.perf_counter(); first = None; buf = bytearray()
    try:
        with urllib.request.urlopen(url, timeout=600) as r:
            sr = int(r.headers.get("X-Sample-Rate", 32000))
            while True:
                c = r.read(4096)
                if not c: break
                if first is None: first = time.perf_counter() - t0
                buf += c
    except Exception as exc:
        tail = ""
        try:
            lines = pathlib.Path("/content/server.log").read_text(
                encoding="utf-8", errors="replace").strip().splitlines()
            tail = "\\n".join("    " + l for l in lines[-12:])
        except Exception:
            pass
        raise RuntimeError(
            f"串流中斷（{type(exc).__name__}: {exc}）。已收到 {len(buf)} bytes。\\n"
            f"  server log 末尾：\\n{tail}") from exc
    return first, time.perf_counter() - t0, bytes(buf), sr

print("暖身一次（不計入）…"); measure(TEXTS[1][1])
print()
print(f"{'文字':10s} {'字數':>4s}  {'TTFA':>9s} {'總時間':>9s} {'音訊':>8s} {'即時倍率':>9s}")
print("-" * 58)
rows = []
for label, text in TEXTS:
    ttfa, total, pcm, sr = measure(text)
    secs = len(pcm) / 2 / sr
    rows.append((label, text, ttfa, total, secs, pcm, sr))
    print(f"{label:10s} {len(text):4d}  {ttfa*1000:8.0f}ms {total*1000:8.0f}ms "
          f"{secs:7.2f}s {secs/total:8.2f}×")

print()
slow = [r for r in rows if r[4] / r[3] < 1]
if slow:
    print("⚠ 這些長度的即時倍率 < 1，串流會追不上播放：", ", ".join(r[0] for r in slow))
else:
    print("✓ 全部即時倍率 > 1 —— 串流不會中斷")
"""),

    md("### 聽聽看"),
    code("""
import struct
from IPython.display import Audio, display

def wav_bytes(pcm, sr):
    head = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)))
    return head + pcm

for label, text, ttfa, total, secs, pcm, sr in rows:
    print(f"【{label}】TTFA {ttfa*1000:.0f}ms · {secs:.2f}s · {text[:36]}")
    display(Audio(wav_bytes(pcm, sr), rate=sr))
"""),

    md("""
## 8. 發音字典：遇到唸錯的字就改掉

兩種常見的唸錯：

- **英文**：`text/english.py:322` 對「不在 cmudict 裡、長度 ≤ 3」的詞一律**逐字母拼讀**。
  `GPU`→G-P-U 是對的，但 `wav` 會變成 "double-u ay vee"。
- **中文**：g2pw 整句推理，遇到人名與專有名詞會猜錯聲調（例如 `檔案` 被唸成 dǎng）。

下面先查這段文字每個詞實際會怎麼唸，再直接改。**改完立刻生效，不用重啟。**

控制台（上一格那個網頁）也有同樣的「發音字典」介面，用滑鼠點比較快。
"""),
    code("""
import json, urllib.request

PORT = globals().get("PORT", 8000)

def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

TEXT = "我把檔案存成 wav 然後上傳到 server"

for it in api("POST", "/dict/lookup", {"text": TEXT})["items"]:
    cur = " ".join(it.get("pinyin") or it.get("phones") or [])
    tag = ""
    if it["custom"]:
        tag = "← 自訂"
    elif it.get("letter_by_letter"):
        tag = "← 逐字母拼讀，多半是錯的"
    elif it.get("shadows_single_chars"):
        tag = "（整詞條目，改單字不會有反應）"
    print(f'{it["lang"]:3s} {it["word"]:10s} {cur:28s} {tag}')
"""),
    md("""
改一個詞試試看。英文給 ARPAbet 音素（`W EY1 V`），中文給拼音（`dang4 an4`）。
打錯會回 400 並指名是哪個音素 —— 不會默默把音丟掉。

中文的 key 會自動轉成簡體（管線裡本來就會做繁簡轉換），所以打繁體也沒關係。
"""),
    code("""
# 英文：wav 當一個字唸，而不是 W-A-V
print(api("PUT", "/dict", {"lang": "en", "word": "wav", "value": ["W", "EY1", "V"]})["entries"]["en"])

# 中文：檔案 的 檔 是四聲
r = api("PUT", "/dict", {"lang": "zh", "word": "檔案", "value": ["dang4", "an4"]})
print("存成的 key:", r["stored_key"], "（繁體已轉簡體）" if r["normalized"] else "")

# 改完再查一次
for it in api("POST", "/dict/lookup", {"text": TEXT})["items"]:
    if it["custom"]:
        print("已套用:", it["word"], " ".join(it.get("pinyin") or it.get("phones")))

# 要還原成 upstream 原本的讀音：
# api("DELETE", "/dict?lang=zh&word=" + urllib.parse.quote("檔案"))
"""),

    md("""
## 9. 備份（**Colab 關掉就什麼都沒了，這格要跑**）

Colab 的 VM 是暫時的。權重與程式碼隨時能重抓，但**你改的設定與自訂發音重做不回來**。

下面會打包 `server/config.yaml` 與 `server/pronunciation.rep`，然後直接下載到你電腦。
下次重開 Colab，把檔案上傳回來跑 `restore` 就回到原狀。
"""),
    code("""
!python3 scripts/backup.py export --out /content/gsv-backup.tar.gz

from google.colab import files
files.download("/content/gsv-backup.tar.gz")
"""),
    md("""
下次要還原（先把備份檔上傳到 `/content/`）：

```python
from google.colab import files
up = files.upload()                      # 選你的 gsv-backup.tar.gz
!python3 scripts/backup.py list    /content/gsv-backup.tar.gz
!python3 scripts/backup.py restore /content/gsv-backup.tar.gz
```

還原後要重跑「啟動 server」那一格才會生效。
"""),

    md("""
## 10. 換模型 / 換參考音訊（選用）

上傳自己的權重後，用 `add_voice.py` 檢查並安裝。它會從**檔頭**判定版本、
核對 GPT 與 SoVITS 的符號集、檢查參考音訊長度是否在 3–10 秒內。
"""),
    code("""
# from google.colab import files
# up = files.upload()        # 上傳 .pth / .ckpt / .wav

# !python3 scripts/add_voice.py --dry-run --name 新角色 \\
#     --sovits /content/新角色_e8.pth \\
#     --gpt    /content/新角色-e15.ckpt \\
#     --ref    /content/參考.wav \\
#     --ref-text "參考音訊的逐字稿"

# 檢查通過後加 --server http://127.0.0.1:8000 直接套用到執行中的 server
"""),

    md("""
## 11. 停止

Colab 的執行階段結束時 server 會自動結束，通常不需要手動停。
"""),
    code("""
# proc.terminate()
"""),
]


def check_cells() -> int:
    """把每個 code cell 當 Python 編譯一次。

    產生器是「Python 字串裡再放 Python」，所以跳脫很容易弄錯：寫 "\\n"
    才會在 cell 裡得到 "\\\\n"，少一層就變成真正的換行、把字串literal 截斷。
    這種錯只有在 Colab 上執行到那個 cell 才會爆，所以在這裡先編譯一次。

    含 ! 或 % 這種 IPython magic 的 cell 不是合法 Python，跳過。
    """
    bad = 0
    for i, c in enumerate(CELLS):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(l.lstrip().startswith(("!", "%")) for l in src.splitlines()):
            continue
        try:
            compile(src, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            bad += 1
            print(f"  cell {i} 語法錯誤: {exc.msg}（第 {exc.lineno} 行）")
            if exc.text:
                print(f"    {exc.text.strip()[:100]}")
    return bad


def main() -> int:
    bad = check_cells()
    if bad:
        print(f"{bad} 個 cell 語法錯誤，不寫出檔案")
        return 1
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = REPO / "Colab.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"寫出 {out}（{len(CELLS)} 個 cell）")
    print(f"\nColab 連結：")
    print(f"  https://colab.research.google.com/github/"
          f"{REPO_URL.split('github.com/')[1]}/blob/{BRANCH}/Colab.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
