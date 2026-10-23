# tts_model_only_inference

GPT-SoVITS 的**純推論** server。單一目標：**超低延遲**。

主力走 **v2Pro**（單次前向，無 CFM），v4 作為品質比較與離線用途保留。模型可換。邊緣設備以網路連線使用，**不在設備上跑推論**。

> **狀態：server 已實作，並在 CPU 上完成端到端實測（見 §11.1）。尚未在 GPU 上實測。**
> 本文件記錄完整設計決策與依據。所有結論附原始碼行號或官方文件原文，不憑印象。
> 開發環境無 GPU，但足以跑完整條推論鏈：已實際合成出音訊，BERT 剪裁的等價性自檢也通過。
> 仍須在目標機器上實測的是**音質判斷與 GPU 延遲**（見 §11.3）。
> 初版計劃中被實作推翻的結論，都保留在原處標明修正，不無聲刪除。

---

## 目錄

- [0. 範圍](#0-範圍)
- [使用方式](#使用方式)
- [1. 為什麼重做](#1-為什麼重做)
- [2. 洛神模型的事實](#2-洛神模型的事實)
- [3. 為什麼主力是 v2Pro](#3-為什麼主力是-v2pro)
- [4. 設計原則：「不動音質」的操作型定義](#4-設計原則不動音質的操作型定義)
- [5. 低延遲設計](#5-低延遲設計)
- [6. 對外介面](#6-對外介面)
  - [6.0 控制台](#60-控制台)
  - [6.1 裝一組新的模型 / 參考音訊](#61-裝一組新的模型--參考音訊)
  - [6.2 換模型與換參考音訊](#62-換模型與換參考音訊)
- [7. vendoring：用加法，不用刪除法](#7-vendoring用加法不用刪除法)
- [8. 依賴瘦身](#8-依賴瘦身)
- [9. 取樣參數](#9-取樣參數)
- [10. Repo 結構](#10-repo-結構)
- [11. 驗證策略](#11-驗證策略)
- [12. 未決事項](#12-未決事項)
- [13. 引用來源](#13-引用來源)
- [附錄：改這個專案前必讀](#附錄改這個專案前必讀)

---

## 0. 範圍

| | |
|---|---|
| **目標** | 超低延遲語音合成 server |
| **主力版本** | v2Pro（單次前向） |
| **備用版本** | v4（品質比較 / 離線）、v2ProPlus（若取得權重） |
| **模型** | 可換，熱切換 |
| **語言** | 中文 + 英文 |
| **邊緣設備** | 以 HTTP / WebSocket 連線，**不在設備上推論** |
| **不支援** | v3、中英以外語言、訓練、WebUI |

**延遲指標以 TTFA（time-to-first-audio，從收到請求到第一塊音訊送出）為主**，而非總體 RTF。理由見 §5.1。

### 範圍變更記錄

初版計劃曾包含一個「在邊緣設備上離線跑的極簡版」。該範圍已取消 —— 邊緣設備改為連線使用。

這個變更移除了大量複雜度：原本為邊緣版設計的「參考音訊衍生物離線預算成常數檔」，在 server 上用**啟動時暖機 `prompt_cache`** 就能達到完全相同的效果（見 §5.3），而且 upstream 已有現成機制。

---

## 使用方式

### 一鍵啟動

```bash
./start.sh          # Linux / macOS（多數檔案管理器可直接雙擊）
start.bat           # Windows（雙擊）
python3 scripts/start.py    # 或直接執行
```

它會依序：檢查 Python 版本 → 建立虛擬環境 → 偵測 GPU 並裝對應的 torch → 下載權重 → 啟動 server → 開啟控制台。

**會立刻彈出一個進度頁面**，而不是讓你對著終端機等。第一次要下載約 2.7 GB，裝依賴也要幾分鐘，所以進度必須看得見。裝完會自動交棒給真正的 server 並切換到控制台。

```
  0.1s  進度頁
  3.3s  環境就緒，正在交棒給 server…
  4.8s  server 啟動中，正在載入模型並暖機…
 72.7s  控制台（就緒）
```

交棒期間 port 上沒有東西在聽（uvicorn 會先跑完 lifespan 才綁 port），進度頁會耐心等待而不是顯示連線錯誤。

常用參數：

```bash
python3 scripts/start.py --port 9000        # 換 port
python3 scripts/start.py --device cpu       # 強制裝 CPU 版 torch
python3 scripts/start.py --no-browser       # 不自動開瀏覽器
python3 scripts/start.py --setup-only       # 只建環境
python3 scripts/start.py --download-only    # 只下載權重
```

### Colab：不想在本機裝任何東西

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtswwan/tts_model_only_inference/blob/claude/blissful-hawking-safc6x/Colab.ipynb)

沒有 GPU、或只想先聽聽效果的話用這個。Colab 的 T4 是免費的，而這個專案要的就是 GPU。

`Colab.ipynb` 由 `scripts/make_colab.py` 產生（**不要直接改 .ipynb** —— 改產生器再重跑，見 §10）。
整本從上到下執行即可，會依序：檢查 GPU → clone → 處理 torchaudio 相容性 → 裝其餘套件 →
下載權重與 nltk 語料 → 啟動 server → 用 Colab 的 port proxy 開出控制台 → 跑一輪 TTFA 量測並直接在 notebook 裡播放結果。

兩件 Colab 特有的事值得知道：

- **torchaudio**：Colab 預裝的 torch 版本會變。notebook 先**實測** `torchaudio.load` 能不能讀 wav
  （而不是比對版本號），不行才依序試 `torchcodec` → 降級到 `<2.9`。詳見地雷 1。
- **numpy**：upstream 釘 `numpy<2.0`。如果 Colab 當下是 2.x，pip 會降級它，而**已經 import 過的 numpy 不會換掉**
  —— notebook 會偵測並提示你重新啟動執行階段。

產生器自己會把每個 code cell 編譯一次才寫出檔案。因為 cell 的原始碼是「Python 字串裡再放 Python」，
跳脫少一層就會把字串截斷，而那種錯只有在 Colab 上執行到那一格才會爆。

### 可攜版：解壓縮就能用

不想處理環境的話，用這個。產出的資料夾**自帶 Python、自帶套件、自帶權重**，目標機器不需要安裝 Python、不需要 pip、不需要網路。

```bash
python3 scripts/make_bundle.py --out gsv-portable              # CPU 版
python3 scripts/make_bundle.py --out gsv-portable --cuda cu121 # GPU 版
python3 scripts/make_bundle.py --out gsv-portable --archive    # 另外產生 tar.gz
```

複製整個資料夾到目標機器，執行 `啟動.sh`（或 `啟動.bat`）即可。

```
gsv-portable/
  python/      自帶的 CPython 與所有套件
  app/         程式碼與模型權重
    nltk_data/ 英文前端的 nltk 語料（少了它，含英文的句子會失敗）
  啟動.sh / 啟動.bat
  README.txt
```

CPU 版實測 **4.91 GB**（Python 與套件 2.04 GB + 程式碼與權重 2.87 GB），建置約 13 秒（套件已快取時）。

**為什麼不用一般的 venv**：venv 的 `pyvenv.cfg` 與 script 的 shebang 都寫絕對路徑，搬家就壞。這裡用 python-build-standalone（透過 `uv` 取得），它設計上可重定位，套件直接裝進它自己的 `site-packages` 而不是 venv。

**為什麼不用 PyInstaller 打成單一執行檔**：PyTorch 有大量 `.so` 與資料檔，打成單檔會是 2–3 GB 的執行檔、啟動時要解壓到暫存目錄，而且 torch 的動態載入常常打包失敗。一個資料夾更可靠也更好除錯。

實測（建置後搬到完全不同的路徑再執行）：

| | |
|---|---|
| `./啟動.sh` | 56 秒就緒 |
| `/health` | `ready=true`、`quality.clean=true` |
| 合成 | TTFA 3581 ms、2.50 s 音訊 |
| 用的直譯器 | bundle 自帶的 Python 3.11.15 / torch 2.8.0+cpu |
| console script | `pip` / `isympy` / `ttx` 搬家後仍可執行 |

### 需要多少空間

| 項目 | 大小 | 說明 |
|---|---|---|
| `git clone` | **約 26 MB** | 其中 17.3 MB 是中英文前端詞典（`cmudict` 7 MB、`engdict_cache` 5.7 MB、`polyphonic` 2.9 MB）。程式碼本身不到 1 MB |
| 模型權重 | **約 2.85 GB** | BERT 1.3 G、G2PW 608 M、CNHubert 361 M、洛神 GPT 149 M + SoVITS 129 M、fast_langdetect 126 M、SV 103 M |
| nltk 語料 | **約 20 MB** | 英文前端要的 `cmudict` 與 `averaged_perceptron_tagger`，下載到專案內的 `nltk_data/` |
| 虛擬環境（CPU） | **約 1.8 GB** | 實測值 |
| 虛擬環境（CUDA） | 較大 | torch 的 CUDA wheel 另外會拉 `nvidia-*` 相依套件；裝完後控制台會顯示實際佔用 |

CPU 版總計約 **4.7 GB**。

只抓 PyTorch 格式的權重 —— 前一版的 8.25 GB 裡有 4.69 GB（57%）是 TensorFlow / Flax / fairseq 的重複權重（見 §7）。

### 系統需求

- **Python 3.10 ~ 3.12**。不能用 3.13：upstream 釘 `numpy<2.0`，而它在 3.13 沒有 wheel。另外 `jieba_fast` 只有 2019 年的 source tarball，必須就地編譯 C 擴充（3.11 / 3.12 實測可通過）
- Debian / Ubuntu 可能要先 `apt install python3-venv`（venv 在那裡是獨立套件）
- 約 5 GB 磁碟空間
- 能連到 `huggingface.co` 與 `pypi.org`（權重與套件都從那裡抓）
- GPU 非必要。CPU 跑得動但約 0.6–0.8× 即時（見 §11.1）

**ffmpeg 不是硬性需求。** 實測 `torchaudio` 的 `soundfile` 後端可以直接讀 16-bit PCM wav，而 `soundfile` 是 `librosa` 的相依套件一定會裝。`TTS.py:18` 的 `import ffmpeg` 是 pip 套件 `ffmpeg-python`，只有在轉 mp3/aac 時才會呼叫 ffmpeg 執行檔，本專案的路徑不走它。裝了也無妨。

**CUDA 版本自動選擇。** 啟動器讀 `nvidia-smi` 回報的驅動版本來挑 PyTorch channel：驅動 ≥ 525 用 `cu121`，更舊用 `cu118`。寫死 `cu121` 會在舊驅動的機器上裝完才發現跑不起來。要覆寫用 `--cuda cu118`（或 `cu124`、`cpu`）。

### 手動安裝

```bash
pip install -r requirements.txt
scripts/fetch_models.sh          # 等同 start.py --download-only
cd server && python -m gsv_server
```

啟動只有這一條路徑。**不要用 `uvicorn gsv_server.api:app`** —— `app` 需要帶著設定建構，而 `bootstrap` 必須在 import upstream 之前完成。前一版正是因為「用什麼指令啟動」會改變實際執行的程式碼而出問題（§1.1）。

### API

```bash
# 完整音檔
curl -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "慢慢來也沒關係，只要沒有停下來，你其實一直都在前進。"}' \
  -o out.wav

# 串流 raw PCM（s16le 單聲道，取樣率在 X-Sample-Rate 標頭）
curl -N 'http://localhost:8000/tts/stream?text=你好' -o out.pcm

# 狀態與延遲量測
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

WebSocket `/tts/ws`：送 `{"text": "..."}`，收到 `{"event":"begin",...}`、多個二進位 PCM 區塊、最後 `{"event":"end","ttfa_ms":...}`。

控制台見 §6.0，換模型 / 參考音訊見 §6.1 與 §6.2。

### 量延遲

```bash
python scripts/bench.py --url http://localhost:8000 --profiles quality
```

量的是**客戶端收到第一個 byte 的時間**，不是伺服器自己回報的數字。同時回報「即時倍率」—— 大於 1 表示生成快於播放，串流可以不中斷。

### 設定

`server/config.yaml`，任何一項都可用環境變數覆寫（雙底線代表層級）：

```bash
GSV_LATENCY__PROFILE=low GSV_SERVER__PORT=9000 python -m gsv_server
```

未知的設定鍵會直接報錯，不會被安靜忽略。

---

## 1. 為什麼重做

前一版（`Sihrrieidhr/GPT-SoVITS-Lite`）不是「風格不好」，是有會真的出錯的地方。

### 1.1 `inference_server.py` 是兩個完整程式被接在同一個檔案裡

第 87 行：

```python
    uvicorn.run(app, host="0.0.0.0", port=8000)#!/usr/bin/env python3
```

後面那個 shebang 被當成行尾註解吃掉，第 88 行開始**又是一支完整的程式**，重新定義了 `app`、`InferRequest`、`InferResponse`、`startup_event`、`infer`、`health`，以及一整個 330 行的 `InferenceEngine`。

417 行裡 330 行（79%）是死碼 —— 但**哪一半是死的取決於啟動方式**：

| 啟動方式 | 實際執行 | 原因 |
|---|---|---|
| `python3 inference_server.py` | 前半（1–87 行） | 第 87 行 `uvicorn.run()` 就地阻塞，後 330 行永不執行 |
| `uvicorn inference_server:app` | **後半** | `__name__ != "__main__"`，整檔執行完，後半定義覆蓋前半 |

兩半是**不同的推理引擎、不同的 bug**。同一份檔案換個啟動指令就換一套行為。

### 1.2 後半的參考音訊取樣率錯誤（`inference_server.py:306-314`）

```python
if sr != 16000: ... 重採樣到 16000
refer, _ = spectrogram_torch(aud_t, filter_length, self.hps.data.sampling_rate, ...)
```

音訊被重採樣到 16 kHz，但 STFT 參數餵的是 `hps.data.sampling_rate`（v4 為 32 kHz）。頻譜與取樣率不匹配 → 音色偏移。

對照 upstream `GPT_SoVITS/inference_webui.py:517` 的 `get_spepc()`：它重採樣到 `hps.data.sampling_rate` 才算頻譜，並有 `if maxx > 1: audio /= min(2, maxx)` 的音量保護（後半也漏了）。

`standalone_inference.py` 的 `get_spec()` 這段是對的，**只有 server 的後半寫錯**。

### 1.3 最吃記憶體的那段沒包 `no_grad()`（`standalone_inference.py:226-284`）

`torch.no_grad()` 只包了 SSL（193 行）、GPT（215 行）、vocoder（288 行）。**SoVITS decode + CFM 迴圈完全沒包**，而那是整條 pipeline 最重的部分，還是個迴圈：

```python
cfm_res.append(cfm)       # 這些 tensor 都帶著 autograd graph
mel2 = cfm[:, :, -Tmin:]  # 而且一輪接一輪串起來
```

graph 跨 chunk 累積，長文本 OOM。

### 1.4 API 設計問題

- `output` 是**客戶端傳入的路徑**，server 綁 `0.0.0.0` 後照著寫檔 → 任意檔案寫入
- API **只回傳伺服器端路徑字串，不回傳音檔 bytes** → 跨機器呼叫拿不到音訊
- 例外一律回 HTTP 200 + `str(e)`，洩漏內部錯誤
- 無併發控制：全域 engine、無鎖無佇列

### 1.5 其他

| 問題 | 位置 |
|---|---|
| 預設路徑寫死在別人機器上（`/home/ubuntu/project/...`） | `standalone_inference.py:310-313` |
| 語言寫死中文，多語言前端完全用不到 | `standalone_inference.py:60` |
| 無長文切句，長文本撞 `early_stop_num` 被截斷 | — |
| 角色模型檔名寫死 | `standalone_inference.py:120, 132` |
| `.half()` 後下一行馬上 `.float()` | `inference_server.py:156-158` |
| `CUDA_VISIBLE_DEVICES='0,1'` 寫死在原始碼 | `inference_server.py:122` |
| 每次請求重新載入參考音訊 2–3 次 | `standalone_inference.py:194, 226, 231` |
| 69 個 `__pycache__/*.pyc` 被 commit | — |
| `requirements.txt` 是 upstream 原封不動抄的（含 `tensorboard`、`pytorch-lightning`、`gradio`、`funasr`、`modelscope`、`ctranslate2`） | — |
| README 描述的 `inference_webui.py (711行)` 不存在，只剩它的 `.pyc` | — |
| README「方法二」的版本釘選與 `requirements.txt` 衝突（`transformers==4.37.2` vs `>=4.43`；`librosa==0.10.1` vs `==0.10.2`） | — |
| 權重 4.69 GB / 8.25 GB（57%）是多餘的 | 見 §7 |

### 1.6 結論：基底換掉

**不沿用那支手寫腳本，改用 upstream 的 `GPT_SoVITS/TTS_infer_pack/TTS.py`。**

理由：`TTS.py` 是 upstream 維護的推論引擎，已正確處理參考頻譜取樣率、`no_grad`、長文切句、v4 的 LoRA merge、v2Pro 的 `sv_emb`、以及**串流**。那支腳本是把這些重新手寫一遍然後寫錯幾處。用 `TTS.py` 等於繼承 upstream 的正確性。

對低延遲而言還有一個決定性理由：`TTS.py` 的 `prompt_cache` 與 `decode_streaming` 正是達成低 TTFA 所需的兩個機制（§5）。

---

## 2. 洛神模型的事實

以下全部從 checkpoint 直接解出（本環境無 torch，用 zipfile + 自訂 Unpickler 讀 metadata）。

### 2.1 GPT（s1）`洛神-e15.ckpt` — 155 MB

```
top-level keys : ['weight', 'config', 'info']
weight         : 295 tensors
info           : 'GPT-e15'
config.pretrained_s1               : 'GPT_SoVITS/pretrained_models/s1v3.ckpt'
config.output_dir                  : 'logs/洛神/logs_s1_v4'
config.train.half_weights_save_dir : 'GPT_weights_v4'
config.train.epochs                : 15
config.train.precision             : '16-mixed'
config.data.max_sec                : 54
config.model.n_layer               : 24
config.model.hidden_dim            : 512
config.model.embedding_dim         : 512
config.model.head                  : 16
config.model.linear_units           : 2048
config.model.phoneme_vocab_size    : 732
config.model.vocab_size            : 1025
config.model.EOS                   : 1024
config.inference.top_k             : 15
```

重點：

- `phoneme_vocab_size = 732` → 文字前端用 **v2 符號集**（`symbols2.py`）
- **s1（GPT）在 v2Pro 與 v3/v4 之間共用**（兩者底模都是 `s1v3.ckpt`），所以這份 ckpt 兩個版本都能用。只有 SoVITS 那半要對應版本
- `n_layer = 24`、`hidden_dim = 512` → 這是 AR 解碼的成本來源，也是延遲的主要項（§5.1）

### 2.2 SoVITS v2Pro `洛神_e8.pth` — 134.9 MB 【主力】

來源：`Sihrrieidhr/haah`

```
檔頭前 2 bytes : b'05'  →  head2version: ['v2', 'v2Pro', False]   ← is_lora = False
lora_rank      : 無（完整權重）
weight         : 678 tensors
info           : '8epoch_296iteration'
```

tensor 分布：

| 前綴 | 數量 | 作用 |
|---|---|---|
| `enc_p` | 235 | 文字 / semantic 編碼器 |
| `dec.` | 290 | 解碼器（640× 上採樣，單次前向直出波形） |
| `flow` | 124 | normalizing flow |
| `ref_enc` | 18 | 參考編碼器 |
| **`sv_emb`** | 2 | **SV 聲紋嵌入投影** |
| `quantizer` | 4 | codebook |

無 `enc_q`（訓練用後驗編碼器，`savee()` 會跳過）、無 `cfm`、無 LoRA。

config（與 upstream `configs/s2v2Pro.json` 架構參數**完全吻合**）：

```
model.version              = 'v2Pro'        data.sampling_rate  = 32000
model.gin_channels         = 1024           data.filter_length  = 2048
model.inter_channels       = 192            data.hop_length     = 640
model.hidden_channels      = 192            data.win_length     = 2048
model.filter_channels      = 768            data.n_mel_channels = 128
model.n_heads              = 2              data.n_speakers     = 300
model.n_layers             = 6
model.kernel_size          = 3              save_weight_dir = 'SoVITS_weights_v2Pro'
model.p_dropout            = 0.0            name            = '洛神'
model.resblock             = '1'
model.resblock_kernel_sizes     = [3, 7, 11]
model.resblock_dilation_sizes   = [[1,3,5], [1,3,5], [1,3,5]]
model.upsample_rates            = [10, 8, 2, 2, 2]
model.upsample_initial_channel  = 512
model.upsample_kernel_sizes     = [16, 16, 8, 2, 2]
model.semantic_frame_rate       = '25hz'
```

兩個關鍵推論：

1. **`upsample_rates` 乘積 = 10×8×2×2×2 = 640 = `hop_length`** → 單次前向直出 32 kHz 波形，**不需獨立 vocoder**
2. **`gin_channels = 1024` = 512（`ref_enc`）+ 512（`sv_emb` 投影）** —— 這就是 v2Pro 與 v2 的差異所在（v2 為 512）。權重裡確實有 `sv_emb.weight` / `sv_emb.bias` 佐證

註：`config.train.lora_rank = '32'` 是訓練期殘留欄位（字串型別），權重中無任何 LoRA tensor，不生效。

### 2.3 SoVITS v4 `洛神_e8.pth` — 75.5 MB 【備用】

來源：`Sihrrieidhr/GPT-SoVITS-Lite`（與 2.2 同檔名，不同檔案）

```
檔頭前 2 bytes : b'04'  →  head2version: ['v2', 'v4', True]   ← is_lora = True
lora_rank      : 32
weight         : 258 tensors（其中 176 個是 LoRA）
info           : '8epoch_296iteration'
```

LoRA 只掛在 CFM 的 estimator attention 上（`cfm.base_model.model.estimator.transformer_blocks.N.attn.to_{q,k,v}` / `to_out.0`）。

**這份只含 LoRA 增量 + quantizer codebook，不是完整 SoVITS** → 執行時需額外載入 v4 底模 `s2Gv4.pth`（769 MB）後 merge，再加 `vocoder.pth`（58 MB）。

### 2.4 版本標記的判定依據

GPT-SoVITS 把 zip 的前兩個 byte 換成版本標記（`process_ckpt.py:30-38` 的 `my_save2()`），讀取時再補回 `PK`（`:129-138` 的 `load_sovits_new()`）。對照表在 `process_ckpt.py:72-80`：

```python
head2version = {
    b"00": ["v1", "v1", False],      b"04": ["v2", "v4", True],
    b"01": ["v2", "v2", False],      b"05": ["v2", "v2Pro", False],
    b"02": ["v2", "v3", False],      b"06": ["v2", "v2ProPlus", False],
    b"03": ["v2", "v3", True],
}   # [文字版本, 模型版本, is_lora]
```

---

## 3. 為什麼主力是 v2Pro

### 3.1 架構差異（取自 `TTS_infer_pack/TTS.py` 實際分支）

| | **v2Pro** 【主力】 | **v4** 【備用】 |
|---|---|---|
| Synthesizer | `SynthesizerTrn`（`TTS.py:539`） | `SynthesizerTrnV3`（`TTS.py:548`） |
| semantic → 音訊 | **單次前向**，`use_vocoder = False`（`TTS.py:545`） | CFM flow-matching **ODE 預設 32 步**（`sample_steps`，`TTS.py:1057`），再過獨立 HiFiGAN vocoder |
| 輸出取樣率 | 32 kHz | 48 kHz |
| 權重需求 | **單檔 134.9 MB，完整** | LoRA 75.5 MB + 底模 769 MB + vocoder 58 MB，需 merge |
| 載入成本 | 直接載入 | 載底模 → 套 LoRA → `merge_and_unload()` |
| 額外程式碼 | `eres2net/`（88 K）+ `sv.py`（1.2 K） | `f5_tts/`（68 K，CFM 的 DiT） |
| 額外依賴 | — | `peft`、`x_transformers` |
| **upstream 串流支援** | **有**（`TTS.py:1068`、`:1431` `decode_streaming`） | **無** |

對低延遲而言的關鍵：

- **那個 8**：v4 每個 chunk 要跑 8 次 DiT 前向，再過 480× 上採樣到 48 kHz 的 vocoder。v2Pro 一次前向出波形
- **串流只在 v2Pro 這條路上有**（`TTS.py:1068` 的條件是 `streaming_mode and not self.configs.use_vocoder`）。沒有串流就沒有低 TTFA
- **v2Pro 權重完整**，省掉 769 MB 底模載入與 LoRA merge

### 3.2 品質：upstream 自己說 v2Pro 更好

GPT-SoVITS 官方 README「V2Pro Release Notes」原文：

> **"Slightly higher VRAM usage than v2, surpassing v4's performance, with v2's hardware cost and speed."**

以及同段：

> **"v1/v2 and the v2Pro series share the same characteristics, while v3/v4 have similar features. For training sets with average audio quality, v1/v2/v2Pro can deliver decent results, but v3/v4 cannot. Additionally, the synthesized tone and timebre of v3/v4 lean more toward the reference audio rather than the overall training set."**

兩個含義：

1. **v3/v4 挑訓練資料**。資料品質中等時 v3/v4 撐不住，v1/v2/v2Pro 還能給出像樣結果
2. **v4 的音色偏向「參考音訊」，v2Pro 偏向「整個訓練集」**。對 fine-tune 出來的角色聲音，通常要的是訓練出的音色穩定重現，而非跟著每次的參考片段跑

所以 **v2Pro 不是「快但差一點」的妥協** —— 依官方說法它品質更好、更快、權重更小、且是唯一支援串流的那條。

### 3.3 保留意見

- 以上是**官方說法，不是本專案實測**。最終要用耳朵判斷，這是保留 v4 支援的理由
- v4 原生 48 kHz，v2Pro 32 kHz。但人聲能量基本在 8–16 kHz 以下，32 kHz 已完整涵蓋語音頻帶；48 kHz 的差別主要在寬頻音樂素材。**對 TTS 而言 32 kHz 不是品質瓶頸**，除非下游有 48 kHz 檔案規格要求
- **v2ProPlus 比 v2Pro 更好且仍是單次前向**。若取得權重應一併支援並實測比較

---

## 4. 設計原則：「不動音質」的操作型定義

這是本專案最硬的約束，所以要定義清楚。

**只做可證明等價的優化。** 分三類：

### ✅ 類別一：可證明等價 —— 會做

輸出張量在數值上完全相同，不是近似：

| 優化 | 為何等價 |
|---|---|
| 啟動時暖機 `prompt_cache`（§5.3） | 同樣輸入經同樣模型得同樣張量。只是算的時機從請求中移到啟動時 |
| 啟動時跑 dummy 推論暖機（§5.4） | 不影響任何輸出，只是把一次性初始化成本挪到啟動 |
| BERT 截掉第 23、24 層（§5.5） | `TextPreprocessor.py:197` 取 `hidden_states[-3]` = 第 22 層輸出。後兩層結果從未被讀取 |
| 移除 BERT 的 MLM 輸出頭（§5.5） | 21128×1024 的詞表投影，輸出從頭到尾沒被使用 |
| 英文不呼叫 BERT（§5.6） | `TextPreprocessor.py:215-218` 對非中文**本來就回傳 `torch.zeros`**。不呼叫與呼叫後丟棄等價 |
| 文字前端結果按文本快取（§5.7） | 同一文本 → 同一音素序列與 BERT 特徵 |
| 刪除訓練 / 匯出 / 其他介面程式碼（§7） | 推論路徑不經過 |

### 這些約束是程式強制的，不只是預設值

「預設正確」不夠 —— 音質問題不會出現在任何 log 或 metric 上，所以有三條路可以悄悄繞過約束，三條都堵了。

**1. BERT 剪裁的等價性在啟動時實際驗證，不通過就拒絕啟動**

剪裁前先用五段探針文字跑一次原模型留下輸出（取法與 `TextPreprocessor.py:196-197` 完全相同），剪完後逐位元比對。**不是逐位元相同就 raise `TrimEquivalenceError` 並拒絕啟動**，而不是印個警告繼續跑。

不給容差是刻意的：兩次前向是同樣的輸入、權重與運算順序，結果本來就該完全相同。有差異就代表剪裁邏輯不成立（最可能是 `hidden_states` 的負索引補齊沒對上，例如換了 transformers 版本），而不是浮點誤差。寧可開不起來，也不要悄悄變差。

失敗時訊息會指示設 `latency.trim_bert: false` 重新啟動 —— 那只會變慢，不會變差。

**2. 單一請求不能要求 `profile=low`**

`latency.allow_low_profile` 預設 `false`。否則任何呼叫端送 `{"profile":"low"}` 就能拿到 upstream 自評 Medium quality 的輸出，而且不留痕跡。要開放必須由 server 端明確設定。HTTP 回 **403**，WebSocket 回同樣的理由。

若 `latency.profile` 本身就設成 `low`，那是 server 擁有者的明確選擇，不受此限。

**3. 任何偏離 upstream 預設的設定會在啟動時列出來**

涵蓋 `profile`、`fixed_length_chunk`、`text_split_method`、`speed_factor`、以及三個取樣參數。例如 `speed_factor` 一旦不是 `1.0`，`TTS.py:1099` 會強制關閉 `split_bucket` 並對輸出音訊重採樣 —— 這種連帶效果不該安靜發生。

同一份資訊也出現在 `GET /health` 的 `quality` 區塊，可以從外部確認，不必翻 log：

```json
{
  "quality": {
    "clean": true,
    "bert_trim_verified": true,
    "bert_trim_enabled": true,
    "allow_low_profile": false,
    "deviations": []
  }
}
```

`clean: true` 表示所有會影響輸出的設定都與 upstream 預設一致，且 BERT 剪裁已通過等價自檢。

### ❌ 原計劃中被否決的項目

以下是初版計劃列為「會做」、但實作時查證後證明做不到或代價不符的。留在這裡
是為了不讓被推翻的結論無聲消失。

| 原計劃項目 | 為何否決 |
|---|---|
| 拆除 `pytorch-lightning` 包裝 | **做不到。** `TTS.py:24` import 的是 `AR/models/t2s_lightning_module.py` 的 `Text2SemanticLightningModule`，而它繼承 `LightningModule`。要拆就得改 upstream 的檔案。所幸 PL 只在啟動時 import 一次，**不影響 TTFA**，保留它對低延遲目標沒有損失 |
| 移除 `torchmetrics` | **做不到。** `AR/models/t2s_model.py` 直接 import，是 AR 模型本體在用，不是訓練殘留 |
| 移除 `matplotlib` | **做不到。** `BigVGAN/utils0.py`、`GPT_SoVITS/utils.py`、`AR/modules/lr_schedulers.py` 三個都在可達集合內 |
| 移除 `x_transformers` | **做不到。** `module/models.py:13` 無條件 `from f5_tts.model import DiT`，連只跑 v2Pro 也躲不掉 |
| 砍 `BigVGAN/` | **做不到。** `TTS.py:25` 無條件 import，與是否使用 v3 無關。程式碼留（364 K），權重不下載 |
| 砍 `tools/AP_BWE_main/` | **做不到。** `TTS.py:33` import `tools/audio_sr.py`，後者 `sys.path.append` 後載入它 |
| 砍 `module/{ddp_utils,distrib}.py` | **做不到。** `core_vq.py:42-43` 直接 import，不是訓練專用 |
| 用自寫分段器取代 `LangSegmenter` | 見下方類別二。另外實作時發現 `langsegmenter.py` 已有 `full_en()` / `full_cjk()` 的 regex 快速路徑，純中文或純英文根本不會呼叫 lid 模型，省下的只有混語情境 |

### ⚠️ 類別二：會改變輸出 —— 不做，或做成預設關閉的明確選項

| 項目 | 為何 |
|---|---|
| 砍 G2PW（635 MB）改用 pypinyin fallback | 中文多音字消歧變差。`chinese2.py:28` 寫死 `is_g2pw = True`，`:192` 有 fallback 但品質較低。**決定：留 G2PW** |
| 用自寫 zh/en 分段器取代 `LangSegmenter` + `lid.176.bin`（131 MB） | 混語分段影響音素序列 → 影響音訊。且 `langsegmenter.py` 已有 `full_en()` / `full_cjk()` regex 快速路徑，純中文或純英文**根本不會呼叫 lid 模型**。**決定：留著** |
| int8 量化 | 音質風險。**決定：不做** |
| **更細的切句粒度換取更低 TTFA** | 切句改變送進 GPT 的文本 → 改變韻律。**做成明確參數，預設跟隨 upstream**（§5.8） |
| 降低 v4 的 CFM 步數（8 步） | 直接動音質。**不做** |
| `parallel_infer` 批次推論 | padding 可能改變結果。**做成選項，預設關** |

### ❓ 類別三：需實測 —— 做成選項

| 項目 | 說明 |
|---|---|
| fp16 vs fp32 | upstream GPU 路徑預設 fp16（`is_half`），數值上與 fp32 不同。可切換，預設跟隨 upstream |
| `torch.compile` / CUDA graph | 對 AR decode 有幫助但首次編譯慢，且可能有數值差異。選項，預設關 |
| 取樣參數 | 見 §9 |

---

## 5. 低延遲設計

### 5.1 延遲預算：先搞清楚時間花在哪

一次請求的計算鏈：

| 階段 | 內容 | 成本特性 |
|---|---|---|
| **1. 文字前端** | jieba 分詞 → G2PW onnx（中文多音字）→ BERT 22 層 | 每句一次，**可快取** |
| **2. AR GPT 解碼** | 每秒音訊 25 個 token（`semantic_frame_rate = '25hz'`），每 token 一次 24 層 × 512 維 forward（kv-cache） | **固有序列化，無法平行，延遲主項** |
| **3. SoVITS 解碼** | `enc_p` → `flow` → `dec`（640× 上採樣）單次前向 | 一次，可分 chunk |

**結論：AR 解碼是延遲主項，而且它天生是序列的。** 無法靠平行化解決。

因此優化方向不是「讓總時間變短」，而是**讓第一塊音訊盡早送出** —— 這就是為什麼指標用 TTFA 而不是總體 RTF。

```
TTFA ≈ 文字前端 + (第一個 chunk 的 token 數 × 每 token 時間) + 該 chunk 的 SoVITS 解碼
```

分母上的每一項都要壓，而且 **chunk 要小**。

### 5.2 串流：用 upstream 的實作，但它有品質代價

**必須先修正初版計劃的一個錯誤說法。** 初版寫「串流用 upstream 現成的 `decode_streaming` 即可，不影響音質」—— 這是錯的。upstream 在 `TTS.py:1026-1033` 對自家參數的標註寫得很清楚：

```
"return_fragment": False,     # (Best Quality, Slowest response speed. old version of streaming mode)
"streaming_mode": False,      # return audio chunk by chunk. (Medium quality, Slow response speed)
"fixed_length_chunk": False,  # (lower quality, faster response speed)
```

`streaming_mode` 被作者自己標為 **Medium quality**。推測原因是 `decode_streaming` 以 `overlap_length` 預設僅 2 個 semantic token 的重疊分塊解碼，VITS decoder 每塊看到的上下文比全序列解碼少，邊界會有瑕疵。

反而 `return_fragment` 被標為 **Best Quality** —— 它是逐段完整解碼後逐段吐出。

更關鍵的是，模式選擇會**換掉 AR 解碼的實作**（`TTS.py:1066-1083`）：

| 設定 | AR 實作 |
|---|---|
| `parallel_infer=True, streaming_mode=False`（upstream 預設） | `infer_panel_batch_infer` |
| `streaming_mode=True`（強制 `parallel_infer=False`） | `infer_panel_naive` |
| 兩者皆關 | `infer_panel_naive_batched` |

三條是不同的程式碼路徑，批次與 padding 行為不同，即使固定 seed 也可能產生不同音訊。

此外 `return_fragment` 與 `streaming_mode` 互斥（`TTS.py:1087-1089` 會強制關掉前者），而兩者都會強制關閉 `split_bucket`（`:1091-1093`）。

**因此本專案的延遲梯度如下，並以 profile 明確區分：**

| profile | 設定 | upstream 自評品質 | TTFA |
|---|---|---|---|
| （都關，非本專案選項） | — | 基準 | 最差，要等全文 |
| **`quality`（預設）** | `return_fragment=True` + `parallel_infer=True` | **Best Quality** | 第一段合成完 |
| `low` | `streaming_mode=True`（`parallel_infer` 被強制關） | Medium quality | 更低 |
| `low` + `fixed_length_chunk` | 同上再加該旗標 | lower quality | 最低 |

預設 `quality`：`return_fragment=True` 且保持 `parallel_infer=True`，使 AR 走 `infer_panel_batch_infer` —— **與 upstream 預設同一條路徑**，同時獲得逐段串流。

`low` 必須明確設定才會啟用。不為了延遲偷偷犧牲音質。

另外注意：`streaming_mode` 只在 `not self.configs.use_vocoder` 時可用（`TTS.py:1068`），也就是**只有 v2Pro / v2 這條路有**。v4 設了會被 upstream 自動退回 `return_fragment`（`:1071-1073`），所以本專案在設定驗證階段就直接拒絕 `v4 + profile=low` 的組合。

註：`GPT_SoVITS/stream_v2pro.py` **不採用**。它第一行自述「这是一个实验性质的实现」，且建構在 `torch.jit` 匯出之上。

### 5.3 啟動時暖機 `prompt_cache`：把兩個模型從請求路徑上移除

追 `TTS.py` 的 `prompt_cache`，**參考音訊只產生四樣東西**：

| 快取內容 | 來源 | 需要哪個模型 | 行號 |
|---|---|---|---|
| `prompt_semantic` | SSL 特徵 → `vq.extract_latent` | **CNHubert（378 MB）** | `TTS.py:833` |
| `refer_spec` | 32 kHz 頻譜 | librosa / torchaudio | `TTS.py:763-776` |
| `sv_emb` | eres2net 聲紋嵌入 | **SV 模型** | `TTS.py:1259-1264` |
| prompt 的 `phones` + `bert_features` | 參考文本 | BERT | `TTS.py:1164-1165` |

最終合成只吃這些：

```python
# TTS.py:1316-1317 / 1332-1333
audio = self.vits_model.decode(pred_semantic, phones, refer_audio_spec,
                               speed=speed_factor, sv_emb=sv_emb)
```

**只要參考音訊不變，這四樣永遠是同一組常數。** 在啟動時算完填進 `prompt_cache`：

- 請求路徑上**完全不碰 CNHubert 和 SV 模型**
- 請求路徑上**完全沒有音檔 I/O、librosa 載入、重採樣**（前一版每次請求讀參考音訊 2–3 次）
- 零品質損失（張量完全相同）

快取失效條件見 `TTS.py:1133-1134`（參考音訊路徑變更，或 v2Pro 的 `refer_spec[0][1] is None`）。切換模型 / 參考音訊時重新暖機。

### 5.4 啟動時跑 dummy 推論

第一次真實推論會吃掉 CUDA kernel 編譯、cuDNN autotune、onnxruntime session 初始化、jieba 詞典載入等一次性成本。啟動時跑一次完整的 dummy 請求把這些付掉，避免第一個真實請求出現延遲尖峰。

### 5.5 BERT 的兩處浪費

1. **第 23、24 層算完從未被讀取**。`TextPreprocessor.py:197`：
   ```python
   res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
   ```
   `hidden_states[-3]` 對 24 層模型 = 第 22 層輸出（`hidden_states` 有 25 個元素：embeddings + 24 層）。
   → **截斷 encoder 到 22 層**，省 2/24 的 BERT 時間

2. **用 `AutoModelForMaskedLM` 載入，但 MLM 輸出頭從未被使用**。那是 21128×1024 的詞表投影。
   → **整個移除**

另外 `BertModel` 預設會算 `pooler_output`，也沒被使用 → 一併移除（`base.pooler = None`，`BertModel.forward` 對 `None` 有判斷）。

**初版計劃估的「約 47 M 參數」偏高，要修正。** `chinese-roberta-wwm-ext-large` 的詞表投影權重與 input embeddings 是綁定的（`tie_word_embeddings`），所以移除 MLM 頭省下的是 transform 那一層與 bias，不是整個 21128×1024 的詞表矩陣。實際省下的是「2 層 encoder + MLM transform + pooler」。

不過**跳過 MLM 頭的每次請求計算省下來了** —— 那個 1024→21128 的矩陣乘法是按序列長度計費的，即使權重與 embeddings 共用也還是要算。

確切數字由 `scripts/verify_equivalence.py` 實測印出，不在此寫死估計值。

**負索引的陷阱**：單純把 encoder 截到 `num_layers - 2` 層是**不等價**的。截完以後 `hidden_states` 只有 `num_layers - 1` 個元素，`[-3]` 會指到更前面的層。所以截層後必須把 `hidden_states` 補回原本的長度（重複最後一個元素），`[-3]` 才仍落在同一層的輸出上。實作見 `server/gsv_server/bert_trim.py`。

### 5.6 英文不走 BERT

`TextPreprocessor.py:211-219`：

```python
def get_bert_inf(self, phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
    else:
        feature = torch.zeros((1024, len(phones)), dtype=torch.float32).to(self.device)
    return feature
```

純英文句子的 BERT 特徵**本來就是零張量**。→ 純英文輸入完全跳過 BERT。

### 5.7 參考音訊聲紋（sv_emb）快取

`sv_emb` 只由參考音訊決定，但 upstream 在**逐段合成的迴圈裡每次重算**（`TTS.py:1259-1264`）：

```python
sv_emb = [] if self.is_v2pro else None
...
    if self.is_v2pro:
        sv_emb.append(self.sv_model.compute_embedding3(audio_tensor))
```

而 `audio_tensor` 是 `_get_ref_spec()` 建好之後一直重用的同一個物件。

CPU 實測 `compute_embedding3` 一次 **826 ms** —— 長文切成兩段就是每個請求白做 1.65 秒，其中第一段的 826 ms 直接算在 TTFA 裡。

以輸入張量的物件識別為鍵做快取（同一個輸入 → 同一個輸出，等價）。字典持有該張量的強參照避免 `id()` 被重用，並額外做 `is` 檢查。參考音訊更換時清空。

`/metrics` 的 `sv_cache` 會回報命中率。

### 5.8 文字前端快取

同一文本的音素序列與 BERT 特徵是確定性的 → 按文本內容快取 `(phones, bert_features, norm_text)`。對重複句型（提示語、固定回應）效果顯著，可讓階段 1 歸零。

### 5.9 切句粒度：延遲與韻律的取捨

TTFA 與第一個 chunk 的 token 數成正比 → 切得細則延遲低。

**但切句會改變送進 GPT 的文本，進而改變韻律。** 這屬於 §4 類別二。

**決定：做成明確參數，預設跟隨 upstream 的切句策略**（`text_segmentation_method.py`）。文件中記錄取捨，由實測決定是否調整。不偷偷為了延遲犧牲韻律。

### 5.10 傳輸層

- **直接送 raw PCM**，不在請求路徑上做 WAV / 其他編碼
- chunk 盡量小（與 §5.8 的切句粒度是不同層次的事：這裡只影響網路傳輸，不影響合成）
- WebSocket 或 HTTP chunked，由客戶端選擇

### 5.11 併發

- **單 worker + 請求佇列**，GPU 操作序列化
- 理由：GPU 上多請求並行會互相拖慢，且前一版的全域無鎖 engine 會直接對撞
- 低延遲的前提是**可預測**的延遲，佇列比爭搶更可預測

---

## 6. 對外介面

| 端點 | 行為 |
|---|---|
| `POST /tts` | 合成並**直接回傳音檔 bytes** |
| `GET /tts?stream=1` 或 WebSocket | **chunked raw PCM 串流**，供低延遲播放 |
| `POST /model` | 切換模型 / 版本，切換後自動重新暖機（§5.3、§5.4） |
| `POST /reference` | 切換參考音訊 / 參考文本，自動重新暖機 |
| `GET /` | 網頁控制台（見 §6.0） |
| `GET /runtime` | 環境偵測：torch / CUDA 裝置 / CPU 執行緒 |
| `GET /candidates` | 掃描可用的模型與參考音訊 |
| `GET /health` | 狀態、當前模型與版本、裝置、暖機是否完成，以及 `quality` 區塊（見 §4） |
| `GET /metrics` | TTFA、每 token 時間、佇列長度 |

### 6.0 控制台

server 在 `GET /` 提供一個網頁控制台。**單一 HTML 檔、零外部依賴**（不連任何 CDN）—— 它必須能在離線的 GPU 機器上操作本機 server。

```
http://<server>:8000/
```

| 區塊 | 內容 |
|---|---|
| **首字延遲 TTFA** | 主視覺。最近一次 / 中位數 / 最小值 / 即時倍率，加上歷史長條圖 |
| 環境 | 平台、Python、torch/torchaudio、CUDA 裝置（名稱、VRAM、sm、fp16 可用性）、CPU 執行緒 |
| 運算裝置 | 下拉切換 `auto` / `cpu` / `cuda:N` + fp16 開關，套用後自動重建並暖機 |
| 模型與參考音訊 | 從 `pretrained_models/` 掃出的候選，SoVITS 標註檔頭判定的版本，參考音訊標註長度與是否在 3–10 秒內 |
| 合成 | 輸入文字 → 播放 → 下載 WAV |
| 音質狀態 | `clean` / BERT 剪裁是否通過逐位元驗證 / profile / 切句方式 / 偏離清單 |
| 執行統計 | 執行時間、請求數、產生的音訊總長、模型載入時間、兩個快取的命中率 |
| 最近的請求 | 字數、TTFA、總時間、音訊長度、即時倍率（< 1 標黃） |

**TTFA 是在瀏覽器端量的**，用 `fetch` 讀 `/tts/stream` 的第一個 chunk，不是採信伺服器自己回報的數字 —— 和 `scripts/bench.py` 同一個口徑。即時倍率 < 1 時會明確告訴你「生成追不上播放，串流在這個長度會斷」。

支援端點：`GET /runtime`（環境偵測）、`GET /candidates`（掃描可用的模型與音檔，SoVITS 的版本讀檔頭、參考音訊讀長度；會排除 SV / BERT / CNHubert 等支援模型）。

控制台已用 Chromium 實際驅動測過：三次合成、零 JS 錯誤、各區塊正確填入。

### 6.1 裝一組新的模型 / 參考音訊

模型與參考音訊都在 `server/config.yaml`，**不是寫死的**：

| 項目 | 設定 | 執行中切換 |
|---|---|---|
| GPT (s1) / SoVITS (s2) | `models.t2s` / `models.vits` | `POST /model` |
| 參考音訊 / 逐字稿 | `reference.audio` / `reference.text` | `POST /reference` |

任何欄位都可用環境變數覆寫（`GSV_MODELS__VITS=...`）。

**但換一組新的不只是改路徑。** 有五個條件必須滿足，而其中三個不滿足時失敗的樣子並不明顯：

1. 權重必須放在 `server/GPT_SoVITS/pretrained_models/` 底下 —— upstream 把這個相對路徑寫死在 `sv.py:6` 與 `text/chinese2.py:35`
2. `version` 必須和 SoVITS 權重的**檔頭**一致（`process_ckpt.py:100`），不一致時 upstream 會照檔案走並安靜自動修正
3. v3/v4 的角色權重只含 LoRA 增量，需要對應底模才能 merge（`TTS.py:560-581`）
4. 參考音訊長度必須在 3–10 秒之間（`TTS.py:816` 的硬性檢查）
5. 路徑打錯時 upstream 會**安靜退回預設路徑**（`TTS.py:342-353`），結果是載到底模而非你的角色模型

`scripts/add_voice.py` 把這些全部檢查完、檔案放好、並印出該填的設定：

```bash
python scripts/add_voice.py --name 新角色 \
    --sovits /path/新角色_e8.pth \
    --gpt    /path/新角色-e15.ckpt \
    --ref    /path/參考.wav \
    --ref-text "參考音訊的逐字稿"

    --dry-run          只檢查，不複製檔案
    --apply            檢查通過後寫進 config.yaml（原檔備份為 .yaml.bak）
    --server URL       直接套用到執行中的 server，不必重啟
```

它會自動從檔頭判定 `version`、核對 GPT 與 SoVITS 的符號集是否相符、把參考音訊轉成 server 的原生取樣率與單聲道（只做可證明不影響輸出的轉換），最後用 server 自己的 `preflight()` 再確認一次。

實測的拒絕情境：

| 情境 | 訊息 |
|---|---|
| v4 LoRA 權重但缺底模 | 列出缺哪些檔案、該放到哪個路徑 |
| 參考音訊 1.5 秒 | 「長度 1.50s 不在 3~10 秒之間（`TTS.py:816` 會報錯）」 |
| 逐字稿留空 | 「它是參考音訊的逐字稿，影響音色」 |
| GPT 與 SoVITS 符號集不符 | 指出 `phoneme_vocab_size` 與 `text_version` 對不上 |

#### 暖機：把換好的東西變成立刻可用

裝好之後，**啟動時與每次切換後都會自動暖機**，這是「成本付在切換時而不是之後每個請求上」的地方：

| 暖機步驟 | 之後每次請求省掉的 |
|---|---|
| `set_ref_audio` → `prompt_semantic`、`refer_spec` | CNHubert 前向、音檔 I/O、重採樣 |
| 第一次 `compute_embedding3` | SV 聲紋，CPU 實測 826 ms／段（見 §5.7） |
| 英文前端（丟棄式 `clean_text("warm up", "en")`） | `g2p_en` 的 `cmudict` 解析與 tagger 載入，**CPU 實測 6.39 s → 6 ms** |
| dummy 推論 | CUDA/MKL kernel、onnxruntime session、jieba 詞典的一次性成本 |

CPU 實測：`/model` 約 24 s、`/reference` 約 5 s。暖機**假設你給的東西是對的** —— 驗證是 `add_voice.py` 與 `preflight()` 的工作，不是暖機的。

#### 修英文發音

`text/english.py:322` 的 `qryword()` 對「不在 cmudict 裡、長度 <= 3」的詞**一律逐字母拼讀**：

```python
# oov 长度小于等于 3 直接读字母
if len(word) <= 3:
```

對縮寫這是對的，對剛好能當單字唸的詞就錯了。實測：

| 詞 | upstream 的音素 | 聽起來 | 對不對 |
|---|---|---|---|
| `wav` | `D AH1 B AH0 L Y UW0 EY1 V IY1` | "double-u ay vee" | **錯** |
| `gpu` | `JH IY1 P IY1 Y UW1` | "G-P-U" | 對 |
| `png` | `P IY1 EH1 N JH IY1` | "P-N-G" | 對 |
| `hz` | `HH ER1 T Z` | "hertz" | 對（在字典裡） |
| `vs` | `V ER1 S AH0 Z` | "versus" | 對（在字典裡） |

掃過 30 幾個這個專案會用到的短詞，**只有 `wav` 是真的錯**。所以不做大規模改字典，只修確定錯的。

**怎麼修**：加一行到 `server/pronunciation.rep`，重啟 server。

```
WAV W EY1 V
```

查某個詞現在怎麼唸：

```bash
python3 scripts/check_en.py wav gpu api      # 逐字母拼讀的會被標出來
python3 scripts/check_en.py "the wav file"   # 整句
python3 scripts/check_en.py --no-custom wav  # 看 upstream 原本的結果
```

**為什麼不直接改 `text/engdict-hot.rep`**：upstream 自己留了那個熱修字典（裡面已經有 CHATGPT、JSON、CONDA），`get_dict()` 會在讀完 cache 之後無條件套用。但它在 vendored 樹裡、被 `manifest.txt` 管著，`vendor_upstream.sh` 會重新複製，手改會被蓋掉。所以 `pronunciation.py` 用同樣的手法（覆寫 `_g2p.cmu` 的條目），資料放在 vendored 樹之外，upstream 一個字都不用改。

**為什麼載入時一定要驗證**：`replace_phs()`（english.py:111-121）對不在 `symbols` 裡的音素是**靜默丟掉**的，只 print 一行。所以一個打錯的音素不會報錯，只會讓那個字少一個音。`pronunciation.validate()` 在 `preflight()` 就擋掉，並指名是哪個字的哪個音素。

**這會改變輸出**，所以它算 §4 的「刻意偏離」—— `/health` 的 `custom_pronunciations` 會列出所有生效的詞，不會悄悄改。

#### 為什麼切換語言的地方會卡頓

先排除兩個常見猜測，兩個都實測過：

**不是空格。** 中英之間有沒有空格，產生的音素序列**完全相同**：

| 輸入 | 音素 |
|---|---|
| `這是wav測試` | `zh e4 sh ir4` + `W EY1 V` + `c e4 sh ir4` |
| `這是 wav 測試` | 同上，一模一樣 |
| `這是  wav  測試` | 同上，一模一樣 |

空格在中文側被 `text_normalize()` 清掉，在英文側被 `g2p()` 的 filter 丟掉（`english.py:367` 把 `" "` 濾掉）。片段之間 `phones = sum(phones_list, [])` 也沒有插入任何東西。

**不是插入的靜音。** `fragment_interval: 0.3` 確實會插 0.3 秒**真正的數位靜音**，但那是**片段之間**（`TTS.py:1551-1561`）。實測一段 13.66 s、八句話的文字，全零區塊只有 1 個、在最尾端。語言邊界上那個 60 ms 的低能量段峰值 RMS 是 0.0097 —— 不是零，是模型自己產生的猶豫。

**真正的原因在 `TextPreprocessor.py:212-222`：非中文片段的 BERT 特徵是全零。**

```python
if language == "zh":
    feature = self.get_bert_feature(norm_text, word2ph)
else:
    feature = torch.zeros((1024, len(phones)))   # ← 英文片段沒有任何條件訊號
```

而且每個片段的 BERT 是**各自獨立**算的（`TextPreprocessor.py:175-180` 的迴圈），再 `torch.cat` 接起來。實測 `這個 server 用 v2Pro 模型`：

| 片段 | 語言 | 音素數 | BERT 特徵 |
|---|---|---|---|
| `這個 ` | zh | 4 | 真實 BERT |
| `server ` | en | 4 | **全零** |
| `用 ` | zh | 2 | 真實 BERT |
| `v2Pro ` | en | 7 | **全零** |
| `模型` | zh | 4 | 真實 BERT |

所以每個語言邊界上，條件訊號都是從真實的 1024 維特徵直接掉到全零再跳回來，而且中文片段的 BERT 只看得到自己那一小段、看不到前後文。這是硬斷點，聽起來就是卡頓。

**為什麼不「修」它**：這個模型就是**用這個管線訓練出來的** —— 訓練時英文片段的 BERT 也是全零。推論時改成餵真實特徵會造成 train/inference 不一致，結果大概率更差而不是更好。這不是能靠改幾行就賺到的東西，所以不動。

**實際有效的做法**，依效果排序：

1. **換成雙語或英文的參考音訊**（效果最大）。洛神的參考音訊是純中文，模型從沒聽過這個說話者講英文。用 `scripts/add_voice.py` 裝一組雙語參考。
2. **減少不必要的語言切換**。`波形檔` 比 `wav` 好、`每秒三萬二千赫` 比 `32000 Hz` 好 —— 邊界越少，斷點越少。
3. **長文字如果聽到句間空隙太長**，調低 `fragment_interval`（那是純粹的靜音填充，不經過合成，調它不影響音質）。

**但發音不是英文品質的主因**：洛神的參考音訊是**純中文**（`我想，你们不必担忧我们的成长…`），SoVITS 也是中文語料訓練的。所以英文會帶中文口音，這是模型與參考音訊的限制，不是 G2P 能修的。想要好的英文需要英文或雙語的參考音訊 + 對應模型（用 `scripts/add_voice.py` 裝）。

### 6.2 換模型與換參考音訊

兩個操作都是**先驗證、通過才動**，失敗時舊狀態完全不受影響。

```bash
# 換模型（含版本）—— 會重新載入並暖機
curl -X POST localhost:8000/model -H 'Content-Type: application/json' \
  -d '{"version":"v2Pro","vits":"GPT_SoVITS/pretrained_models/洛神_e8_v2Pro.pth"}'

# 換參考音訊 / 參考文本 —— 只重算參考張量，不重載模型
curl -X POST localhost:8000/reference -H 'Content-Type: application/json' \
  -d '{"audio":"GPT_SoVITS/pretrained_models/reference/ref_pad.wav",
       "text":"對應這段音訊的逐字稿"}'
```

CPU 實測耗時：換模型約 24 s（要重載全部模型），換參考音訊約 5 s。

**為什麼換模型必須重建整個引擎**：`prompt_cache` 的 `prompt_semantic` 是用當前 SoVITS 模型的 `extract_latent` 算出來的（`TTS.py:830`）。換了 SoVITS 卻沿用舊的 `prompt_semantic` 就是髒資料。重建引擎會重跑暖機，自動處理掉。

**換模型時先釋放舊的再載入新的**，而不是先建新的再換掉舊的 —— 後者會讓兩份模型同時在 GPU 上、VRAM 短暫加倍，V100 上可能直接 OOM。代價是這段時間請求收到 503，這比 OOM 好。

#### preflight：為什麼不能把路徑檢查交給 upstream

`engine.preflight()` 在載入任何模型之前先檢查完。有兩個必須自己做的理由：

**1. 路徑不存在時 upstream 會安靜退回預設路徑。** `TTS.py:342-353`：

```python
if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
    self.vits_weights_path = self.default_configs[version]["vits_weights_path"]
    print(f"fall back to default vits_weights_path: {self.vits_weights_path}")
```

只有一行 print，沒有錯誤。結果是你要求模型 X、實際載到 v2Pro 底模，而且不知情。對「換模型」這個操作來說，安靜載錯模型比直接失敗糟得多。

**2. SoVITS 的版本是從檔頭判定的，不是從設定。** `process_ckpt.py:100` 的 `get_sovits_version_from_path_fast()` 讀檔案前兩個 byte。所以設定寫 `version: v4`、檔案是 v2Pro 時，upstream 會照檔案走並自動修正 —— 你要的東西沒拿到，卻沒有任何提示。

preflight 把這兩種情況都變成 **HTTP 400**，並附上可行動的訊息：

```
設定無法使用：
  - version 設為 v4，但 .../洛神_e8_v2Pro.pth 的檔頭判定為 v2Pro（is_lora=False）。
    版本是從檔頭讀的，不是從設定 —— 請改成 version=v2Pro，或換成真正的 v4 權重。
```

它也會檢查 v3/v4 的 LoRA 權重是否有對應底模可 merge（`TTS.py:560-581` 需要 `s2Gv4.pth`）。

#### 實測過的失敗情境

| 情境 | 結果 |
|---|---|
| `/model` 指到不存在的檔案 | 400，舊引擎完全不受影響 |
| `/model` 的 version 與檔頭不符 | 400，附上檔頭判定結果 |
| `/model` 通過 preflight 但載入失敗 | 500，並嘗試還原為先前的模型 |
| `/reference` 指到不存在的音檔 | 400，參考設定不被改動 |
| `/reference` 參考文本為空 | 400 |

修正前一版的問題：

- **不接受客戶端指定輸出路徑**（原本是任意檔案寫入）
- 錯誤回**正確的 HTTP status code**，不是 200 + `str(e)`
- 設定走 YAML + env，**零寫死路徑**
- 請求指定 `profile=low` 回 **403**，除非 server 端開放（見 §4）

`/metrics` 是新增的 —— 既然目標是低延遲，就要能量測。

---

## 7. vendoring：用加法，不用刪除法

### 方法

**唯一的真實清單是 [`scripts/manifest.txt`](scripts/manifest.txt)。** `scripts/vendor_upstream.sh` 只複製那份清單列出的路徑，清單裡沒有的就不存在。

初版用的是「整份複製再刪除」，已改掉。理由：**刪除法留下的是未知的多餘檔案，加法留下的是明確清單。** 加法還有一個更實際的好處 —— 清單中有任何路徑在來源不存在時腳本直接中止，而刪除法對「不小心留下的東西」沒有任何訊號。

清單分組，每組都寫明為何需要：

| 組 | 內容 | 依據 |
|---|---|---|
| `[reachable]` | 65 個 .py | `scripts/import_graph.py` 的可達性分析，可用 `--manifest` 重新產生 |
| `[pkg]` | 套件標記 `__init__.py` | 靜態分析不會把它們列為可達，缺了 import 會失敗 |
| `[dynamic]` | `eres2net/`、`AP_BWE_main/` | 執行期以 `sys.path.append` 載入，靜態分析追不到 |
| `[data-en]` / `[data-zh]` | 詞典檔 | 以 `open()` 讀取，import 分析看不到 —— 另外掃字串常數找出來的 |
| `[i18n]` | `en_US` + `zh_CN` + `zh_TW` | `I18nAuto` 固定回退 `en_US` 後**無條件開檔**，缺了會 FileNotFoundError |
| `[configs]` | `s2*.json` | 推論不讀，但 §2.2 引用 `s2v2Pro.json` 作為架構依據，留著讓引用可查核 |
| `[gitignore]` | upstream 自己的排除規則 | 保留它們，權重不會被誤 commit |
| `[license]` | `LICENSE` ×11 | 本專案重新散布 upstream 的程式碼 |

結果：**110 個檔案、81 個 .py**。體積仍是 18 MB，因為幾乎全部是中英文前端的詞典檔（`cmudict*.rep` 7 MB、`engdict_cache.pickle` 5.7 MB、`polyphonic*` 2.9 MB），那些是必要的。vendored 的程式碼本身只有 128 KB 加上 `GPT_SoVITS/` 下的 .py。

### 交叉檢查：加法樹 vs 刪除法樹

換方法時沒有直接取代，而是兩種方法各建一棵樹逐檔對照，**任何差異都必須解釋**。這抓到一項加法漏掉的東西：

> `BigVGAN/alias_free_activation/torch/__init__.py` 不是空的 —— 它有 `from .filter / .resample / .act import *`。`bigvgan.py:19` 雖然是直接 import 子模組（PEP 420 namespace package 也能解析），但這是實際使用的套件的標記檔，不帶沒有道理。

值得記下的是 **`--verify` 抓不到這一項** —— `import_graph.py` 的 resolver 把純目錄當 namespace package 接受了。是兩棵樹的對照抓到的。靜態驗證有盲區，交叉檢查才補得上。

對照剩下的 24 項差異全部確認為「刪除法多留的」：4 個 README、`configs/` 下 7 個訓練 yaml 與 `tts_infer.yaml`、以及 10 個有 `en_US` fallback 保護的語言檔。

### 不在清單中的（以及為什麼）

| 類別 | 項目 |
|---|---|
| 訓練 | `webui.py`（82 K）、`s1_train.py`、`s2_train*.py`（3 支）、`prepare_datasets/`、`AR/data/`、`module/{data_utils,losses,ddp_utils,distrib}.py` |
| 匯出 / 其他介面 | `onnx_export.py`、`export_torch_script*.py`（82 K）、`inference_gui.py`、`inference_webui*.py`、`stream_v2pro.py`、`module/{attentions_onnx,models_onnx}.py` |
| 資料集工具 | `tools/asr/`、`tools/uvr5/`、`tools/denoise-model/` |
| 非中英文 | `text/japanese.py`、`korean.py`、`cantonese.py`、`ja_userdic/`（17 MB） |
| v3 專用 | `BigVGAN/`（364 K）、`tools/AP_BWE_main/` |
| 雜項 | Colab / Kaggle notebooks、`install.sh` / `install.ps1`、`docs/` 非中英文版本 |

### 權重

**需要**：BERT（`chinese-roberta-wwm-ext-large`，僅 `pytorch_model.bin`）、CNHubert（`chinese-hubert-base`，僅 `pytorch_model.bin`）、G2PW onnx、`lid.176.bin`、SV ckpt、洛神 GPT + 洛神 v2Pro

**v4 另需**：`s2Gv4.pth`（769 MB）+ `vocoder.pth`（58 MB）

**前一版的多餘權重**（合計 4.69 GB / 8.25 GB = 57%）：

| 檔案 | 大小 | 多餘原因 |
|---|---|---|
| `bert/tf_model.h5` | 1303 MB | TensorFlow 版，PyTorch 推論用不到 |
| `bert/flax_model.msgpack` | 1302 MB | Flax 版，同上 |
| `hubert/chinese-hubert-base-fairseq-ckpt.pt` | 1136 MB | fairseq 原始 ckpt，已有 `pytorch_model.bin` |
| `GPT_SoVITS/G2PWModel/g2pW.onnx` | 635 MB | 與 `text/G2PWModel/` 下那份重複 |
| `pretrained_models/g2pw-chinese/pytorch_model.bin` | 318 MB | g2pW 的 PyTorch 版，已有 onnx |

---

## 8. 依賴瘦身

### 移除

`tensorboard`、`gradio`、`funasr`、`faster-whisper`、`ctranslate2`、`modelscope`、`av`、`pyopenjtalk`、`g2pk2`、`ko_pron`、`ToJyutping`、`python_mecab_ko`、`sentencepiece`、`chardet`、`psutil`、`rotary_embedding_torch`

**不能移除**（見 §4 的「原計劃中被否決的項目」）：`pytorch-lightning`、`torchmetrics`、`matplotlib`、`x_transformers`、`huggingface_hub`、`requests`。

清單由 `scripts/import_graph.py` 掃出 `server/` 實際 import 的外部套件而來，不是從 upstream 的 `requirements.txt` 抄改。

### 保留

`torch`、`torchaudio`、`transformers`、`numpy<2`、`scipy`、`librosa`、`numba`、`ffmpeg-python`、`jieba_fast` / `jieba`、`cn2an`、`pypinyin`、`g2p_en`、`wordsegment`、`PyYAML`、`chardet`、`onnxruntime`（G2PW）、`fast_langdetect`、`split-lang`、`opencc`、`fastapi`、`pydantic`

**v4 另需**：`peft`、`x_transformers`、`rotary_embedding_torch`

### CPU 實測發現的三個依賴問題

這三個 upstream 都沒有處理，今天全新安裝一定會踩到：

**1. `torchaudio` 必須有上界** —— torchaudio 自 2.8/2.9 起移除自己的 I/O backend，`torchaudio.load()` 改為委派給 `torchcodec`。而 `TTS.py:773` 的 `_get_ref_spec()` 正是用它讀參考音訊。

實測：2.7.1 正常；**2.11.0 會 ImportError 要求 torchcodec**，而 torchcodec 的預編譯 wheel 連結了 CUDA（需要 `libnvrtc.so.13`），在純 CPU 機器上載不起來。upstream 的 `requirements.txt` 完全沒釘 torchaudio。

→ 本專案釘 `torchaudio>=2.1,<2.9`，實測通過的組合是 torch 2.7.1 + torchaudio 2.7.1。

**2. Python 必須是 3.10 ~ 3.12** —— upstream 釘 `numpy<2.0`，而 numpy 1.26.x 沒有 Python 3.13 的 wheel（numpy 要到 2.1 才支援 3.13）。3.13 無法同時滿足這個釘選。

另外 `jieba_fast` 只有 2019 年的 source tarball、**完全沒有 wheel**，必須就地編譯 C 擴充（已實測在 3.11 可編譯通過，版本 0.39）。不可改用 `jieba` 代替：`text/chinese2.py:20` 硬性要求 `jieba_fast`，而兩者分詞若有任何差異會改變 `word2ph` → 改變 BERT 特徵對映 → **改變音訊**。

**3. 英文需要 nltk 語料** —— `text/english.py:5` 用 `g2p_en`，它需要 nltk 的 `averaged_perceptron_tagger` 與 `cmudict`，而那是**第一次使用時才下載**的。中文不受影響（走 `chinese2` 完全不碰 `g2p_en`，已實測：把語料移走後純中文的 `clean_text` 照樣過）。

這件事有三個後果，一個比一個不明顯：

1. **離線機器上英文合成會失敗。**
2. **可攜版會壞。** `nltk.download()` 預設寫 `~/nltk_data`，那在可攜資料夾外面 —— 整包搬到目標機器後英文就沒了。
   → 改成下載到專案內的 `nltk_data/`，由 `bootstrap.py` 的 `_use_local_nltk_data()` 在 import nltk 之前設好 `NLTK_DATA`；`make_bundle.py` 也會把它複製進可攜版。
3. **失敗的樣子完全看不出原因。** `/tts/stream` 是 chunked 串流，HTTP 200 早就送出去了，所以錯誤發生在串流「中途」——
   客戶端只拿到 `http.client.IncompleteRead(0 bytes read)`。這個 bug 就是這樣浮現的：純中文都好，只有含英文的句子會斷。
   → `engine.english_frontend_ready()` 在啟動時就查一次並警告，`/health` 的 `english_ready` 也看得到。

→ `scripts/fetch_models.sh`（與 `scripts/start.py --download-only`）會一併抓好這兩份語料（約 20 MB）。

### 一併修正

- 明確列出 `torch`（前一版只列 `torchaudio`，靠它間接帶入）
- 版本釘選只有一處，不再出現 README 與 `requirements.txt` 衝突
- `__pycache__` 加進 `.gitignore`，不再 commit

---

## 9. 取樣參數

三組候選值，來源不同：

| 來源 | top_k | top_p | temperature |
|---|---|---|---|
| **upstream 執行時預設**（`TTS.py:1011-1013`、`:1042-1044`；`api_v2.py:30-32`、`:161-163`） | **5** | **1** | **1** |
| 洛神 ckpt 的 `config.inference.top_k` | 15 | — | — |
| 前一版寫死的值 | 20 | 0.6 | 0.6 |

**決定：以 upstream 執行時預設（5 / 1 / 1）為基準**，三個參數全部開放為執行時可調。

理由：洛神 ckpt 裡的 `inference.top_k = 15` 是**訓練 config** 的欄位，不是推論預設；前一版的 20 / 0.6 / 0.6 來源不明。upstream 預設是這條 pipeline 實際被驗證過的設定。

三組都記錄於此以便 A/B。

**對延遲的影響**：`top_k` 不影響每 token 時間（取樣成本相對 forward 可忽略）。但取樣參數會影響生成長度與是否提早觸發 EOS，間接影響總時間。

---

## 10. Repo 結構

```
start.sh / start.bat      一鍵啟動（雙擊即可）
requirements.txt          由 import 圖掃出，非從 upstream 抄改
Colab.ipynb               產生物 —— 要改請改 scripts/make_colab.py
nltk_data/                英文前端的 nltk 語料（下載而來，不進 git）

server/
  config.yaml             設定；可用 GSV_* 環境變數覆寫
  pronunciation.rep       自訂英文發音（本專案的檔案，不是 vendored）
  GPT_SoVITS/             vendored upstream —— 原封不動，不改
  tools/                  vendored upstream —— 原封不動，不改
  gsv_server/             本專案自己的程式碼
    bootstrap.py          建立 upstream 預期的 sys.path 與工作目錄
    settings.py           設定載入與驗證
    bert_trim.py          BERT 等價剪裁
    pronunciation.py      自訂英文發音的載入、驗證與套用
    engine.py             暖機、快取、量測、GPU 操作序列化
    api.py                HTTP + WebSocket
    console.html          網頁控制台（單一檔、零外部依賴）
    wavio.py              最小 WAV 封裝
    __main__.py           唯一的啟動進入點

scripts/
  start.py                一鍵啟動：建環境 → 下載權重 → 啟動 server → 開控制台
  make_bundle.py          打包成解壓縮即可用的可攜版（自帶 Python 與權重）
  add_voice.py            裝一組新的模型 + 參考音訊（檢查、放檔、產生設定）
  prepare.py              離線剪裁 BERT 等啟動時間優化（選用）
  manifest.txt            vendored 檔案的唯一真實清單（每組都寫明理由）
  vendor_upstream.sh      依 manifest 複製；清單中的路徑不存在就中止
  import_graph.py         可達性分析 / 斷鏈驗證 / --manifest 重新產生清單
  fetch_models.sh         下載權重（只抓 PyTorch 格式）與英文 nltk 語料
  make_colab.py           產生 Colab.ipynb（寫出前會先編譯每個 code cell）
  verify_equivalence.py   實測等價優化是否真的等價（需 GPU）
  bench.py                量 TTFA（只用標準庫）
  check_en.py             查某個英文詞實際會被唸成什麼音素
```

`Colab.ipynb` 是**產生物**。notebook 的 JSON 很難讀也很難 review，而且手改之後下次重跑產生器就被蓋掉，所以唯一的真實來源是 `scripts/make_colab.py`。

`server/GPT_SoVITS/` 與 `server/tools/` 底下的檔案**一律不修改**。需要改變行為時，一律在 `gsv_server/` 這一層處理（例如工作目錄、BERT 替換、快取包裝），而不是去動 upstream 的程式碼。這是為了讓 upstream 更新時能直接重跑 `vendor_upstream.sh` 而不需要重新 rebase 任何修改。

---

## 11. 驗證策略

### 11.1 已完成：CPU 端到端實測

本專案程式碼在一台**無 GPU**（4 核 Xeon 2.8GHz / 15 GB RAM）的環境中撰寫。原本預期只能做靜態保證，但該環境足以實際跑完整條推論鏈，所以已經跑了。

環境：Python 3.11 / torch 2.7.1+cpu / torchaudio 2.7.1+cpu / numpy 1.26.4 / transformers 4.50.0，v2Pro + 洛神，`device=cpu`、`is_half=false`（`TTS_Config:328` 會自動把 CPU 的 `is_half` 降為 false）。

**啟動**

| 階段 | 時間 |
|---|---|
| 模型載入 | 21.4 s |
| `set_ref_audio`（參考張量預算） | 2.3 s |
| 英文前端暖機 | 4.0 s |
| dummy 暖機 | 11.9 s |
| **合計** | **41.0 s** |

**合成**（`profile=quality`、`cut1`、取樣參數 5/1/1）

| 文字 | TTFA | 總時間 | 音訊長度 | 即時倍率 |
|---|---|---|---|---|
| 8 字 | 2565 ms | 2.87 s | 1.74 s | **0.607×** |
| 26 字 | 6068 ms | 6.44 s | 5.14 s | **0.799×** |
| 44 字（中英混合） | 6102 ms | 6.58 s | 4.94 s | **0.751×** |

中英混合那列是有意義的：它的 TTFA **低於**同批的 26 字純中文句（6595 ms），說明英文前端那 6.39 s 的一次性成本已經在暖機時付掉，沒有落在第一個含英文的請求上。沒有這個暖機步驟時，第一個含拉丁字母的句子 TTFA 會多出 6 秒以上。

AR 解碼速率約 45 token/s（4 核）。v2Pro 的 semantic frame rate 是 25 Hz，所以每秒音訊需要 25 個 token → 約 0.55 s 的計算。

**結論：CPU 上可以跑，但即時倍率約 0.6–0.8×，達不到實時。** 要實時播放需要 GPU。

**暖機確實有效**：dummy 暖機本身花了 11.9 s，但緊接著的第一個真實請求（8 字）只用 2.87 s —— 一次性成本（CUDA/MKL kernel、onnxruntime session、jieba 詞典）確實被提前付掉了。

**輸出音訊的健全性檢查**

| | 短句 | 長句 |
|---|---|---|
| 零交越率 | 0.064 | 0.081 |
| RMS / 峰值 | 3156 / 29486 | 3017 / 28289 |
| 語速 | 0.22 s/字 | 0.20 s/字 |

零交越率落在語音的典型範圍（白噪音約 0.5），峰值未削波，語速一致 —— 是真的語音輸出。

### 11.2 等價優化的實測結果

啟動自檢（§4）在 CPU 實測中通過：

```
BERT 已剪裁：24 → 22 層，省下 26,265,224 個參數（8.07%）；5 段探針文字的輸出逐位元相同
所有會影響輸出的設定都與 upstream 預設一致
```

`/health` 的 `quality` 區塊回報 `clean: true`、`bert_trim_verified: true`。

26,265,224 這個實測值也確認了 §5.5 的修正是對的 —— 詞表投影權重與 input embeddings 綁定，所以移除 MLM 頭省下的遠少於初版估計的 47 M。

### 11.3 仍必須在目標機器上做的

- **音質判斷**。CPU 跑出的音訊可以確認「不是噪音、不削波、語速正常」，但好不好聽要用耳朵，而且 GPU 上的 fp16 與 CPU 的 fp32 數值不同。
- **延遲實測**。CPU 的 0.6–0.8× 不能外推到 GPU。
- **v2Pro vs v4 的音質比較**（§3.2 仍只是官方說法）。

### 11.4 等價性驗證（對應 §4 類別一）

每一項「可證明等價」的優化都要有對應驗證：

| 優化 | 驗證方式 |
|---|---|
| 暖機 `prompt_cache` | 與請求中即時計算的張量逐元素比對，要求完全相同 |
| BERT 截 22 層 | 截層前後的 `hidden_states[-3]` 輸出逐元素比對 |
| 移除 MLM 頭 | 同上 |
| 英文零 BERT | 與呼叫 BERT 後丟棄的結果比對 |
| 文字前端快取 | 快取命中與未命中的 `(phones, bert_features, norm_text)` 比對 |

### 11.5 端到端比對

以同一組（參考音訊、參考文本、目標文本、seed、取樣參數）在改造前後產出音訊，比對波形。需 GPU 機器。

### 11.6 延遲基準測試

`tools/` 提供基準腳本，量測並記錄：

- **TTFA**（主要指標）
- 每 token 的 AR 解碼時間
- 各階段分解時間（文字前端 / AR / SoVITS 解碼）
- 冷啟動 vs 暖機後的差異（驗證 §5.4 有效）
- 快取命中 vs 未命中（驗證 §5.3、§5.7 有效）

---

## 12. 未決事項

| # | 項目 | 狀態 |
|---|---|---|
| 1 | **目標機器規格** | 已知前一版跑在 V100 × 2。需確認本專案部署的實際機器 |
| 2 | **「超低延遲」的具體數字** | 目前無目標值。TTFA 要壓到多少才算達標？這決定要不要動用 §4 類別二、類別三的手段 |
| 3 | **v2Pro vs v4 的實際音質** | 需在目標機器上實測比較。§3.2 僅為官方說法 |
| 4 | **v2ProPlus** | 若能取得洛神的 v2ProPlus 權重（檔頭 `06`），應一併實測 |
| 5 | **參考音訊是否固定** | §5.3 的暖機效益取決於參考音訊的變動頻率。若每次請求都換參考音訊，該優化失效 |

第 2 項仍是最關鍵的 —— 「超低延遲」沒有目標數字的話，無法判斷 §4 類別一的等價優化是否已足夠，或是否需要動用會影響音質的手段（`profile=low`、更細的切句）。

**下一步**：在目標機器上依序執行 `scripts/verify_equivalence.py`（確認等價性成立）與 `scripts/bench.py`（取得 TTFA 實測值），再據以決定要不要動用類別二的手段。在有實測數字之前，不預先犧牲任何音質。

---

## 13. 引用來源

### 本專案

| 項目 | 來源 |
|---|---|
| 前一版（診斷對象） | `Sihrrieidhr/GPT-SoVITS-Lite`（HuggingFace） |
| 洛神 GPT `洛神-e15.ckpt` | `Sihrrieidhr/GPT-SoVITS-Lite` |
| 洛神 SoVITS v2Pro `洛神_e8.pth`【主力】 | `Sihrrieidhr/haah` |
| 洛神 SoVITS v4 `洛神_e8.pth`【備用】 | `Sihrrieidhr/GPT-SoVITS-Lite` |

### upstream

- GPT-SoVITS：https://github.com/RVC-Boss/GPT-SoVITS
- 版本特性原文：官方 README「V2Pro Release Notes」、「V4 Release Notes」
- 本文件所有行號引用皆對應 upstream `main` 分支（基底 commit 將於程式碼落地時釘選）

---

## 附錄：改這個專案前必讀

四個讀原始碼才看得到、憑經驗寫一定會踩的地方。都已在程式碼註解中標明，
這裡集中列出。

### 1. `TTS_Config` 只讀 `"custom"` 鍵

`TTS.py:304-306`：

```python
configs_ = deepcopy(self.default_configs)
configs_.update(configs)
self.configs = configs_.get("custom", configs_["v2"])
```

把設定平鋪傳入（`{"device":..., "version":"v2Pro", ...}`）會被 `update` 放到最外層，然後因為沒有 `"custom"` 鍵而**整組被忽略、退回 v2 的預設值**。不會報錯，只會安靜地載入錯誤的模型。正確形式是 `{"custom": {...}}`。

### 2. `text_lang` 不可用 `"auto"`

`TextPreprocessor.py:148-150` 的 `auto` 分支會原樣採用偵測到的語言（`langlist.append(tmp["lang"])`），一旦判為日文或韓文就去 import `text/japanese.py` 或 `korean.py` —— 這兩個模組已在 vendoring 時移除（本專案只支援中英文），會直接 ImportError。

正確值是 `"zh"`。upstream 的註解是「按中英混合识别」，走 `TextPreprocessor.py:157-169` 的 else 分支，該分支對非英文段落一律以傳入的 `language` 覆蓋（註解：「因无法区别中日韩文汉字,以用户输入为准」），因此只會載入 `chinese2` 與 `english` 兩個模組。

設定驗證會攔下 `auto`（`settings.py` 的 `ALLOWED_LANGS`）。

### 3. BERT 截層的負索引陷阱

`TextPreprocessor.py:197` 取 `hidden_states[-3]`，是**負索引**。單純把 encoder 截短，`hidden_states` 的長度跟著變，`[-3]` 就會指到更前面的層，輸出不再相同。截層後必須把 `hidden_states` 補回原長度。見 `server/gsv_server/bert_trim.py`。

### 4. 串流模式有品質代價，而且會換掉 AR 實作

upstream 在 `TTS.py:1026-1033` 自己標註 `streaming_mode` 是 "Medium quality"、`fixed_length_chunk` 是 "lower quality"、`return_fragment` 是 "Best Quality"。並且模式選擇會在 `infer_panel_batch_infer` / `infer_panel_naive` / `infer_panel_naive_batched` 之間切換（`:1066-1083`），是不同的程式碼路徑。詳見 §5.2。

### 另外：工作目錄是有意義的

`TTS.py:13` 在模組層級執行 `now_dir = os.getcwd()`，`sv.py:5-6` 用 `os.getcwd()` 組出 eres2net 與 SV 權重路徑，`text/chinese2.py:35` 寫死相對路徑 `"GPT_SoVITS/text/G2PWModel"`。所以 `bootstrap.setup()` 必須在 import 任何 upstream 模組**之前**完成。修這些相對路徑的替代做法是改 upstream 的檔案，本專案選擇不改。
