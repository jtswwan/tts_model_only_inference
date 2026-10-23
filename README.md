# tts_model_only_inference

GPT-SoVITS 的**純推論** server。單一目標：**超低延遲**。

主力走 **v2Pro**（單次前向，無 CFM），v4 作為品質比較與離線用途保留。模型可換。邊緣設備以網路連線使用，**不在設備上跑推論**。

> **狀態：計劃階段，尚無程式碼。**
> 本文件記錄完整設計決策與依據。所有結論附原始碼行號或官方文件原文，不憑印象。
> 程式碼落地後補上「使用方式」章節。

---

## 目錄

- [0. 範圍](#0-範圍)
- [1. 為什麼重做](#1-為什麼重做)
- [2. 洛神模型的事實](#2-洛神模型的事實)
- [3. 為什麼主力是 v2Pro](#3-為什麼主力是-v2pro)
- [4. 設計原則：「不動音質」的操作型定義](#4-設計原則不動音質的操作型定義)
- [5. 低延遲設計](#5-低延遲設計)
- [6. 對外介面](#6-對外介面)
- [7. 保留 / 刪除清單](#7-保留--刪除清單)
- [8. 依賴瘦身](#8-依賴瘦身)
- [9. 取樣參數](#9-取樣參數)
- [10. Repo 結構](#10-repo-結構)
- [11. 驗證策略](#11-驗證策略)
- [12. 未決事項](#12-未決事項)
- [13. 引用來源](#13-引用來源)

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
| semantic → 音訊 | **單次前向**，`use_vocoder = False`（`TTS.py:545`） | CFM flow-matching **8 步 ODE**，再過獨立 HiFiGAN vocoder |
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
| 拆除 `pytorch-lightning` 包裝 | 推論只用到 `t2s_model.Text2SemanticDecoder`。`LightningModule` 只是訓練期外殼 |
| 刪除訓練 / 匯出 / 其他介面程式碼（§7） | 推論路徑不經過 |

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

### 5.2 串流：用 upstream 的實作，不自己寫

`TTS.py:1068`：

```python
elif not parallel_infer and streaming_mode and not self.configs.use_vocoder:
```

`TTS.py:1431`：

```python
audio_chunk, latent, latent_mask = self.vits_model.decode_streaming(..., sv_emb=sv_emb, ...)
```

條件 `not self.configs.use_vocoder` 表示**串流只在 v2Pro / v2 這條（無獨立 vocoder）路上可用**。這是 §3.1 說「串流只有 v2Pro 有」的原始碼依據。

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

兩項合計約 47 M 參數。對延遲的貢獻在階段 1。

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

### 5.7 文字前端快取

同一文本的音素序列與 BERT 特徵是確定性的 → 按文本內容快取 `(phones, bert_features, norm_text)`。對重複句型（提示語、固定回應）效果顯著，可讓階段 1 歸零。

### 5.8 切句粒度：延遲與韻律的取捨

TTFA 與第一個 chunk 的 token 數成正比 → 切得細則延遲低。

**但切句會改變送進 GPT 的文本，進而改變韻律。** 這屬於 §4 類別二。

**決定：做成明確參數，預設跟隨 upstream 的切句策略**（`text_segmentation_method.py`）。文件中記錄取捨，由實測決定是否調整。不偷偷為了延遲犧牲韻律。

### 5.9 傳輸層

- **直接送 raw PCM**，不在請求路徑上做 WAV / 其他編碼
- chunk 盡量小（與 §5.8 的切句粒度是不同層次的事：這裡只影響網路傳輸，不影響合成）
- WebSocket 或 HTTP chunked，由客戶端選擇

### 5.10 併發

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
| `GET /health` | 狀態、當前模型與版本、裝置、暖機是否完成 |
| `GET /metrics` | TTFA、每 token 時間、佇列長度 |

修正前一版的問題：

- **不接受客戶端指定輸出路徑**（原本是任意檔案寫入）
- 錯誤回**正確的 HTTP status code**，不是 200 + `str(e)`
- 設定走 YAML + env，**零寫死路徑**

`/metrics` 是新增的 —— 既然目標是低延遲，就要能量測。

---

## 7. 保留 / 刪除清單

### 保留（原封不動，不改寫）

```
TTS_infer_pack/      TTS.py, TextPreprocessor.py, text_segmentation_method.py
AR/                  models/, modules/, utils/            （不含 data/）
module/              models.py, modules.py, attentions.py, commons.py,
                     mel_processing.py, mrte_model.py, quantize.py,
                     core_vq.py, transforms.py
feature_extractor/   cnhubert.py                （暖機時用，不在請求路徑上）
text/                中文 + 英文前端（含 G2PW、LangSegmenter）
process_ckpt.py, utils.py, configs/
tools/i18n/, tools/my_utils.py
v2Pro：              sv.py + eres2net/          （暖機時用）
v4：                 f5_tts/model/
```

### 刪除

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

`tensorboard`、`pytorch-lightning`（§4）、`gradio`、`funasr`、`faster-whisper`、`ctranslate2`、`modelscope`、`torchmetrics`、`av`、`pyopenjtalk`、`g2pk2`、`ko_pron`、`ToJyutping`、`python_mecab_ko`

### 保留

`torch`、`torchaudio`、`transformers`、`numpy<2`、`scipy`、`librosa`、`numba`、`ffmpeg-python`、`jieba_fast` / `jieba`、`cn2an`、`pypinyin`、`g2p_en`、`wordsegment`、`PyYAML`、`chardet`、`onnxruntime`（G2PW）、`fast_langdetect`、`split-lang`、`opencc`、`fastapi`、`pydantic`

**v4 另需**：`peft`、`x_transformers`、`rotary_embedding_torch`

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
server/    推論 server。多版本 + 可換模型 + 串流
tools/     暖機 / 基準測試 / 等價性驗證腳本
docs/      設計決策記錄
```

---

## 11. 驗證策略

### 11.1 本開發環境的限制（必須寫明）

本專案程式碼在 Claude Code 的 cloud container 中撰寫，該環境：

- **無 GPU**（無 `nvidia-smi`、無 `/dev/nvidia*`）
- 4 核 CPU / 15 GB RAM
- 未安裝 torch

**因此無法在此環境產出音檔或量測延遲。** 能做到的是靜態保證：

1. 語法與 import 完整性檢查
2. 與 upstream 逐行核對張量流程（形狀、取樣率、dtype、裝置）
3. checkpoint metadata 解析驗證（已完成，見 §2）

**音質與延遲必須在目標機器上實測。**

### 11.2 等價性驗證（對應 §4 類別一）

每一項「可證明等價」的優化都要有對應驗證：

| 優化 | 驗證方式 |
|---|---|
| 暖機 `prompt_cache` | 與請求中即時計算的張量逐元素比對，要求完全相同 |
| BERT 截 22 層 | 截層前後的 `hidden_states[-3]` 輸出逐元素比對 |
| 移除 MLM 頭 | 同上 |
| 英文零 BERT | 與呼叫 BERT 後丟棄的結果比對 |
| 文字前端快取 | 快取命中與未命中的 `(phones, bert_features, norm_text)` 比對 |
| 拆 pytorch-lightning | 同一輸入下 `infer_panel` 輸出比對（固定 seed） |

### 11.3 端到端比對

以同一組（參考音訊、參考文本、目標文本、seed、取樣參數）在改造前後產出音訊，比對波形。需 GPU 機器。

### 11.4 延遲基準測試

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

第 2 項是現在最需要你回答的 —— 「超低延遲」沒有數字的話，無法判斷 §4 類別一的等價優化是否已足夠，或是否需要動用會影響音質的手段。

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
