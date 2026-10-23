# tts_model_only_inference

GPT-SoVITS 的**純推論**改裝版。兩個產出：一個能離線跑在邊緣設備上的極簡版（v2Pro + 固定洛神），一個可換模型的推論 server（v4 / v2Pro / v2ProPlus）。

> **狀態：計劃階段，尚無程式碼。**
> 這份文件記錄完整的設計決策與依據。所有結論都附上原始碼行號或官方文件原文，不憑印象。
> 程式碼落地後，本文件的「使用方式」章節才會補上。

---

## 目錄

- [0. 兩個產出](#0-兩個產出)
- [1. 為什麼重做](#1-為什麼重做)
- [2. 洛神模型的事實](#2-洛神模型的事實)
- [3. 為什麼邊緣版用 v2Pro 而不是 v4](#3-為什麼邊緣版用-v2pro-而不是-v4)
- [4. 設計原則：「不動音質」的操作型定義](#4-設計原則不動音質的操作型定義)
- [5. 產出 A：邊緣版](#5-產出-a邊緣版)
- [6. 產出 B：推論 server](#6-產出-b推論-server)
- [7. 保留 / 刪除清單](#7-保留--刪除清單)
- [8. 依賴瘦身](#8-依賴瘦身)
- [9. 取樣參數](#9-取樣參數)
- [10. Repo 結構](#10-repo-結構)
- [11. 驗證策略](#11-驗證策略)
- [12. 未決事項](#12-未決事項)
- [13. 引用來源](#13-引用來源)

---

## 0. 兩個產出

| | **A. 邊緣版** | **B. 推論 server** |
|---|---|---|
| 位置 | `edge/` | `server/` |
| 版本 | v2Pro 固定 | v4 / v2Pro / v2ProPlus 可切 |
| 模型 | 洛神固定，寫死 | 可換，熱切換 |
| 目標 | 離線、最小、實時 | 多模型、串流、供邊緣設備連線 |
| 設定 | 無 config 解析，常數寫死 | YAML + API |
| 版本分支 | 無 | 有（沿用 upstream 既有分支） |

兩邊**不共用程式碼**。共用就會違反「每一行都要用到」的要求：邊緣版不該為了遷就 server 的多版本抽象而帶進用不到的分支。

---

## 1. 為什麼重做

前一版（`Sihrrieidhr/GPT-SoVITS-Lite`）不是「風格不好」，是有會真的出錯的地方。摘錄確認過的問題：

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

對照 upstream `GPT_SoVITS/inference_webui.py:517` 的 `get_spepc()`：它重採樣到 `hps.data.sampling_rate` 才算頻譜，並且有 `if maxx > 1: audio /= min(2, maxx)` 的音量保護（後半也漏了）。

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
| 69 個 `__pycache__/*.pyc` 被 commit | — |
| `requirements.txt` 是 upstream 原封不動抄的（含 `tensorboard`、`pytorch-lightning`、`gradio`、`funasr`、`modelscope`、`ctranslate2`） | — |
| README 描述的 `inference_webui.py (711行)` 不存在，只剩它的 `.pyc` | — |
| README「方法二」的版本釘選與 `requirements.txt` 衝突（`transformers==4.37.2` vs `>=4.43`；`librosa==0.10.1` vs `==0.10.2`） | — |
| 權重 4.69 GB / 8.25 GB（57%）是多餘的 | 見 §5.6 |

### 1.6 結論：基底換掉

**不沿用那支手寫腳本，改用 upstream 的 `GPT_SoVITS/TTS_infer_pack/TTS.py`。**

理由：`TTS.py` 是 upstream 維護的推論引擎，已正確處理 v4 的 LoRA merge、參考頻譜取樣率、`no_grad`、長文切句、v2Pro 的 `sv_emb`、以及串流。那支腳本是把這些重新手寫一遍然後寫錯幾處。用 `TTS.py` 等於繼承 upstream 的正確性。

---

## 2. 洛神模型的事實

以下全部從 checkpoint 直接解出（本環境無 torch，用 zipfile + 自訂 Unpickler 讀 metadata）。

### 2.1 GPT（s1）`洛神-e15.ckpt`

```
top-level keys : ['weight', 'config', 'info']
weight         : 295 tensors
info           : 'GPT-e15'
config.pretrained_s1          : 'GPT_SoVITS/pretrained_models/s1v3.ckpt'
config.output_dir             : 'logs/洛神/logs_s1_v4'
config.train.half_weights_save_dir : 'GPT_weights_v4'
config.train.exp_name         : '洛神'
config.train.epochs           : 15
config.train.precision        : '16-mixed'
config.data.max_sec           : 54
config.model.n_layer          : 24
config.model.hidden_dim       : 512
config.model.embedding_dim    : 512
config.model.head             : 16
config.model.linear_units     : 2048
config.model.phoneme_vocab_size : 732
config.model.vocab_size       : 1025
config.model.EOS              : 1024
config.inference.top_k        : 15
```

重點：
- `phoneme_vocab_size = 732` → 文字前端用 **v2 符號集**（`symbols2.py`）
- `pretrained_s1 = s1v3.ckpt` → s1 底模是 v3/v4 系列共用的那個
- **s1（GPT）在 v2Pro 與 v3/v4 之間是共用的**，所以這份 ckpt 兩邊都能用。只有 SoVITS 那半要對應版本。

### 2.2 SoVITS v4 `洛神_e8.pth`

```
檔頭前 2 bytes : b'04'
lora_rank      : 32
weight         : 258 tensors（其中 176 個是 LoRA）
info           : '8epoch_296iteration'
```

GPT-SoVITS 把 zip 的前兩個 byte 換成版本標記（`process_ckpt.py:30-38` 的 `my_save2()`）。查 `process_ckpt.py:72-80` 的 `head2version`：

```python
b"04": ["v2", "v4", True]   # 文字版本 v2, 模型版本 v4, is_lora=True
```

LoRA 只掛在 CFM 的 estimator attention 上（`cfm.base_model.model.estimator.transformer_blocks.N.attn.to_{q,k,v}` / `to_out.0`）。這份 ckpt **只含 LoRA 增量 + quantizer codebook**，不是完整 SoVITS → 執行時需要 v4 底模 `s2Gv4.pth`（769 MB）載入後 merge。

### 2.3 SoVITS v2Pro

已確認存在。config 尚未取得（待補）。架構參數取自 upstream `GPT_SoVITS/configs/s2v2Pro.json`：

```
data.sampling_rate   32000      model.inter_channels          192
data.filter_length    2048      model.hidden_channels         192
data.hop_length        640      model.filter_channels         768
data.win_length       2048      model.n_heads                   2
data.n_mel_channels    128      model.n_layers                  6
data.n_speakers        300      model.kernel_size               3
                                model.p_dropout               0.0
                                model.resblock                "1"
                                model.resblock_kernel_sizes    [3,7,11]
                                model.upsample_rates           [10,8,2,2,2]
                                model.upsample_initial_channel 512
                                model.upsample_kernel_sizes    [16,16,8,2,2]
                                model.n_layers_q                3
                                model.gin_channels           1024
                                model.semantic_frame_rate    "25hz"
```

**v2Pro 與 v2 只差兩個值**：`gin_channels` 512 → **1024**、`p_dropout` 0.1 → 0.0。多出來的 512 維就是 SV 聲紋嵌入的位置（見 §5.1）。

`upsample_rates` 乘積 = 10×8×2×2×2 = **640 = hop_length** → 單次前向直出 32 kHz 波形，不需獨立 vocoder。

---

## 3. 為什麼邊緣版用 v2Pro 而不是 v4

### 3.1 架構差異（取自 `TTS_infer_pack/TTS.py` 實際分支）

| | **v4** | **v2Pro** |
|---|---|---|
| Synthesizer | `SynthesizerTrnV3`（`TTS.py:548`） | `SynthesizerTrn`（`TTS.py:539`） |
| semantic → 音訊 | CFM flow-matching **8 步 ODE**，再過獨立 HiFiGAN vocoder | **單次前向**，`use_vocoder = False`（`TTS.py:545`） |
| 輸出取樣率 | 48 kHz | 32 kHz |
| 額外權重 | `s2Gv4.pth` 769 MB + `vocoder.pth` 58 MB | SV 聲紋 ckpt（小） |
| 額外程式碼 | `f5_tts/`（68 K，CFM 的 DiT） | `eres2net/`（88 K）+ `sv.py`（1.2 K） |
| 額外依賴 | `peft`（LoRA merge）、`x_transformers` | — |
| upstream 串流支援 | 無 | **有**（`TTS.py:1068`、`:1431` `decode_streaming`） |

關鍵在那個 **8**：v4 每個 chunk 要跑 8 次 DiT 前向，再過一個 480× 上採樣到 48 kHz 的 vocoder。v2Pro 是一次前向出波形。

### 3.2 品質：upstream 自己說 v2Pro 更好

GPT-SoVITS 官方 README「V2Pro Release Notes」原文：

> **"Slightly higher VRAM usage than v2, surpassing v4's performance, with v2's hardware cost and speed."**

以及同段：

> **"v1/v2 and the v2Pro series share the same characteristics, while v3/v4 have similar features. For training sets with average audio quality, v1/v2/v2Pro can deliver decent results, but v3/v4 cannot. Additionally, the synthesized tone and timebre of v3/v4 lean more toward the reference audio rather than the overall training set."**

兩個含義：

1. **v3/v4 挑訓練資料**。資料品質中等時 v3/v4 撐不住，v1/v2/v2Pro 還能給出像樣結果。
2. **v4 的音色偏向「參考音訊」，v2Pro 偏向「整個訓練集」**。對一個 fine-tune 出來的角色聲音，通常要的是訓練出的音色穩定重現，而非跟著每次的參考片段跑。

所以 **v2Pro 不是「快但差一點」的妥協選項** —— 依官方說法它品質更好且更快。

### 3.3 保留意見（必須寫下來）

- 以上是**官方說法，不是本專案實測**。最終要用耳朵判斷。
- v4 原生 48 kHz，v2Pro 32 kHz。但人聲能量基本在 8–16 kHz 以下，32 kHz 已完整涵蓋語音頻帶；48 kHz 的差別主要在寬頻音樂素材。**對 TTS 而言 32 kHz 不是品質瓶頸**，除非下游有 48 kHz 的檔案規格要求。
- **v2ProPlus 比 v2Pro 更好且仍是單次前向** → 若追求品質上限，server 版應一併支援並實測比較。

---

## 4. 設計原則：「不動音質」的操作型定義

這是本專案最硬的約束，所以要定義清楚什麼叫「不動音質」。

**只做可證明等價的優化。** 分三類：

### ✅ 類別一：可證明等價 —— 會做

輸出張量在數值上完全相同，不是近似：

| 優化 | 為何等價 |
|---|---|
| 參考音訊衍生物離線預算（§5.1） | 同樣的輸入經同樣的模型，得到同樣的張量。只是算的時機從執行時移到離線 |
| BERT 截掉第 23、24 層（§5.2） | `TextPreprocessor.py:197` 取 `hidden_states[-3]` = 第 22 層輸出。後兩層的結果從未被讀取 |
| 移除 BERT 的 MLM 輸出頭（§5.2） | 21128×1024 的詞表投影，輸出從頭到尾沒被使用 |
| 英文不呼叫 BERT（§5.3） | `TextPreprocessor.py:215-218` 對非中文**本來就回傳 `torch.zeros`**。不呼叫與呼叫後丟棄等價 |
| 拆除 `pytorch-lightning` 包裝（§5.4） | 推論只用到 `t2s_model.Text2SemanticDecoder`。`LightningModule` 只是訓練期的外殼 |
| 刪除訓練 / 匯出 / 其他介面程式碼（§7） | 推論路徑不經過 |

### ⚠️ 類別二：會改變行為 —— 不做（除非先證明等價）

| 項目 | 為何不做 |
|---|---|
| 砍 G2PW（635 MB）改用 pypinyin fallback | 中文多音字消歧會變差。`chinese2.py:28` 寫死 `is_g2pw = True`，`:192` 有 fallback 路徑但品質較低。**決定：留 G2PW** |
| 用自寫 zh/en 分段器取代 `LangSegmenter` + `lid.176.bin`（131 MB） | 混語分段影響音素序列 → 影響音訊。而且 `langsegmenter.py` 已有 `full_en()` / `full_cjk()` 的 regex 快速路徑，純中文或純英文**根本不會呼叫 lid 模型**，省下的只是混語情境。**決定：留著**。要換必須先在語料上證明音素序列完全一致，再以 flag 開啟 |
| int8 量化 | 音質風險，需實測。**決定：不做** |

### ❓ 類別三：需要實測才能決定 —— 做成選項

| 項目 | 說明 |
|---|---|
| fp16 vs fp32 | upstream GPU 路徑預設 fp16（`is_half`），但數值上與 fp32 不同。做成可切換，預設跟隨 upstream |
| `torch.compile` | 對 AR decode 有幫助但首次編譯慢。做成選項，預設關閉 |
| 取樣參數 | 見 §9 |

---

## 5. 產出 A：邊緣版

**定位**：v2Pro + 洛神固定，離線、最小、實時。無版本分支、無 config 解析、無 i18n、無多模型抽象。

### 5.1 核心手法：參考音訊衍生物離線預算

追 `TTS.py` 的 `prompt_cache`，**參考音訊只產生四樣東西**：

| 預算內容 | 來源 | 原本需要 | 行號 |
|---|---|---|---|
| `prompt_semantic` | SSL 特徵 → `vq.extract_latent` | **CNHubert（378 MB）** | `TTS.py:833` |
| `refer_spec` | 32 kHz 頻譜 | librosa / torchaudio | `TTS.py:763-776` |
| `sv_emb` | eres2net 聲紋嵌入 | **SV 模型** | `TTS.py:1259-1264` |
| prompt 的 `phones` + `bert_features` | 參考文本 | BERT（僅一次） | `TTS.py:1164-1165` |

最終合成只吃這些：

```python
# TTS.py:1316-1317 / 1332-1333
audio = self.vits_model.decode(pred_semantic, phones, refer_audio_spec,
                               speed=speed_factor, sv_emb=sv_emb)
```

固定參考音訊後，這四樣**永遠是同一組常數**。

**→ 離線算完存成一個 tensor 檔，邊緣版完全不需要帶 CNHubert 和 SV 模型。**

省 **378 MB + 聲紋模型**，而且是**零品質損失**（張量完全相同）。連參考 `.wav` 本身、以及 librosa 的載入 / 重採樣路徑都不用進去。

預算腳本放 `tools/`，在有 GPU 的機器上跑，產出 `edge/` 需要的常數檔。

### 5.2 BERT 的兩處浪費

1. **第 23、24 層算完從未被讀取**。`TextPreprocessor.py:197`：
   ```python
   res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
   ```
   `hidden_states[-3]` 對 24 層模型 = 第 22 層輸出（`hidden_states` 有 25 個元素：embeddings + 24 層）。
   → **截斷 encoder 到 22 層**

2. **用 `AutoModelForMaskedLM` 載入，但 MLM 輸出頭從未被使用**。那是 21128×1024 的詞表投影。
   → **整個移除**

兩項合計約 47 M 參數。

### 5.3 英文不走 BERT

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

純英文句子的 BERT 特徵**本來就是零張量**。→ 純英文輸入完全不呼叫 BERT。

### 5.4 拆除 pytorch-lightning

`AR/models/t2s_lightning_module.py` 的 `Text2SemanticLightningModule` 繼承 `LightningModule`，但推論只用到它內部的 `t2s_model.Text2SemanticDecoder`。拆掉可省一個重依賴並縮短啟動時間。只動載入，不動數學。

### 5.5 串流

**不自己寫。** upstream `TTS.py:1068` 已有 `streaming_mode and not self.configs.use_vocoder` 分支，`:1431` 呼叫 `self.vits_model.decode_streaming(...)` —— 正好只在 v2Pro 這條（無 vocoder）路上有。

註：`GPT_SoVITS/stream_v2pro.py` 不採用。它第一行自述「这是一个实验性质的实现」，且建構在 `torch.jit` 匯出之上。

### 5.6 體積估算（fp16，保留 G2PW）

| 項目 | 大小 | 備註 |
|---|---|---|
| BERT（截 22 層 + 去 MLM 頭） | ~556 MB | 原 325 M 參數，移除約 47 M |
| GPT 洛神-e15 | 155 MB | 已是 fp16 |
| SoVITS v2Pro 洛神 | ~100 MB | 待確認 |
| G2PW onnx | 635 MB | 中文多音字，保留 |
| `lid.176.bin` | 131 MB | 混語分段，保留 |
| 預算的參考張量 | <1 MB | — |
| **合計** | **~1.58 GB** | |

**不帶**：CNHubert 378 MB、SV 模型、v4 底模 769 MB、v4 vocoder 58 MB、BigVGAN、AP_BWE。

**誠實說明**：程式碼可以做到「每一行都用到」，但**權重有硬下限**。在「不動音質」的約束下，1.58 GB 就是地板。要再往下只剩量化，那屬於類別二，不做。

### 5.7 邊緣版不做的事

- 不支援換模型（要換就用 server 版）
- 不支援 v3 / v4
- 不支援中英以外的語言
- 不做 int8
- 不做 WebUI

---

## 6. 產出 B：推論 server

**定位**：供邊緣設備連線的推論主機，可換模型。

- 直接用完整的 `TTS_infer_pack/TTS.py`（原生按 version 分支）
- **同時支援 v4 / v2Pro / v2ProPlus**。多留 v2Pro 系列幾乎免費（`eres2net/` 88 K），且依 §3.2 它可能是品質更好的那條 → 必須能實測比較
- 不支援 v3（v4 是 v3 的替代品，官方自述；可省 `BigVGAN/` 364 K 與 AP_BWE 超解析）
- 模型熱切換 API，權重按需載入
- 串流輸出（`decode_streaming`）

### 6.1 對外介面

| 端點 | 行為 |
|---|---|
| `POST /tts` | **直接回傳音檔 bytes**（不是伺服器路徑） |
| `GET /tts?stream=1` | chunked PCM 串流，供實時播放 |
| `POST /model` | 切換模型 / 版本 |
| `GET /health` | 狀態、當前模型、裝置 |

修正前一版的問題：

- **不接受客戶端指定輸出路徑**（原本是任意檔案寫入）
- 錯誤回**正確的 HTTP status code**，不是 200 + `str(e)`
- **單 worker + 請求佇列**，GPU 操作序列化，避免併發對撞
- 設定走 YAML + env，**零寫死路徑**

---

## 7. 保留 / 刪除清單

### 保留（原封不動，不改寫）

```
TTS_infer_pack/      TTS.py, TextPreprocessor.py, text_segmentation_method.py
AR/                  models/, modules/, utils/            （不含 data/）
module/              models.py, modules.py, attentions.py, commons.py,
                     mel_processing.py, mrte_model.py, quantize.py,
                     core_vq.py, transforms.py
feature_extractor/   cnhubert.py                          （server 版；邊緣版不需要）
text/                中文 + 英文前端（含 G2PW、LangSegmenter）
process_ckpt.py, utils.py, configs/
tools/i18n/, tools/my_utils.py
v2Pro 專用：         sv.py + eres2net/                    （server 版；邊緣版預算後不需要）
v4 專用：            f5_tts/model/                        （server 版）
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

### 權重瘦身

只留 PyTorch 格式。前一版的多餘權重：

| 檔案 | 大小 | 多餘原因 |
|---|---|---|
| `bert/tf_model.h5` | 1303 MB | TensorFlow 版，PyTorch 推論用不到 |
| `bert/flax_model.msgpack` | 1302 MB | Flax 版，同上 |
| `hubert/chinese-hubert-base-fairseq-ckpt.pt` | 1136 MB | fairseq 原始 ckpt，已有 `pytorch_model.bin` |
| `GPT_SoVITS/G2PWModel/g2pW.onnx` | 635 MB | 與 `text/G2PWModel/` 下那份重複 |
| `pretrained_models/g2pw-chinese/pytorch_model.bin` | 318 MB | g2pW 的 PyTorch 版，已有 onnx |

合計 4.69 GB / 8.25 GB = **57%**。

---

## 8. 依賴瘦身

### 移除

`tensorboard`、`pytorch-lightning`（§5.4）、`gradio`、`funasr`、`faster-whisper`、`ctranslate2`、`modelscope`、`torchmetrics`、`av`、`pyopenjtalk`、`g2pk2`、`ko_pron`、`ToJyutping`、`python_mecab_ko`

邊緣版額外移除：`peft`、`x_transformers`、`rotary_embedding_torch`（皆為 v3/v4 CFM 專用）

### 保留

`torch`、`torchaudio`、`transformers`、`numpy<2`、`scipy`、`librosa`、`numba`、`ffmpeg-python`、`jieba_fast` / `jieba`、`cn2an`、`pypinyin`、`g2p_en`、`wordsegment`、`PyYAML`、`chardet`、`onnxruntime`（G2PW）、`fast_langdetect`、`split-lang`、`opencc`

server 版額外保留：`fastapi`、`pydantic`、`peft`、`x_transformers`

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

理由：洛神 ckpt 裡那個 `inference.top_k = 15` 是**訓練 config** 的欄位，不是推論預設；前一版的 20 / 0.6 / 0.6 來源不明。upstream 預設是這個 pipeline 實際被驗證過的設定。

三組都記錄在此，方便 A/B。

---

## 10. Repo 結構

```
edge/      v2Pro + 洛神固定。無版本分支、無 config 解析、無 i18n
server/    完整 TTS.py。多版本 + 可換模型 + 串流
tools/     離線預算腳本（在有 GPU 的機器上跑，產出 edge/ 的常數張量）
docs/      本文件及設計決策記錄
```

`edge/` 與 `server/` **不共用程式碼**。

---

## 11. 驗證策略

### 11.1 本開發環境的限制（必須寫明）

這份專案的程式碼在 Claude Code 的 cloud container 中撰寫，該環境：

- **無 GPU**（無 `nvidia-smi`、無 `/dev/nvidia*`）
- 4 核 CPU / 15 GB RAM
- 未安裝 torch

**因此無法在此環境產出音檔驗證。** 能做到的是靜態保證：

1. 語法與 import 完整性檢查
2. 與 upstream 逐行核對張量流程（形狀、取樣率、dtype、裝置）
3. checkpoint metadata 解析驗證（已完成，見 §2）

**音質與 RTF 必須在目標機器上實測。**

### 11.2 等價性驗證（對應 §4 類別一）

每一項「可證明等價」的優化都要有對應驗證：

| 優化 | 驗證方式 |
|---|---|
| 離線預算參考張量 | 與執行時計算的張量逐元素比對，要求完全相同 |
| BERT 截 22 層 | 截層前後的 `hidden_states[-3]` 輸出逐元素比對 |
| 移除 MLM 頭 | 同上 |
| 英文零 BERT | 與呼叫 BERT 後丟棄的結果比對 |
| 拆 pytorch-lightning | 同一輸入下 `infer_panel` 輸出比對（固定 seed） |

### 11.3 端到端比對

以同一組（參考音訊、參考文本、目標文本、seed、取樣參數）在**改造前後**產出音訊，比對波形。這一步需要 GPU 機器。

---

## 12. 未決事項

| # | 項目 | 狀態 |
|---|---|---|
| 1 | **邊緣設備規格** | 尚未購買。目前按理想情況設計（假設有 CUDA、fp16 可用）。實際硬體確定後需重新評估：純 CPU / ARM 要達到實時，以 24 層 AR GPT（每秒音訊 25 個 token）加每句 22 層 BERT 加 640× 上採樣的 VITS decoder 來看，判斷不樂觀 |
| 2 | **SoVITS v2Pro 洛神的 config** | 已確認檔案存在，config 內容待取得。架構參數暫以 upstream `s2v2Pro.json` 為準 |
| 3 | **v2Pro vs v2ProPlus vs v4 的實際音質** | 需在目標機器上實測比較。§3.2 僅為官方說法 |
| 4 | **邊緣版的定位** | 本地推論是為了「離線 fallback」還是「低延遲」？答案影響是否值得為它犧牲任何東西。目前假設兩者都要，因此不做任何有損優化 |
| 5 | **邊緣版 RAM 下限** | 1.58 GB 權重需放得下，加上執行時的啟動記憶體 |

---

## 13. 引用來源

### 本專案

- 前一版：`Sihrrieidhr/GPT-SoVITS-Lite`（HuggingFace，private → 已確認內容，診斷見 §1）
- 洛神模型：`洛神-e15.ckpt`（GPT）、`洛神_e8.pth`（SoVITS v4+LoRA）、v2Pro 版（待取得 config）

### upstream

- GPT-SoVITS：https://github.com/RVC-Boss/GPT-SoVITS
- 版本特性原文：官方 README「V2Pro Release Notes」、「V4 Release Notes」
- 本文件所有行號引用，皆對應 upstream `main` 分支（本專案基底 commit 將於程式碼落地時釘選）
