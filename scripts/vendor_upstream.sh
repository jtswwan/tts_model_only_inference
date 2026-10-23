#!/usr/bin/env bash
# 從 upstream GPT-SoVITS 取出推論所需的子集到 server/。
#
# 保留的檔案一律原封不動 —— 不改 upstream 的程式碼，只決定帶哪些進來。
# 刪除清單由 scripts/import_graph.py 的可達性分析決定，不靠人工判斷；
# 其中四項例外在下方 NOTE 標出，它們是靜態分析看不到的動態載入。
#
# 用法: scripts/vendor_upstream.sh <upstream-checkout> [dest]
set -euo pipefail

SRC="${1:?用法: $0 <upstream-checkout> [dest]}"
DEST="${2:-server}"

[[ -d "$SRC/GPT_SoVITS" ]] || { echo "錯誤: $SRC 看起來不是 GPT-SoVITS checkout" >&2; exit 1; }

echo ">>> 從 $SRC 複製到 $DEST"
rm -rf "$DEST/GPT_SoVITS" "$DEST/tools"
mkdir -p "$DEST"
cp -r "$SRC/GPT_SoVITS" "$DEST/GPT_SoVITS"
cp -r "$SRC/tools" "$DEST/tools"

cd "$DEST"
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo ">>> 刪除：訓練"
rm -rf GPT_SoVITS/prepare_datasets GPT_SoVITS/AR/data GPT_SoVITS/AR/text_processing
rm -f  GPT_SoVITS/s1_train.py GPT_SoVITS/s2_train.py \
       GPT_SoVITS/s2_train_v3.py GPT_SoVITS/s2_train_v3_lora.py \
       GPT_SoVITS/module/data_utils.py GPT_SoVITS/module/losses.py

echo ">>> 刪除：ONNX / TorchScript 匯出"
rm -f  GPT_SoVITS/onnx_export.py \
       GPT_SoVITS/export_torch_script.py GPT_SoVITS/export_torch_script_v3v4.py \
       GPT_SoVITS/AR/models/t2s_lightning_module_onnx.py \
       GPT_SoVITS/AR/models/t2s_model_onnx.py \
       GPT_SoVITS/AR/modules/activation_onnx.py \
       GPT_SoVITS/AR/modules/embedding_onnx.py \
       GPT_SoVITS/AR/modules/patched_mha_with_cache_onnx.py \
       GPT_SoVITS/AR/modules/transformer_onnx.py \
       GPT_SoVITS/module/attentions_onnx.py GPT_SoVITS/module/models_onnx.py

echo ">>> 刪除：其他使用者介面（本專案只提供 server）"
rm -f  GPT_SoVITS/inference_cli.py GPT_SoVITS/inference_gui.py \
       GPT_SoVITS/inference_webui.py GPT_SoVITS/inference_webui_fast.py \
       GPT_SoVITS/download.py
# stream_v2pro.py 是 upstream 自述的實驗性 torch.jit 實作，本專案改用
# TTS.py 內建的 decode_streaming（TTS.py:1068, :1431）
rm -f  GPT_SoVITS/stream_v2pro.py

echo ">>> 刪除：非中英文前端"
rm -rf GPT_SoVITS/text/ja_userdic
rm -f  GPT_SoVITS/text/japanese.py GPT_SoVITS/text/korean.py GPT_SoVITS/text/cantonese.py
rm -f  GPT_SoVITS/feature_extractor/whisper_enc.py

echo ">>> 刪除：未被使用的 f5_tts backbone（v4 的 CFM 只用 dit）"
rm -f  GPT_SoVITS/f5_tts/model/backbones/mmdit.py GPT_SoVITS/f5_tts/model/backbones/unett.py

echo ">>> 刪除：資料集製作工具"
rm -rf tools/asr tools/uvr5 tools/denoise-model
rm -f  tools/cmd-denoise.py tools/slice_audio.py tools/slicer2.py \
       tools/subfix_webui.py tools/assets.py tools/my_utils.py \
       tools/i18n/scan_i18n.py

echo ">>> 刪除：非必要文件與 BigVGAN 自身的訓練/推論腳本"
rm -rf GPT_SoVITS/BigVGAN/tests GPT_SoVITS/BigVGAN/nv-modelcard++
# bigvgan.py 並未 import 這些，可達性分析確認不在推論路徑上
rm -f  GPT_SoVITS/BigVGAN/train.py GPT_SoVITS/BigVGAN/inference.py \
       GPT_SoVITS/BigVGAN/inference_e2e.py GPT_SoVITS/BigVGAN/discriminators.py \
       GPT_SoVITS/BigVGAN/loss.py

echo ">>> 刪除：經查證確實走不到的分支"
# AR/utils/ —— 全樹零引用（get_newest_ckpt / check_txt_file 等訓練期輔助）
rm -rf GPT_SoVITS/AR/utils
# eres2net 的未使用變體 —— sv.py 只 import ERes2NetV2
rm -f  GPT_SoVITS/eres2net/ERes2Net.py GPT_SoVITS/eres2net/ERes2Net_huge.py
# BigVGAN 的 CUDA kernel 路徑 —— bigvgan.py:93 僅在 use_cuda_kernel=True 時
# 於方法內載入，而 TTS.py:626 寫死 False（upstream 註解：True 會因缺 Ninja
# 而 RuntimeError）。順帶移除 activation1d.py import 時編譯 CUDA 的風險。
rm -rf GPT_SoVITS/BigVGAN/alias_free_activation/cuda
# BigVGAN 的權重設定檔 —— 本專案不支援 v3，不會載入 BigVGAN 權重
rm -rf GPT_SoVITS/BigVGAN/configs
rm -f  GPT_SoVITS/BigVGAN/requirements.txt
# 授權文件保留：GPT_SoVITS/BigVGAN/LICENSE、incl_licenses/

# ---------------------------------------------------------------------------
# NOTE 保留的四項，皆為可達性分析曾誤判可刪、經查證必須留下者：
#
#   GPT_SoVITS/eres2net/        sv.py:5 以 sys.path.append(os.getcwd()+...) 動態
#                               載入，靜態分析追不到。v2Pro 的 SV 聲紋模型必需。
#   GPT_SoVITS/BigVGAN/         TTS.py:25 無條件 import，與是否使用 v3 無關。
#   tools/AP_BWE_main/          TTS.py:33 import tools/audio_sr.py，後者再
#                               sys.path.append 後 from datasets1.dataset import。
#   module/{ddp_utils,distrib}.py  core_vq.py:42-43 直接 import，非訓練專用。
#
# 以上四者只保留程式碼，對應的權重不入 repo。
# ---------------------------------------------------------------------------

echo
echo ">>> 結果"
echo "  .py 檔案數: $(find . -name '*.py' | wc -l)"
echo "  總體積:     $(du -sh . | cut -f1)"
