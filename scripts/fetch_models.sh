#!/usr/bin/env bash
# 下載推論所需的權重到 server/GPT_SoVITS/pretrained_models/。
#
# 只抓 PyTorch 格式。前一版的 8.25 GB 裡有 4.69 GB（57%）是多餘的：
# BERT 的 TensorFlow 與 Flax 版各 1.3 GB、CNHubert 的 fairseq 原始 ckpt
# 1.1 GB、重複一份的 g2pW.onnx 635 MB、以及已有 onnx 版的 g2pw-chinese
# PyTorch 權重 318 MB。這裡一個都不抓。
set -euo pipefail

DEST="${1:-server/GPT_SoVITS/pretrained_models}"
mkdir -p "$DEST"

HF=https://huggingface.co
VOICE=Sihrrieidhr/GPT-SoVITS-Lite   # 洛神 GPT、BERT、CNHubert、G2PW、參考音訊
V2PRO=Sihrrieidhr/haah              # 洛神 SoVITS v2Pro
GSV=lj1995/GPT-SoVITS               # upstream 官方權重（SV 聲紋模型）

urlenc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1"; }

get() {  # get <repo> <repo內路徑> <目的路徑>
    local repo="$1" src="$2" out="$DEST/$3"
    if [[ -s "$out" ]]; then
        printf '  已存在，跳過  %s\n' "$3"
        return
    fi
    mkdir -p "$(dirname "$out")"
    printf '  下載中        %s\n' "$3"
    if ! curl -fsSL --retry 3 -o "$out.part" "$HF/$repo/resolve/main/$(urlenc "$src")"; then
        rm -f "$out.part"
        printf '  失敗          %s  (%s: %s)\n' "$3" "$repo" "$src" >&2
        return 1
    fi
    mv "$out.part" "$out"
}

echo ">>> 洛神模型"
get "$VOICE" "models/洛神-e15.ckpt" "洛神-e15.ckpt"
# v2Pro 與 v4 的檔案在各自的 repo 裡同名，落地時改名區分（檔頭分別是 05 與 04）
get "$V2PRO" "洛神_e8.pth"          "洛神_e8_v2Pro.pth"

echo ">>> 參考音訊（固定，啟動時會預先算完其衍生張量）"
get "$VOICE" "audio/19.wav"         "reference/19.wav"

echo ">>> BERT (chinese-roberta-wwm-ext-large) —— 只抓 PyTorch 格式"
for f in config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json added_tokens.json; do
    get "$VOICE" "bert/$f" "chinese-roberta-wwm-ext-large/$f" || true
done

echo ">>> CNHubert (chinese-hubert-base) —— 只抓 PyTorch 格式"
for f in config.json pytorch_model.bin preprocessor_config.json; do
    get "$VOICE" "hubert/$f" "chinese-hubert-base/$f" || true
done

echo ">>> SV 聲紋模型（v2Pro 必需 —— sv.py:6 寫死這個相對路徑）"
get "$GSV" "sv/pretrained_eres2netv2w24s4ep4.ckpt" "sv/pretrained_eres2netv2w24s4ep4.ckpt"

echo ">>> G2PW（中文多音字消歧，保留以免影響音質）"
G2PW_DEST=server/GPT_SoVITS/text/G2PWModel
mkdir -p "$G2PW_DEST"
for f in g2pW.onnx config.py MONOPHONIC_CHARS.txt POLYPHONIC_CHARS.txt \
         bopomofo_to_pinyin_wo_tune_dict.json char_bopomofo_dict.json version record.log; do
    if [[ -s "$G2PW_DEST/$f" ]]; then
        printf '  已存在，跳過  G2PWModel/%s\n' "$f"
        continue
    fi
    printf '  下載中        G2PWModel/%s\n' "$f"
    curl -fsSL --retry 3 -o "$G2PW_DEST/$f" \
        "$HF/$VOICE/resolve/main/$(urlenc "GPT-SoVITS/GPT_SoVITS/text/G2PWModel/$f")" \
        || { rm -f "$G2PW_DEST/$f"; printf '  失敗          G2PWModel/%s\n' "$f" >&2; }
done

echo ">>> 混語分段用的語言偵測模型"
get "$VOICE" "GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin" \
             "fast_langdetect/lid.176.bin" || true

echo ">>> 英文前端需要的 nltk 語料"
# text/english.py:5 用 g2p_en，而 g2p_en 需要 nltk 的 averaged_perceptron_tagger
# 與 cmudict。這兩個是 nltk 在「第一次使用時」才下載的，所以離線機器上英文
# 合成會失敗（中文不受影響，走 chinese2 不碰 g2p_en）。這裡預先抓下來。
if [[ -n "${GSV_SKIP_NLTK:-}" ]]; then
    echo "  已跳過（GSV_SKIP_NLTK）"
elif python3 - <<'PYEOF'
import sys
try:
    import nltk
except ImportError:
    print("  nltk 未安裝，跳過（只影響英文合成）")
    sys.exit(0)
ok = True
for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"):
    try:
        if nltk.download(pkg, quiet=True):
            print(f"  已下載        nltk/{pkg}")
        else:
            print(f"  失敗          nltk/{pkg}")
            ok = False
    except Exception as exc:
        print(f"  失敗          nltk/{pkg}: {exc}")
        ok = False
sys.exit(0 if ok else 1)
PYEOF
then :; else
    echo "  nltk 語料下載未完全成功 —— 中文合成不受影響，英文會失敗。" >&2
    echo "  若網路受限，可在有網路的機器上執行" >&2
    echo "    python -c \"import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')\"" >&2
    echo "  再把 ~/nltk_data 複製過來。" >&2
fi

echo
echo ">>> 完成。各目錄體積："
du -sh "$DEST"/* "$G2PW_DEST" 2>/dev/null | sort -h || true
echo
echo "若要改用 v4（品質比較用），另需 769 MB 的底模與 58 MB 的 vocoder："
echo "  get $GSV gsv-v4-pretrained/s2Gv4.pth    gsv-v4-pretrained/s2Gv4.pth"
echo "  get $GSV gsv-v4-pretrained/vocoder.pth  gsv-v4-pretrained/vocoder.pth"
echo "  並把 v4 版的洛神 LoRA 權重（檔頭 04）放到 $DEST/"
