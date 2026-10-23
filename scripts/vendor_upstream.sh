#!/usr/bin/env bash
# 從 upstream GPT-SoVITS 取出推論所需的子集到 server/。
#
# 加法，不是「整份複製再刪除」：只複製 scripts/manifest.txt 列出的路徑。
# 刪除法留下的是未知的多餘檔案；加法留下的是明確清單，而且清單裡每個項目
# 都有寫明理由。
#
# 複製進來的檔案一律原封不動 —— 本專案不改 upstream 的程式碼。需要改變
# 行為時在 server/gsv_server/ 那一層處理（工作目錄、BERT 替換、快取包裝）。
#
# 清單中有任何路徑在來源不存在就中止，不會安靜跳過 —— 漏掉 .py 會在 import
# 時就炸，但漏掉資料檔要等到第一次請求才炸，寧可現在失敗。
#
# 用法: scripts/vendor_upstream.sh <upstream-checkout> [dest] [manifest]
set -euo pipefail

SRC="${1:?用法: $0 <upstream-checkout> [dest] [manifest]}"
DEST="${2:-server}"
MANIFEST="${3:-$(dirname "$0")/manifest.txt}"

[[ -d "$SRC/GPT_SoVITS" ]] || { echo "錯誤: $SRC 看起來不是 GPT-SoVITS checkout" >&2; exit 1; }
[[ -f "$MANIFEST" ]]       || { echo "錯誤: 找不到清單 $MANIFEST" >&2; exit 1; }

echo ">>> 來源 $SRC"
if rev=$(git -C "$SRC" rev-parse HEAD 2>/dev/null); then
    echo "    commit $rev"
fi
echo ">>> 目的 $DEST"
echo ">>> 清單 $MANIFEST"

rm -rf "$DEST/GPT_SoVITS" "$DEST/tools"

copied=0
missing=()
while IFS= read -r line; do
    line="${line%%#*}"                       # 去掉行內註解
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    [[ -z "$line" ]] && continue
    if [[ ! -f "$SRC/$line" ]]; then
        missing+=("$line")
        continue
    fi
    mkdir -p "$DEST/$(dirname "$line")"
    cp -p "$SRC/$line" "$DEST/$line"
    copied=$((copied + 1))
done < "$MANIFEST"

if (( ${#missing[@]} )); then
    echo >&2
    echo "錯誤: 清單中的這些路徑在來源不存在：" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo >&2
    echo "可能是 upstream 改了檔案配置。請重跑" >&2
    echo "  python scripts/import_graph.py $SRC --manifest" >&2
    echo "更新 [reachable] 那一組，其餘各組需人工確認。" >&2
    exit 1
fi

# pretrained_models/ 只有 .gitignore，但目錄本身必須存在：
# TTS_Config 會 os.makedirs("GPT_SoVITS/configs/")，而 sv.py:6 與
# text/chinese2.py:35 預期權重目錄就在那個相對位置上。
mkdir -p "$DEST/GPT_SoVITS/pretrained_models" "$DEST/GPT_SoVITS/text/G2PWModel"

echo
echo ">>> 結果"
echo "  複製檔案數: $copied"
echo "  .py 檔案數: $(find "$DEST/GPT_SoVITS" "$DEST/tools" -name '*.py' | wc -l)"
echo "  總體積:     $(du -sh "$DEST/GPT_SoVITS" "$DEST/tools" | awk '{s=$1} END {print s}')"
echo
echo ">>> 下一步：驗證沒有斷鏈"
echo "  python scripts/import_graph.py $DEST --verify"
