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

# 權重目錄：絕對不能刪。
#
# upstream 的程式碼把這兩個相對路徑寫死（sv.py:6、text/chinese2.py:35、
# TTS.py 的 now_dir 組路徑），所以權重只能放在 vendored 樹的內部。
# 這也意味著「重跑 vendor」如果直接 rm -rf GPT_SoVITS，會把使用者下載的
# 好幾 GB 權重一起刪掉 —— 那是個很容易踩而且很痛的陷阱。
#
# 因此不整個刪目錄，而是逐檔比對：清單裡沒有、又不在保護路徑下的檔案才刪。
# 這樣既能清掉 upstream 改版後殘留的舊檔，又不會動到權重。
PROTECTED=(
    "GPT_SoVITS/pretrained_models"
    "GPT_SoVITS/text/G2PWModel"
)

is_protected() {
    local path="$1" pre
    for pre in "${PROTECTED[@]}"; do
        [[ "$path" == "$pre"/* || "$path" == "$pre" ]] && return 0
    done
    return 1
}

# 讀清單
declare -A WANTED=()
order=()
missing=()
while IFS= read -r line; do
    line="${line%%#*}"                       # 去掉行內註解
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    [[ -z "$line" ]] && continue
    WANTED["$line"]=1
    order+=("$line")
    [[ -f "$SRC/$line" ]] || missing+=("$line")
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

# 清掉清單外的殘留（保護路徑除外）
removed=0
if [[ -d "$DEST/GPT_SoVITS" || -d "$DEST/tools" ]]; then
    while IFS= read -r existing; do
        rel="${existing#"$DEST"/}"
        is_protected "$rel" && continue
        [[ -n "${WANTED[$rel]:-}" ]] && continue
        rm -f "$existing"
        removed=$((removed + 1))
    done < <(find "$DEST/GPT_SoVITS" "$DEST/tools" -type f 2>/dev/null)
    # 掃掉變空的目錄，但保護路徑留著
    while IFS= read -r d; do
        rel="${d#"$DEST"/}"
        is_protected "$rel" && continue
        rmdir "$d" 2>/dev/null || true
    done < <(find "$DEST/GPT_SoVITS" "$DEST/tools" -type d -empty 2>/dev/null | sort -r)
fi

copied=0
for line in "${order[@]}"; do
    mkdir -p "$DEST/$(dirname "$line")"
    cp -p "$SRC/$line" "$DEST/$line"
    copied=$((copied + 1))
done

# pretrained_models/ 只有 .gitignore，但目錄本身必須存在：
# TTS_Config 會 os.makedirs("GPT_SoVITS/configs/")，而 sv.py:6 與
# text/chinese2.py:35 預期權重目錄就在那個相對位置上。
mkdir -p "$DEST/GPT_SoVITS/pretrained_models" "$DEST/GPT_SoVITS/text/G2PWModel"

echo
echo ">>> 結果"
echo "  複製檔案數: $copied"
echo "  清掉的殘留: $removed"
echo "  保護未動:   ${PROTECTED[*]}"
echo "  .py 檔案數: $(find "$DEST/GPT_SoVITS" "$DEST/tools" -name '*.py' | wc -l)"
echo "  總體積:     $(du -sh "$DEST/GPT_SoVITS" "$DEST/tools" | awk '{s=$1} END {print s}')"
echo
echo ">>> 下一步：驗證沒有斷鏈"
echo "  python scripts/import_graph.py $DEST --verify"
