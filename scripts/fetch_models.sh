#!/usr/bin/env bash
# 下載推論所需的權重與英文 nltk 語料。
#
# 這支是薄包裝 —— 權重清單只有一份，在 scripts/start.py 的 FILES 裡。
# 兩邊各維護一份清單遲早會不同步，所以不那樣做。
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then exec "$cand" scripts/start.py --download-only "$@"; fi
done
echo "找不到 Python 3.10~3.12" >&2; exit 1
