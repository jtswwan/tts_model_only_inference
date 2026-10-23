#!/usr/bin/env bash
# 一鍵啟動（Linux / macOS）。多數檔案管理器可以直接雙擊。
#
# 做的事：建虛擬環境 → 裝依賴 → 下載權重 → 啟動 server → 開啟控制台。
# 第一次執行會下載約 2.7 GB 權重；之後啟動會跳過已完成的步驟。
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# 需要 3.10~3.12：upstream 釘 numpy<2.0，它在 3.13 沒有 wheel。
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' 2>/dev/null; then
            PY="$cand"; break
        fi
        [[ -z "$PY" ]] && PY="$cand"   # 留著當 fallback，讓 start.py 自己報錯
    fi
done
if [[ -z "$PY" ]]; then
    echo "找不到 Python。請先安裝 Python 3.10 ~ 3.12。" >&2
    read -r -p "按 Enter 結束…" _ || true
    exit 1
fi

"$PY" scripts/start.py "$@"
code=$?
# 雙擊啟動時終端機會在結束時關閉，留個暫停才看得到錯誤
if [[ $code -ne 0 && ! -t 0 ]]; then read -r -p "發生錯誤，按 Enter 結束…" _ || true; fi
exit $code
