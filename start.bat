@echo off
rem 一鍵啟動（Windows）。雙擊即可。
rem 建虛擬環境 -> 裝依賴 -> 下載權重 -> 啟動 server -> 開啟控制台。
setlocal
cd /d "%~dp0"

set PY=
rem 需要 3.10~3.12：upstream 釘 numpy<2.0，它在 3.13 沒有 wheel。
for %%v in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v
    )
)
if not defined PY (
    python -c "import sys" >nul 2>&1 && set PY=python
)
if not defined PY (
    echo 找不到 Python。請先安裝 Python 3.10 ~ 3.12 ^(https://www.python.org/downloads/^)
    pause
    exit /b 1
)

%PY% scripts\start.py %*
if errorlevel 1 pause
