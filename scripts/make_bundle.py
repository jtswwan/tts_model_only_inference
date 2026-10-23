#!/usr/bin/env python3
"""打包成「解壓縮就能用」的可攜版本。

產出的目錄完全自給自足：自帶 Python、自帶所有套件、自帶權重。
目標機器上**不需要**安裝 Python、不需要 pip、不需要網路。

為什麼不用一般的 venv：venv 的 pyvenv.cfg 與 script 的 shebang 都寫絕對
路徑，搬到別的位置就壞掉。這裡用 python-build-standalone（透過 uv 取得），
它設計上就是可重定位的，套件直接裝進它自己的 site-packages 而不是 venv。

為什麼不用 PyInstaller 打成單一執行檔：PyTorch 有大量的 .so 與資料檔，
打成單檔會是 2~3GB 的執行檔、啟動時要解壓到暫存目錄，而且 torch 的動態
載入常常打包失敗。一個資料夾反而更可靠也更好除錯。

產出結構
--------
    gsv-portable/
        python/            自帶的 CPython（含所有套件）
        app/               程式碼與權重
        啟動.sh / 啟動.bat  雙擊即可
        README.txt

用法
----
    python3 scripts/make_bundle.py --out /tmp/gsv-portable
    python3 scripts/make_bundle.py --out ... --cuda cu121   # GPU 版
    python3 scripts/make_bundle.py --out ... --archive      # 另外產生 tar.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_VERSION = "3.11"          # 3.10~3.12 都可以，選 3.11 是因為實測過
OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def run(cmd: list[str], **kw) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise SystemExit(f"{BAD} 指令失敗（{r.returncode}）: {' '.join(str(c) for c in cmd)}")


def find_uv() -> str:
    for cand in ("uv", str(Path.home() / ".local/bin/uv"), "/root/.local/bin/uv"):
        if shutil.which(cand) or Path(cand).is_file():
            return cand
    raise SystemExit(
        f"{BAD} 找不到 uv。打包需要它來取得可重定位的 Python。\n"
        "   安裝：curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def copy_tree(src: Path, dst: Path, link: bool) -> None:
    """複製目錄。link=True 時對大檔用硬連結（同一檔案系統才行），省時省空間。

    打包成 tar 時硬連結會被實體化成真正的副本，所以不影響產出。
    """
    for item in src.rglob("*"):
        if "__pycache__" in item.parts or item.name.endswith(".pyc"):
            continue
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if link and item.stat().st_size > 1_000_000:
            try:
                os.link(item, target)
                continue
            except OSError:
                pass
        shutil.copy2(item, target)


RUN_SH = """#!/usr/bin/env bash
# 可攜版啟動器。不需要安裝任何東西。
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PY="./python/bin/python3"
[[ -x "$PY" ]] || { echo "找不到自帶的 Python：$PY" >&2; read -r -p "按 Enter 結束…" _ || true; exit 1; }
cd app/server
exec "../../$PY" -m gsv_server "$@"
"""

RUN_BAT = """@echo off
rem 可攜版啟動器。不需要安裝任何東西。
setlocal
cd /d "%~dp0"
if not exist "python\\python.exe" (
    echo 找不到自帶的 Python：python\\python.exe
    pause
    exit /b 1
)
cd app\\server
"..\\..\\python\\python.exe" -m gsv_server %*
if errorlevel 1 pause
"""

README_TXT = """GPT-SoVITS 推論 server — 可攜版
================================

這個資料夾是自給自足的：自帶 Python、自帶所有套件、自帶模型權重。
目標機器上不需要安裝 Python、不需要 pip、不需要網路。

使用方式
--------
  Linux / macOS :  ./啟動.sh      （多數檔案管理器可直接雙擊）
  Windows       :  啟動.bat       （雙擊）

啟動後瀏覽器開 http://localhost:8000/ 就是控制台。
第一次啟動要載入模型並暖機，CPU 上約一分鐘，GPU 上快很多。

常用參數
--------
  ./啟動.sh --port 9000        換 port
  ./啟動.sh --host 0.0.0.0     開放給同網段的其他機器連

內容
----
  python/   自帶的 CPython 與所有套件
  app/      程式碼與模型權重

注意
----
這個包是為 {platform} 建置的，不能搬到其他作業系統或架構上使用。
建置時間：{built}
torch channel：{channel}
"""


def main() -> int:
    ap = argparse.ArgumentParser(prog="make_bundle.py",
                                 description="打包成解壓縮即可用的可攜版本")
    ap.add_argument("--out", required=True, help="輸出目錄")
    ap.add_argument("--cuda", default="cpu", metavar="CHANNEL",
                    help="PyTorch channel：cpu（預設）/ cu118 / cu121 / cu124")
    ap.add_argument("--python", default=PY_VERSION)
    ap.add_argument("--archive", action="store_true", help="另外產生 .tar.gz")
    ap.add_argument("--bert-fp16-only", action="store_true",
                    help="只帶 fp16 的 BERT，省 582 MB。這份 bundle 只能在 GPU 上跑 —— "
                         "CPU 上 preflight() 會拒絕啟動（fp16 在 CPU 上是真的精度損失）")
    ap.add_argument("--no-weights", action="store_true", help="不含權重（之後用 start.py --download-only 補）")
    ap.add_argument("--copy", action="store_true", help="權重一律實體複製，不用硬連結")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{BAD} {out} 已存在且非空")
    uv = find_uv()
    t0 = time.perf_counter()

    print(f"\n[1/5] 取得可重定位的 Python {args.python}")
    pydir = out / "python"
    pydir.parent.mkdir(parents=True, exist_ok=True)
    staging = out / ".pystage"
    run([uv, "python", "install", "--install-dir", str(staging), args.python])
    # uv 在 install-dir 裡同時會放真目錄與指向它的 symlink。
    # p.is_dir() 會跟隨 symlink，所以必須排除 symlink，否則搬到的是連結，
    # 刪掉 staging 之後就斷了。
    installed = [p for p in staging.iterdir()
                 if p.is_dir() and not p.is_symlink() and p.name.startswith("cpython")]
    if not installed:
        raise SystemExit(
            f"{BAD} uv 沒有裝出 Python。{staging} 的內容："
            + ", ".join(sorted(x.name for x in staging.iterdir())) if staging.exists() else "（不存在）"
        )
    shutil.move(str(installed[0]), str(pydir))
    shutil.rmtree(staging, ignore_errors=True)
    py = pydir / ("python.exe" if os.name == "nt" else "bin/python3")
    if not py.exists():
        cands = list(pydir.rglob("python3")) or list(pydir.rglob("python.exe"))
        if not cands:
            raise SystemExit(f"{BAD} 在 {pydir} 找不到 python 執行檔")
        py = cands[0]
    ver = subprocess.run([str(py), "-c", "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  {OK} {ver} — {py.relative_to(out)}")

    # uv 會在它管理的 Python 裡放 EXTERNALLY-MANAGED，之後拒絕往裡面裝套件。
    # 這份已經搬進 bundle、不再歸 uv 管，所以移除標記是正確的 —— 它本來就是
    # 用來防止動到「系統/工具管理的」直譯器，而這份是我們自己的副本。
    removed = 0
    for marker in pydir.rglob("EXTERNALLY-MANAGED"):
        marker.unlink()
        removed += 1
    if removed:
        print(f"  {OK} 移除 {removed} 個 EXTERNALLY-MANAGED 標記（這份 Python 已屬於 bundle）")

    print(f"\n[2/5] 安裝套件（channel={args.cuda}）")
    index = f"https://download.pytorch.org/whl/{args.cuda}"
    # torchaudio 必須 <2.9：2.9 起 torchaudio.load 改委派給 torchcodec，
    # 而 TTS.py:773 的 _get_ref_spec() 正是用它讀參考音訊。
    run([uv, "pip", "install", "--python", str(py),
         "torch>=2.1,<2.9", "torchaudio>=2.1,<2.9", "--index-url", index])
    run([uv, "pip", "install", "--python", str(py), "-r", str(REPO / "requirements.txt")])

    # pip 產生的 console script 有些 shebang 寫的是「建置當下」的絕對路徑，
    # 搬家後就斷了。shebang 不支援相對路徑，標準解法是讓 sh 先接手、再以
    # 相對於腳本自身的位置去 exec 真正的 python。
    #
    # 用的是 uv 自己為 pip 產生的那一種形式 —— 已驗證可用，不要自創。
    # （自創的代價：第一次試了別的寫法，sh 的引號解析不如預期，43 個腳本
    #   被改成「sh 去執行 Python 內容」，而且只有實際執行才會發現。
    #   所以下面一定要驗證。）
    bindir = pydir / ("Scripts" if os.name == "nt" else "bin")
    if bindir.is_dir() and os.name != "nt":
        q3 = chr(39) * 3
        shim = (
            "#!/bin/sh\n"
            + q3 + "exec' \"$(dirname -- \"$(realpath -- \"$0\")\")/" + py.name
            + "\" \"$0\" \"$@\"\n"
            + "' " + q3 + "\n"
        ).encode()
        rewritten = []
        for script in sorted(bindir.iterdir()):
            if not script.is_file() or script.is_symlink():
                continue
            try:
                raw = script.read_bytes()
            except OSError:
                continue
            lines = raw.split(b"\n")
            if not lines or not lines[0].startswith(b"#!") or b"python" not in lines[0]:
                continue          # 已經是 sh shim，或不是 python 腳本
            script.write_bytes(shim + b"\n".join(lines[1:]))
            script.chmod(0o755)
            rewritten.append(script)

        # 驗證。但不要逐一執行全部 43 個 —— 有些 console script 會載入很大的
        # 資料檔（pypinyin 的拼音字典就讓驗證逾時 60 秒），有些可能等 stdin。
        # 那會讓驗證本身變成不穩定的來源。
        #
        # shim 是機械套用的同樣兩行，所以「形式正確」驗證一次就夠：挑一個
        # 已知輕量的腳本實際執行，確認 sh 有正確把控制權交給 Python。
        if rewritten:
            probe = next((x for x in rewritten if x.name in ("pip", "pip3")), rewritten[0])
            try:
                r = subprocess.run([str(probe), "--version"], capture_output=True,
                                   text=True, timeout=90, cwd="/",
                                   stdin=subprocess.DEVNULL)
                combined = (r.stdout + r.stderr).lower()
            except Exception as exc:
                raise SystemExit(f"{BAD} shim 驗證失敗（{probe.name}）：{type(exc).__name__}: {exc}")
            # sh 誤讀 Python 內容的特徵：找不到指令（127）、": not found"、
            # 或 import 被解析成 ImageMagick 的執行檔
            if r.returncode == 127 or ": not found" in combined or "import-im6" in combined:
                raise SystemExit(
                    f"{BAD} shim 形式不正確 —— {probe.name} 被 shell 誤讀：\n"
                    f"     {combined.strip()[:200]}\n   不要出貨。"
                )
            print(f"  {OK} 改寫 {len(rewritten)} 個 console script 的 shebang"
                  f"（以 {probe.name} 驗證 shim 形式正確）")
    print("\n[3/5] 複製程式碼")
    app = out / "app"
    # server/ 底下的權重很大，用硬連結（同一檔案系統才行，tar 時會實體化）
    copy_tree(REPO / "server", app / "server", link=not args.copy)
    copy_tree(REPO / "scripts", app / "scripts", link=False)
    # 英文前端要的 nltk 語料。放進可攜版，解壓後離線也能念英文
    # （bootstrap.py 會把 NLTK_DATA 指到 app/nltk_data/）。
    if (REPO / "nltk_data").is_dir():
        copy_tree(REPO / "nltk_data", app / "nltk_data", link=not args.copy)
    else:
        print(f"  {WARN} 找不到 {REPO / 'nltk_data'} —— 可攜版將無法合成英文。"
              f"先執行 scripts/fetch_models.sh")
    for f in ("requirements.txt", "README.md"):
        shutil.copy2(REPO / f, app / f)
    if args.bert_fp16_only:
        # GPU 專用的瘦身：只帶 fp16 的 BERT，省 582 MB。
        # fp16 只有在 is_half=True 且非 CPU 時才用得上（TTS.py:490-491），
        # 所以這種 bundle 在 CPU 機器上會被 preflight() 擋下來並說明原因，
        # 而不是安靜地降精度。
        fp32 = app / "server/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        fp16 = Path(str(fp32) + "-fp16")
        if not fp16.is_dir():
            print(f"  {WARN} 找不到 {fp16.name}，先執行 scripts/make_fp16.py")
        elif fp32.is_dir():
            def _bytes(d: Path) -> int:
                return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            saved = _bytes(fp32) - _bytes(fp16)
            shutil.rmtree(fp32)
            print(f"  {OK} 只帶 fp16 的 BERT（省 {saved / 1e6:.0f} MB）—— 這份 bundle 需要 GPU")

    if args.no_weights:
        shutil.rmtree(app / "server/GPT_SoVITS/pretrained_models", ignore_errors=True)
        shutil.rmtree(app / "server/GPT_SoVITS/text/G2PWModel", ignore_errors=True)
        (app / "server/GPT_SoVITS/pretrained_models").mkdir(parents=True, exist_ok=True)
        print(f"  {WARN} 不含權重（--no-weights）")

    print("\n[4/5] 寫入啟動器")
    (out / "啟動.sh").write_text(RUN_SH, encoding="utf-8")
    (out / "啟動.sh").chmod(0o755)
    (out / "啟動.bat").write_text(RUN_BAT, encoding="utf-8")
    (out / "README.txt").write_text(README_TXT.format(
        platform=f"{sys.platform} {os.uname().machine if hasattr(os, 'uname') else ''}".strip(),
        built=time.strftime("%Y-%m-%d %H:%M"), channel=args.cuda), encoding="utf-8")

    def size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    print(f"\n[5/5] 完成（{time.perf_counter()-t0:.0f}s）")
    for label, p in (("python/", pydir), ("app/server/GPT_SoVITS/pretrained_models",
                                          app / "server/GPT_SoVITS/pretrained_models"),
                     ("app/（全部）", app)):
        if p.exists():
            print(f"  {label:44s} {size(p)/1e9:6.2f} GB")
    print(f"  {'合計':44s} {size(out)/1e9:6.2f} GB")

    if args.archive:
        tar = out.with_suffix(".tar.gz")
        print(f"\n打包 {tar} …（硬連結會被實體化，會比目錄大）")
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(out, arcname=out.name)
        print(f"  {OK} {tar} — {tar.stat().st_size/1e9:.2f} GB")

    print(f"\n{OK} 可攜版建好了：{out}")
    print("   整個資料夾複製到目標機器，執行 啟動.sh（或 啟動.bat）即可。")
    print("   目標機器不需要 Python、不需要 pip、不需要網路。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
