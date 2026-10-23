#!/usr/bin/env python3
"""一鍵啟動：建環境 → 下載權重 → 啟動 server → 自動開啟控制台。

只用標準庫，可以直接用系統 Python 執行（不需要先裝任何東西）。

設計上的一個決定：**先用一個極簡的 HTTP server 佔住 port 並立刻開啟瀏覽器**，
在上面顯示安裝進度，裝完才交棒給真正的 server。否則第一次安裝要等十幾分鐘，
使用者只能對著終端機，不知道卡在哪裡。

用法
----
    python3 scripts/start.py                  # 完整流程
    python3 scripts/start.py --no-browser     # 不自動開瀏覽器
    python3 scripts/start.py --download-only  # 只下載權重
    python3 scripts/start.py --setup-only     # 只建環境，不啟動 server
    python3 scripts/start.py --port 9000
    python3 scripts/start.py --device cpu     # 強制裝 CPU 版 torch
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "server"
WEIGHTS = SERVER / "GPT_SoVITS" / "pretrained_models"
VENV = REPO / ".venv"

# 需要 3.10~3.12：numpy<2.0（upstream 的釘選）在 3.13 沒有 wheel，
# 而 jieba_fast 只有 source tarball 必須就地編譯。
PY_MIN, PY_MAX = (3, 10), (3, 12)

HF = "https://huggingface.co"
VOICE = "Sihrrieidhr/GPT-SoVITS-Lite"   # 洛神 GPT、BERT、CNHubert、G2PW、參考音訊
V2PRO = "Sihrrieidhr/haah"              # 洛神 SoVITS v2Pro
GSV = "lj1995/GPT-SoVITS"               # upstream 官方：SV 聲紋模型

# (repo, repo 內路徑, 落地路徑相對於 server/)
# 只抓 PyTorch 格式 —— 前一版的 8.25GB 裡有 4.69GB 是 TensorFlow/Flax/fairseq
# 的重複權重。
FILES: list[tuple[str, str, str]] = [
    (VOICE, "models/洛神-e15.ckpt", "GPT_SoVITS/pretrained_models/洛神-e15.ckpt"),
    (V2PRO, "洛神_e8.pth", "GPT_SoVITS/pretrained_models/洛神_e8_v2Pro.pth"),
    (VOICE, "audio/19.wav", "GPT_SoVITS/pretrained_models/reference/19.wav"),
    (GSV, "sv/pretrained_eres2netv2w24s4ep4.ckpt",
     "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
]
for _f in ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
           "vocab.txt", "special_tokens_map.json", "added_tokens.json"):
    FILES.append((VOICE, f"bert/{_f}",
                  f"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/{_f}"))
for _f in ("config.json", "pytorch_model.bin", "preprocessor_config.json"):
    FILES.append((VOICE, f"hubert/{_f}", f"GPT_SoVITS/pretrained_models/chinese-hubert-base/{_f}"))
for _f in ("g2pW.onnx", "config.py", "MONOPHONIC_CHARS.txt", "POLYPHONIC_CHARS.txt",
           "bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json", "version", "record.log"):
    FILES.append((VOICE, f"GPT-SoVITS/GPT_SoVITS/text/G2PWModel/{_f}",
                  f"GPT_SoVITS/text/G2PWModel/{_f}"))
FILES.append((VOICE, "GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin",
              "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin"))

# 少了這些檔案英文會失敗（中文不受影響，走 chinese2 不碰 g2p_en）
NLTK = ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict")

STEPS = [
    ("python", "檢查 Python 版本"),
    ("venv", "建立虛擬環境"),
    ("deps", "安裝依賴套件"),
    ("weights", "下載模型權重"),
    ("nltk", "下載英文語料"),
    ("server", "啟動 server 並暖機"),
]

STATUS: dict = {
    "steps": [{"id": i, "label": l, "state": "pending", "detail": ""} for i, l in STEPS],
    "current": None, "error": None, "done": False, "log": [],
}
_lock = threading.Lock()


def set_step(step_id: str, state: str, detail: str = "") -> None:
    with _lock:
        for s in STATUS["steps"]:
            if s["id"] == step_id:
                s["state"] = state
                if detail:
                    s["detail"] = detail
        STATUS["current"] = step_id if state == "running" else STATUS["current"]


def log(line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    with _lock:
        STATUS["log"].append(line)
        del STATUS["log"][:-200]
    print(line, flush=True)


# ----------------------------------------------------------------- 進度頁面

SETUP_PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-SoVITS 環境建置中</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%235aa9f0'/%3E%3C/svg%3E">
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--fg:#e6e8ec;--dim:#9aa3b2;--faint:#6b7280;
--accent:#5aa9f0;--ok:#46c08a;--bad:#e2605a;--mono:ui-monospace,Menlo,Consolas,monospace}
@media(prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#f6f7f9;--panel:#fff;
--line:#dfe3e8;--fg:#1a1d23;--dim:#5a6472;--faint:#8b94a3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);padding:0 16px 40px;
font:15px/1.6 system-ui,-apple-system,"Noto Sans TC",sans-serif}
.wrap{max-width:720px;margin:0 auto}
h1{font-size:19px;margin:30px 0 4px}.sub{color:var(--faint);font-size:13px;margin-bottom:24px}
ol{list-style:none;padding:0;margin:0}
li{display:flex;gap:12px;align-items:flex-start;padding:11px 14px;background:var(--panel);
border:1px solid var(--line);border-radius:9px;margin-bottom:8px}
.ic{width:19px;height:19px;flex:0 0 19px;border-radius:50%;border:2px solid var(--line);
margin-top:2px;display:grid;place-items:center;font-size:11px}
li.running .ic{border-color:var(--accent);border-right-color:transparent;animation:s .8s linear infinite}
li.done .ic{border-color:var(--ok);background:var(--ok);color:#06121f}
li.error .ic{border-color:var(--bad);background:var(--bad);color:#fff}
li.pending{opacity:.5}
@keyframes s{to{transform:rotate(360deg)}}
.t{font-weight:550}.d{color:var(--dim);font-size:12.5px;font-family:var(--mono);margin-top:2px;
word-break:break-all}
pre{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px;
font-family:var(--mono);font-size:11.5px;line-height:1.5;color:var(--dim);max-height:260px;
overflow:auto;margin-top:18px;white-space:pre-wrap}
.err{border-color:var(--bad);color:var(--fg)}
.note{color:var(--faint);font-size:12.5px;margin-top:16px}
</style></head><body><div class="wrap">
<h1>GPT-SoVITS 環境建置中</h1>
<div class="sub">完成後會自動切換到控制台。第一次執行要下載約 2.7 GB 權重，請保持這個頁面開著。</div>
<ol id="steps"></ol><pre id="log">等待中…</pre>
<div class="note" id="note"></div>
</div><script>
const ICON={pending:"",running:"",done:"✓",error:"✕"};
let waiting=false;
async function waitForServer(){
  // 交棒空窗期：uvicorn 會先跑完 lifespan（載入模型 + 暖機）才綁 port，
  // 所以這段時間 port 上沒有任何東西在聽。這裡耐心等，不要 reload ——
  // reload 會讓使用者看到瀏覽器的「無法連線」錯誤頁。
  document.getElementById("note").textContent=
    "server 啟動中，正在載入模型並暖機（第一次較久，CPU 上可能要一分多鐘）…";
  for(;;){
    try{ await fetch("/health",{cache:"no-store"}); location.reload(); return; }
    catch(e){ await new Promise(r=>setTimeout(r,1500)); }
  }
}
async function tick(){
  let s;
  try{ s=await (await fetch("/setup-status",{cache:"no-store"})).json(); }
  catch(e){ if(!waiting){waiting=true;waitForServer();} return; }
  document.getElementById("steps").innerHTML=s.steps.map(x=>
    `<li class="${x.state}"><div class="ic">${ICON[x.state]||""}</div><div>
     <div class="t">${x.label}</div>${x.detail?`<div class="d">${x.detail}</div>`:""}</div></li>`).join("");
  const log=document.getElementById("log");
  const atBottom=log.scrollTop+log.clientHeight>=log.scrollHeight-20;
  log.textContent=s.log.join("\\n")||"等待中…";
  log.className=s.error?"err":"";
  if(atBottom)log.scrollTop=log.scrollHeight;
  if(s.error){document.getElementById("note").textContent="建置失敗，詳情見上方紀錄與終端機。";return;}
  if(s.done){
    // 繼續輪詢：等進度頁自己被關掉、fetch 開始失敗時，才會進入
    // waitForServer() 等真 server 上線。這裡若 return 就不再探測，會卡住。
    document.getElementById("note").textContent="環境就緒，正在交棒給 server…";
    setTimeout(tick,700);return;
  }
  setTimeout(tick,700);
}
tick();
</script></body></html>"""


class SetupHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 不要把每個輪詢都印出來
        pass

    def do_GET(self):
        if self.path.startswith("/setup-status"):
            with _lock:
                body = json.dumps(STATUS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = SETUP_PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# --------------------------------------------------------------------- 工具


def run(cmd: list[str], **kw) -> int:
    """執行指令並把輸出同時送到終端機與進度頁。"""
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, **kw)
    for line in p.stdout:
        log("  " + line.rstrip())
    return p.wait()


def venv_python() -> Path:
    return VENV / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def find_python() -> str:
    """找一個版本可用的 Python。"""
    cur = sys.version_info[:2]
    if PY_MIN <= cur <= PY_MAX:
        return sys.executable
    for minor in range(PY_MAX[1], PY_MIN[1] - 1, -1):
        for name in (f"python3.{minor}", f"python3.{minor}.exe"):
            found = shutil.which(name)
            if found:
                return found
    raise RuntimeError(
        f"需要 Python {PY_MIN[0]}.{PY_MIN[1]}~{PY_MAX[0]}.{PY_MAX[1]}，"
        f"目前是 {cur[0]}.{cur[1]} 且找不到其他版本。\n"
        "原因：upstream 釘 numpy<2.0，而它在 3.13 沒有 wheel；"
        "jieba_fast 也只有 source tarball 需要就地編譯。"
    )


def nvidia_driver() -> str | None:
    """回傳 NVIDIA 驅動版本字串，沒有 GPU 則回 None。"""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        first = r.stdout.strip().splitlines()[0].strip()
        return first or None
    except Exception:
        return None


def cuda_channel(driver: str | None) -> str:
    """依驅動版本挑 PyTorch 的 CUDA channel。

    寫死 cu121 會在舊驅動的機器上裝完才發現跑不起來。CUDA 12.x 的
    minor version compatibility 需要驅動 >= 525.60.13；更舊的只能用 cu118
    （需要 >= 450.80.02）。寧可少用一點新特性也不要裝完才失敗。
    """
    if not driver:
        return "cpu"
    try:
        major = int(driver.split(".")[0])
    except Exception:
        return "cu118"
    return "cu121" if major >= 525 else "cu118"


def download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gsv-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last = 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            now = time.time()
            if total and now - last > 0.4:
                last = now
                set_step("weights", "running",
                         f"{label} — {got/1e6:.0f} / {total/1e6:.0f} MB ({got*100//total}%)")
    tmp.replace(dest)


def human(n: float) -> str:
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------- 步驟


def step_python(args) -> str:
    set_step("python", "running")
    py = find_python()
    out = subprocess.run([py, "-c", "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
                         capture_output=True, text=True).stdout.strip()
    set_step("python", "done", f"{out} — {py}")
    log(f"使用 Python {out}（{py}）")
    return py


def step_venv(py: str) -> Path:
    set_step("venv", "running")
    vpy = venv_python()
    if vpy.exists():
        set_step("venv", "done", f"沿用既有的 {VENV.name}/")
        return vpy
    set_step("venv", "running", f"建立 {VENV.name}/")
    if run([py, "-m", "venv", str(VENV)]) != 0:
        raise RuntimeError(
            "建立虛擬環境失敗。Debian/Ubuntu 上 venv 常是獨立套件，"
            f"請先安裝：sudo apt install python3-venv（或對應你的 Python 版本，"
            f"例如 python{sys.version_info.major}.{sys.version_info.minor}-venv）"
        )
    run([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    set_step("venv", "done", str(VENV))
    return vpy


def step_deps(vpy: Path, device: str, cuda_override: str | None) -> None:
    set_step("deps", "running")
    probe = subprocess.run(
        [str(vpy), "-c", "import torch,transformers,librosa,fastapi,jieba_fast;print(torch.__version__)"],
        capture_output=True, text=True)
    if probe.returncode == 0:
        set_step("deps", "done", f"已安裝（torch {probe.stdout.strip()}）")
        return

    driver = nvidia_driver()
    if cuda_override:
        channel = "cpu" if cuda_override == "cpu" else cuda_override
    elif device == "cpu":
        channel = "cpu"
    elif device == "cuda":
        channel = cuda_channel(driver) if driver else "cu121"
    else:
        channel = cuda_channel(driver)
    index = f"https://download.pytorch.org/whl/{channel}"
    set_step("deps", "running", f"安裝 torch（{channel}）—— 這一步最久")
    if driver:
        log(f"偵測結果：NVIDIA 驅動 {driver} → 使用 {channel}")
    else:
        log(f"偵測結果：找不到 nvidia-smi，裝 {channel} 版")
    log(f"（驅動太舊時 cu121 會裝完才失敗，所以依驅動版本挑；可用 --cuda 覆寫）")
    # torchaudio 必須 <2.9：2.9 起 torchaudio.load 改委派給 torchcodec，
    # 而 TTS.py:773 的 _get_ref_spec() 正是用它讀參考音訊。
    if run([str(vpy), "-m", "pip", "install", "torch>=2.1,<2.9", "torchaudio>=2.1,<2.9",
            "--index-url", index]) != 0:
        raise RuntimeError("安裝 torch 失敗")
    set_step("deps", "running", "安裝其餘依賴")
    if run([str(vpy), "-m", "pip", "install", "-r", str(REPO / "requirements.txt")]) != 0:
        raise RuntimeError("安裝依賴失敗")
    set_step("deps", "done", f"虛擬環境 {human(dir_size(VENV))}")


def step_weights() -> None:
    set_step("weights", "running")
    todo = [(r, s, d) for r, s, d in FILES if not (SERVER / d).exists()]
    if not todo:
        set_step("weights", "done", f"已存在（{human(dir_size(WEIGHTS) + dir_size(SERVER/'GPT_SoVITS/text/G2PWModel'))}）")
        return
    log(f"需要下載 {len(todo)} / {len(FILES)} 個檔案")
    for i, (repo, src, dest) in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] {Path(dest).name}"
        url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(src)}"
        set_step("weights", "running", label)
        try:
            download(url, SERVER / dest, label)
            log(f"{label} ✓")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"下載 {src} 失敗（HTTP {e.code}）。"
                               f"若是 private repo，請先把它設為 public 或手動放置檔案。")
    size = dir_size(WEIGHTS) + dir_size(SERVER / "GPT_SoVITS" / "text" / "G2PWModel")
    set_step("weights", "done", human(size))


def step_nltk(vpy: Path) -> None:
    set_step("nltk", "running")
    code = ("import nltk,sys\n"
            f"pkgs={NLTK!r}\n"
            "ok=all(nltk.download(p, quiet=True) for p in pkgs)\n"
            "sys.exit(0 if ok else 1)\n")
    rc = subprocess.run([str(vpy), "-c", code], capture_output=True, text=True).returncode
    if rc == 0:
        set_step("nltk", "done", "已就緒")
    else:
        # 只影響英文；中文走 chinese2 不碰 g2p_en
        set_step("nltk", "done", "下載失敗 — 只影響英文合成，中文不受影響")
        log("nltk 語料下載失敗。中文合成不受影響；英文需要在有網路的機器上執行")
        log("  python -c \"import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')\"")
        log("  再把 ~/nltk_data 複製過來。")


def wait_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(prog="start.py", description="一鍵建環境並啟動 GPT-SoVITS 控制台")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="決定裝 CPU 版還是 CUDA 版的 torch")
    ap.add_argument("--cuda", default=None, metavar="CHANNEL",
                    help="覆寫 PyTorch 的 CUDA channel，例如 cu118 / cu121 / cu124 / cpu。"
                         "預設依 nvidia-smi 回報的驅動版本自動挑")
    args = ap.parse_args()

    if args.download_only:
        try:
            step_weights()
            for st in STATUS["steps"]:
                if st["id"] == "weights":
                    print(f"權重：{st['detail']}")
            return 0
        except Exception as exc:
            print(f"失敗: {exc}", file=sys.stderr)
            return 1

    httpd = None
    url = f"http://127.0.0.1:{args.port}/"
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), SetupHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        log(f"進度頁面 {url}")
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except OSError as exc:
        log(f"（port {args.port} 已被占用，略過進度頁面：{exc}）")

    try:
        py = step_python(args)
        vpy = step_venv(py)
        step_deps(vpy, args.device, args.cuda)
        step_weights()
        step_nltk(vpy)
    except Exception as exc:
        with _lock:
            STATUS["error"] = str(exc)
        for s in STATUS["steps"]:
            if s["state"] == "running":
                s["state"] = "error"
                s["detail"] = str(exc)[:200]
        log(f"\n建置失敗: {exc}")
        if httpd:
            log("進度頁面仍在執行，按 Ctrl-C 結束。")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                httpd.shutdown()
        return 1

    if args.setup_only:
        set_step("server", "done", "--setup-only，未啟動")
        with _lock:
            STATUS["done"] = True
        log("\n環境建置完成。啟動方式：")
        log(f"  cd server && {venv_python()} -m gsv_server")
        if httpd:
            httpd.shutdown()
        return 0

    # 交棒：關掉進度頁面，把 port 讓給真正的 server
    set_step("server", "running", "載入模型並暖機（第一次較久）")
    with _lock:
        STATUS["done"] = True
    time.sleep(1.6)          # 讓進度頁面有機會讀到 done 並準備重新整理
    if httpd:
        httpd.shutdown()
        httpd.server_close()
    time.sleep(0.3)

    log(f"\n啟動 server（{args.host}:{args.port}）…")
    log("（uvicorn 會先載入模型並暖機才開始監聽，這段時間 port 上沒有東西 —— 瀏覽器那邊會自動等待）")
    cmd = [str(venv_python()), "-m", "gsv_server", "--host", args.host, "--port", str(args.port)]
    proc = subprocess.Popen(cmd, cwd=str(SERVER))
    if not args.no_browser:
        def reopen():
            if wait_port(args.port, 180):
                time.sleep(0.5)
                webbrowser.open(url)
        threading.Thread(target=reopen, daemon=True).start()
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
