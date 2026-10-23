#!/usr/bin/env python3
"""把一組新的模型 + 參考音訊裝進這個 server。

server 的模型與參考音訊都在 server/config.yaml 裡，不是寫死的。但換一組
新的並不只是改路徑 —— 有幾個條件必須滿足，不滿足時失敗的樣子不一定明顯：

  1. 權重必須放在 server/GPT_SoVITS/pretrained_models/ 底下。
     upstream 把這個相對路徑寫死在 sv.py:6 與 text/chinese2.py:35，
     放別的地方會找不到。

  2. config 的 version 必須和 SoVITS 權重的「檔頭」一致。
     版本是從檔案前兩個 byte 判定的（process_ckpt.py:100），不是從設定。
     不一致時 upstream 會照檔案走並安靜地自動修正。

  3. v3/v4 的角色權重只含 LoRA 增量，需要對應底模才能 merge。

  4. 參考音訊長度必須在 3~10 秒之間（TTS.py:816 的硬性檢查），
     而 reference.text 必須是那段音訊的逐字稿 —— 不符會影響音色。

  5. 路徑打錯時 upstream 會「安靜退回預設路徑」（TTS.py:342-353），
     結果是載到底模而不是你的角色模型，只有一行 print。

這支工具把以上全部檢查完、檔案放好、並印出該填的設定。

用法
----
    python scripts/add_voice.py --name 新角色 \\
        --sovits /path/新角色_e8.pth \\
        --gpt    /path/新角色-e15.ckpt \\
        --ref    /path/參考.wav \\
        --ref-text "參考音訊的逐字稿"

    # 檢查完直接寫進 config.yaml
    ... --apply

    # 或套用到執行中的 server（不重啟）
    ... --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))

# bootstrap.setup() 會 chdir 到 server/，所以使用者在命令列給的相對路徑
# 必須在那之前就固定下來，否則會被重新解讀成相對於 server/。
_INVOKED_FROM = Path.cwd()


def user_path(value: str) -> Path:
    """把使用者給的路徑解成絕對路徑，基準是「執行這支程式時的目錄」。"""
    p = Path(value)
    return p if p.is_absolute() else (_INVOKED_FROM / p)


from gsv_server import bootstrap  # noqa: E402

VENDOR = bootstrap.setup()
WEIGHTS_DIR = VENDOR / "GPT_SoVITS" / "pretrained_models"

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

# v3/v4 的 LoRA 權重需要的底模（TTS.py:560-581 會載它再 merge）
LORA_BASE = {
    "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
}
# v3/v4 另外需要的 vocoder
EXTRA = {
    "v4": ["GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth"],
}


def inspect_sovits(path: Path) -> dict:
    """從檔頭判定版本。這是 server 執行時實際採用的判定方式。"""
    from process_ckpt import get_sovits_version_from_path_fast

    text_version, model_version, is_lora = get_sovits_version_from_path_fast(str(path))
    with open(path, "rb") as f:
        head = f.read(2)
    return {
        "header": head.decode("latin-1"),
        "text_version": text_version,
        "model_version": model_version,
        "is_lora": bool(is_lora),
    }


def inspect_gpt(path: Path) -> dict:
    """讀 GPT (s1) checkpoint 的 metadata，不載權重到裝置上。"""
    import torch

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(d, dict) or "weight" not in d:
        raise ValueError("不是 GPT-SoVITS 的 s1 checkpoint（缺少 'weight' 鍵）")
    cfg = d.get("config") or {}
    model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
    return {
        "info": d.get("info"),
        "tensors": len(d["weight"]),
        "phoneme_vocab_size": model_cfg.get("phoneme_vocab_size"),
        "n_layer": model_cfg.get("n_layer"),
        "pretrained_s1": cfg.get("pretrained_s1") if hasattr(cfg, "get") else None,
    }


def check_reference(path: Path, target_sr: int) -> tuple[dict, list[str]]:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    ch, n = wav.shape
    dur = n / sr
    info = {"sr": sr, "channels": ch, "seconds": round(dur, 2),
            "peak": round(wav.abs().max().item(), 3)}
    problems = []
    # TTS.py:816：16kHz 下樣本數必須在 48000~160000 之間
    if not (3.0 <= dur <= 10.0):
        problems.append(f"長度 {dur:.2f}s 不在 3~10 秒之間（TTS.py:816 會報錯）")
    if info["peak"] < 0.05:
        problems.append(f"峰值只有 {info['peak']}，音量過低")
    if ch > 2:
        problems.append(f"{ch} 聲道無法處理（只支援單聲道或立體聲）")
    return info, problems


def convert_reference(src: Path, dst: Path, target_sr: int) -> list[str]:
    """只做可證明不影響輸出的轉換。"""
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(src))
    notes = []
    if wav.shape[0] == 2:
        wav = wav.mean(0, keepdim=True)
        notes.append("立體聲 → 單聲道（_get_ref_spec 本來就會 mean(0)）")
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        notes.append(f"{sr} Hz → {target_sr} Hz（server 的原生取樣率）")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), (wav.clamp(-1, 1) * 32767).to(torch.int16),
                    target_sr, encoding="PCM_S", bits_per_sample=16)
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(prog="add_voice.py",
                                 description="把一組新的模型 + 參考音訊裝進這個 server")
    ap.add_argument("--name", required=True, help="這組聲音的名字，會用作目錄名")
    ap.add_argument("--sovits", required=True, help="SoVITS (s2) 權重 .pth")
    ap.add_argument("--gpt", required=True, help="GPT (s1) 權重 .ckpt")
    ap.add_argument("--ref", required=True, help="參考音訊")
    ap.add_argument("--ref-text", required=True, help="參考音訊的逐字稿")
    ap.add_argument("--ref-lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--apply", action="store_true", help="檢查通過後寫進 config.yaml")
    ap.add_argument("--server", default=None, help="套用到執行中的 server，例如 http://localhost:8000")
    ap.add_argument("--dry-run", action="store_true", help="只檢查，不複製任何檔案")
    args = ap.parse_args()

    sovits, gpt, ref = user_path(args.sovits), user_path(args.gpt), user_path(args.ref)
    for label, p in (("--sovits", sovits), ("--gpt", gpt), ("--ref", ref)):
        if not p.is_file():
            print(f"{BAD} {label} 找不到檔案: {p}")
            return 1
    if not args.ref_text.strip():
        print(f"{BAD} --ref-text 不可為空（它是參考音訊的逐字稿，影響音色）")
        return 1

    print("=" * 68)
    print(f"檢查 {args.name}")
    print("=" * 68)

    # ---- 1. SoVITS 版本（決定 config 的 version） ----
    try:
        sv_info = inspect_sovits(sovits)
    except Exception as exc:
        print(f"{BAD} 無法判定 {sovits.name} 的版本: {type(exc).__name__}: {exc}")
        return 1
    version = sv_info["model_version"]
    print(f"\n[SoVITS] {sovits.name}")
    print(f"  檔頭 {sv_info['header']!r} → version={version} "
          f"text_version={sv_info['text_version']} is_lora={sv_info['is_lora']}")
    print(f"  {OK} config.yaml 的 version 必須設成 {version}")

    problems: list[str] = []
    extra_needed: list[str] = []
    if sv_info["is_lora"]:
        base = LORA_BASE.get(version)
        print(f"  {WARN} 這是只含 LoRA 增量的權重，需要底模 {base} 才能 merge")
        if base and not (VENDOR / base).exists():
            problems.append(f"缺少 {version} 的底模: {base}")
            extra_needed.append(base)
    for extra in EXTRA.get(version, []):
        if not (VENDOR / extra).exists():
            problems.append(f"缺少 {version} 需要的 {extra}")
            extra_needed.append(extra)

    # ---- 2. GPT ----
    try:
        gpt_info = inspect_gpt(gpt)
    except Exception as exc:
        print(f"{BAD} {gpt.name} 不是有效的 s1 checkpoint: {exc}")
        return 1
    print(f"\n[GPT] {gpt.name}")
    print(f"  {gpt_info['tensors']} 個 tensor | info={gpt_info['info']!r} "
          f"| n_layer={gpt_info['n_layer']} | phoneme_vocab_size={gpt_info['phoneme_vocab_size']}")
    if gpt_info["pretrained_s1"]:
        print(f"  底模 {gpt_info['pretrained_s1']}")
    # 文字符號集必須對得上：v2 符號集是 732
    if gpt_info["phoneme_vocab_size"] not in (None, 732) and sv_info["text_version"] == "v2":
        problems.append(
            f"GPT 的 phoneme_vocab_size={gpt_info['phoneme_vocab_size']}，"
            f"但 SoVITS 的 text_version=v2（應為 732）—— 兩者的符號集不一致"
        )
    else:
        print(f"  {OK} 符號集與 SoVITS 的 text_version={sv_info['text_version']} 相符")

    # ---- 3. 參考音訊 ----
    target_sr = 24000 if version == "v3" else 32000
    try:
        ref_info, ref_problems = check_reference(ref, target_sr)
    except Exception as exc:
        print(f"{BAD} 無法讀取 {ref.name}: {type(exc).__name__}: {exc}")
        return 1
    print(f"\n[參考音訊] {ref.name}")
    print(f"  {ref_info['sr']} Hz / {ref_info['channels']} 聲道 / "
          f"{ref_info['seconds']}s / 峰值 {ref_info['peak']}")
    print(f"  逐字稿「{args.ref_text}」")
    problems += ref_problems
    if not ref_problems:
        print(f"  {OK} 長度在 3~10 秒之間，音量正常")
    print(f"  prompt_semantic 約 {round(ref_info['seconds'] * 25)} 個 token "
          f"（會接在 AR 序列前面，影響每次請求的延遲）")

    if problems:
        print(f"\n{'=' * 68}")
        print("不能安裝，原因：")
        for p in problems:
            print(f"  {BAD} {p}")
        if extra_needed:
            print("\n缺少的檔案可用 scripts/fetch_models.sh 取得，或手動放到：")
            for e in extra_needed:
                print(f"  server/{e}")
        return 1

    dest_dir = WEIGHTS_DIR / args.name
    rel = lambda p: os.path.relpath(p, VENDOR)  # noqa: E731
    paths = {
        "vits": dest_dir / sovits.name,
        "t2s": dest_dir / gpt.name,
        "ref": dest_dir / f"{ref.stem}_32k.wav" if target_sr == 32000 else dest_dir / f"{ref.stem}_{target_sr//1000}k.wav",
    }

    if args.dry_run:
        print(f"\n{OK} 檢查全部通過（--dry-run，未複製任何檔案）")
    else:
        print(f"\n複製到 {rel(dest_dir)}/")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sovits, paths["vits"])
        shutil.copy2(gpt, paths["t2s"])
        notes = convert_reference(ref, paths["ref"], target_sr)
        for name in ("vits", "t2s", "ref"):
            print(f"  {rel(paths[name])}")
        for n in notes:
            print(f"    - {n}")

    # ---- 4. 設定 ----
    block = {
        "version": version,
        "models": {"t2s": rel(paths["t2s"]), "vits": rel(paths["vits"])},
        "reference": {"audio": rel(paths["ref"]), "text": args.ref_text, "lang": args.ref_lang},
    }
    print(f"\n{'=' * 68}")
    print("要填進 server/config.yaml 的內容：")
    print("=" * 68)
    print(f"version: {version}")
    print("models:")
    print(f"  t2s:  {block['models']['t2s']}")
    print(f"  vits: {block['models']['vits']}")
    print("reference:")
    print(f"  audio: {block['reference']['audio']}")
    print(f"  text: {block['reference']['text']}")
    print(f"  lang: {block['reference']['lang']}")

    if args.apply and not args.dry_run:
        import yaml

        cfg_path = VENDOR / "config.yaml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        raw["version"] = version
        raw.setdefault("models", {}).update(block["models"])
        raw.setdefault("reference", {}).update(block["reference"])
        backup = cfg_path.with_suffix(".yaml.bak")
        shutil.copy2(cfg_path, backup)
        cfg_path.write_text(
            yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"\n{OK} 已寫入 {rel(cfg_path)}（原檔備份為 {rel(backup)}）")
        print("  註：--apply 會重寫 config.yaml，原本的註解會消失，備份裡還有。")

    # ---- 5. 最終驗證：用 server 自己的 preflight 再確認一次 ----
    if not args.dry_run:
        from gsv_server.engine import PreflightError, preflight
        from gsv_server.settings import load

        s = load()
        s.version = version
        s.models.t2s = block["models"]["t2s"]
        s.models.vits = block["models"]["vits"]
        s.reference.audio = block["reference"]["audio"]
        s.reference.text = block["reference"]["text"]
        s.reference.lang = block["reference"]["lang"]
        try:
            detected = preflight(s)
            print(f"\n{OK} server 的 preflight 檢查通過: {detected}")
        except PreflightError as exc:
            print(f"\n{BAD} server 的 preflight 不通過：\n{exc}")
            return 1

    # ---- 6. 套用到執行中的 server ----
    if args.server and not args.dry_run:
        base_url = args.server.rstrip("/")

        def post(path: str, payload: dict) -> None:
            req = urllib.request.Request(
                f"{base_url}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    print(f"  {OK} POST {path} → {r.status}")
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                print(f"  {BAD} POST {path} → {e.code}: {body[:300]}")
                raise SystemExit(1)

        print(f"\n套用到執行中的 server {base_url}")
        # 先換模型（會重新載入並暖機），再換參考音訊
        post("/model", {"version": version, **block["models"]})
        post("/reference", block["reference"])
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=30) as r:
                h = json.load(r)
            print(f"  version={h['version']} 參考={h['reference_audio'].split('/')[-1]} "
                  f"quality.clean={h['quality']['clean']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {WARN} 無法讀取 /health: {exc}")
    elif not args.apply and not args.dry_run:
        print("\n（加 --apply 可直接寫進 config.yaml；加 --server URL 可套用到執行中的 server）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
