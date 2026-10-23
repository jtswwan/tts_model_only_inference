#!/usr/bin/env python3
"""把模型與參考音訊處理成能讓 server 最快啟動的形式。

**先說這個工具不能做什麼。** 實測的啟動時間分佈（CPU，4 核）：

    import TTS_infer_pack（pytorch_lightning / x_transformers / matplotlib）  7.8 s
    init_bert_weights                                                        8.9 s   ← 可處理
    init_vits_weights                                                        2.0 s
    SV / CNHubert / GPT 合計                                                 1.4 s
    暖機 set_ref_audio                                                       2.3 s
    暖機 dummy 推論                                                         11.9 s

這個工具只能動 BERT 那 8.9 秒。import 那 7.8 秒要拆掉 pytorch-lightning
才能省，而那需要修改 upstream（本專案不改）；dummy 暖機是真的推論，
不可能離線算掉。

**對每次請求的延遲，離線處理幾乎幫不上忙。** 唯一的槓桿是參考音訊長度：
prompt_semantic 會被接在 AR 序列前面，每個生成的 token 都要注意它，所以
較短的參考音訊讓每次請求都快一點。但那會改變輸出，所以本工具只「回報」，
不自動改。

子命令
------
    bert       離線剪裁 BERT 並轉成 fp16 safetensors（會逐位元驗證等價）
    reference  檢查並轉換參考音訊，回報它對每次請求延遲的影響
    merge-lora 把 v3/v4 的 LoRA 權重與底模預先 merge，省掉每次啟動的 merge

用法
----
    python scripts/prepare.py bert      --out <目錄> [--dtype fp16|fp32]
    python scripts/prepare.py reference <輸入.wav> --out <輸出.wav>
    python scripts/prepare.py merge-lora --out <輸出.pth>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
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

bootstrap.setup()

from gsv_server.settings import load  # noqa: E402

OK = "\033[32m✓\033[0m"
BAD = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"


# --------------------------------------------------------------------- BERT


def cmd_bert(args) -> int:
    """離線剪裁 BERT，存成 fp16 safetensors。

    做三件事，全部可證明等價（見 server/gsv_server/bert_trim.py 的說明）：
      1. 剪到推論實際取用的那一層（hidden_states[-3] = 第 num_layers-2 層）
      2. 轉成目標 dtype
      3. 存成 safetensors（mmap 載入，不經過 pickle）

    並在 config 裡留下 gsv_pretrimmed 與 gsv_original_num_hidden_layers，
    讓 server 知道不要再剪一次 —— 對 22 層的模型再剪會變成取第 20 層。

    **dtype 必須和 server 執行時一致**，否則「等價」不成立：
      GPU（is_half=true）執行時是 fp16，所以用 fp16
      CPU（is_half 被強制 false）執行時是 fp32，所以用 fp32
    fp16 不是近似 —— server 本來就會對 fp32 權重做 .half()，離線先做
    與執行時再做得到完全相同的數值。
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from gsv_server import bert_trim

    settings = load(args.config)
    src = settings.resolve(settings.models.bert)
    out = user_path(args.out)
    if not os.path.isdir(src):
        print(f"{BAD} 找不到 BERT 目錄: {src}")
        return 1
    if out.exists() and not args.force:
        print(f"{BAD} {out} 已存在（加 --force 覆寫）")
        return 1

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"來源   {src}")
    print(f"輸出   {out}")
    print(f"dtype  {args.dtype}（server 執行時必須用對應的 is_half 設定）")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(src)
    original = AutoModelForMaskedLM.from_pretrained(src).eval()
    load_before = time.perf_counter() - t0
    n_layers = original.config.num_hidden_layers
    params_before = sum(p.numel() for p in original.parameters())
    print(f"\n原始   {n_layers} 層 / {params_before:,} 參數 / 載入 {load_before:.2f}s")

    # 基準：server 執行時實際會用到的輸出
    reference_model = original.half() if dtype == torch.float16 else original
    baseline = bert_trim.probe(reference_model, tok, "cpu")

    keep = bert_trim.needed_layer_index(n_layers)
    print(f"剪裁   保留前 {keep} 層（推論只取 hidden_states[-3] = 第 {keep} 層的輸出）")

    import torch.nn as nn

    trimmed = AutoModelForMaskedLM.from_pretrained(src).eval()
    trimmed.bert.encoder.layer = nn.ModuleList(list(trimmed.bert.encoder.layer)[:keep])
    trimmed.config.num_hidden_layers = keep
    setattr(trimmed.config, bert_trim.PRETRIMMED_FLAG, True)
    setattr(trimmed.config, bert_trim.ORIGINAL_LAYERS_KEY, n_layers)
    trimmed = trimmed.to(dtype)
    params_after = sum(p.numel() for p in trimmed.parameters())

    # 驗證：用和 server 一樣的包裝（補齊 hidden_states 長度）再比對
    wrapper, stats = bert_trim.trim(trimmed)
    after = bert_trim.probe(wrapper, tok, "cpu")
    try:
        bert_trim.assert_equivalent(baseline, after)
    except bert_trim.TrimEquivalenceError as exc:
        print(f"\n{BAD} 等價性驗證失敗，不寫出任何檔案：\n{exc}")
        return 1
    print(f"{OK} 等價性驗證通過：{len(baseline)} 段探針文字的輸出逐位元相同")

    out.mkdir(parents=True, exist_ok=True)
    # 存的是「剪過但仍是 BertForMaskedLM」的完整結構 —— 因為 TTS.py:487
    # 寫死用 AutoModelForMaskedLM 載入，存成 BertModel 會讓它抱怨缺權重
    # 並隨機初始化。MLM 頭很小（詞表投影與 embeddings 綁定），留著無妨，
    # server 執行時仍會在記憶體裡把它丟掉。
    to_save = AutoModelForMaskedLM.from_pretrained(src).eval()
    to_save.bert.encoder.layer = nn.ModuleList(list(to_save.bert.encoder.layer)[:keep])
    to_save.config.num_hidden_layers = keep
    setattr(to_save.config, bert_trim.PRETRIMMED_FLAG, True)
    setattr(to_save.config, bert_trim.ORIGINAL_LAYERS_KEY, n_layers)
    to_save.to(dtype).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    def dirsize(p):
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)

    t0 = time.perf_counter()
    AutoModelForMaskedLM.from_pretrained(out)
    load_after = time.perf_counter() - t0

    print(f"\n{'':20s} {'前':>14s} {'後':>14s}")
    print(f"{'層數':20s} {n_layers:>14d} {keep:>14d}")
    print(f"{'參數':20s} {params_before:>14,} {params_after:>14,}")
    print(f"{'磁碟':20s} {dirsize(src)/1e6:>13.1f}M {dirsize(out)/1e6:>13.1f}M")
    print(f"{'載入時間':20s} {load_before:>13.2f}s {load_after:>13.2f}s")
    if load_before > 0:
        print(f"\n{OK} 載入快了 {load_before - load_after:.2f}s（{(1-load_after/load_before)*100:.0f}%）")
    print(f"\n接下來把 config.yaml 的 models.bert 指到：\n  {os.path.relpath(out, REPO / 'server')}")
    if args.dtype == "fp16":
        print(f"{WARN} fp16 版本只在 is_half=true（GPU）時等價。CPU 請用 --dtype fp32。")
    else:
        print(f"{WARN} fp32 版本對應 is_half=false。GPU 上要用 fp16 才省得到記憶體頻寬。")
    return 0


# ---------------------------------------------------------------- reference


def cmd_reference(args) -> int:
    """檢查並轉換參考音訊，回報它對每次請求的延遲影響。

    只做可證明不影響輸出的轉換：
      - 轉成 server 的原生取樣率（避免 _get_ref_spec 每次啟動重採樣）
      - 立體聲轉單聲道（_get_ref_spec 本來就會 audio.mean(0)）
      - 存成 16-bit PCM

    不做的：裁切靜音。那會改變 prompt_semantic，進而改變輸出。
    會回報它值多少，由你決定。
    """
    import numpy as np
    import torch
    import torchaudio

    settings = load(args.config)
    target_sr = 32000 if settings.version != "v3" else 24000
    src = user_path(args.input)
    if not src.is_file():
        print(f"{BAD} 找不到 {src}")
        return 1

    wav, sr = torchaudio.load(str(src))
    ch, n = wav.shape
    dur = n / sr
    print(f"來源   {src}")
    print(f"       {sr} Hz / {ch} 聲道 / {dur:.2f}s")

    problems = []
    # TTS.py:816 的硬性限制（換算成 16kHz 的樣本數 48000~160000）
    if not (3.0 <= dur <= 10.0):
        problems.append(
            f"長度 {dur:.2f}s 不在 3~10s 之間 —— TTS.py:816 會因此報錯。"
            "（它檢查 16kHz 下的樣本數在 48000~160000 之間）"
        )
    peak = wav.abs().max().item()
    if peak < 0.05:
        problems.append(f"峰值只有 {peak:.3f}，音量過低")
    if peak >= 1.0:
        print(f"{WARN} 峰值 {peak:.3f} 已達滿刻度；_get_ref_spec 會做 audio /= min(2, maxx)")

    # 每次請求的影響
    prompt_tokens = round(dur * 25)  # semantic_frame_rate = 25hz
    print(f"\n對每次請求的影響")
    print(f"  prompt_semantic 約 {prompt_tokens} 個 token（{dur:.2f}s × 25 Hz）")
    print(f"  這些 token 會被接在 AR 序列前面，每個生成的 token 都要注意它們。")
    print(f"  參考音訊每短 1 秒，AR 序列就短 25 個 token。")
    if dur > 6:
        print(f"  {WARN} {dur:.2f}s 偏長。縮短會讓每次請求都快一些，")
        print(f"     但會改變 prompt_semantic → 改變輸出。本工具不自動做。")

    if problems:
        print()
        for p in problems:
            print(f"{BAD} {p}")
        return 1

    # 可證明不影響輸出的轉換
    changed = []
    if ch == 2:
        wav = wav.mean(0, keepdim=True)
        changed.append("立體聲 → 單聲道（_get_ref_spec 本來就會 mean(0)）")
    elif ch > 2:
        print(f"{BAD} {ch} 聲道無法處理")
        return 1
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        changed.append(f"{sr} Hz → {target_sr} Hz（省掉每次啟動的重採樣）")

    out = user_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16)
    torchaudio.save(str(out), pcm, target_sr, encoding="PCM_S", bits_per_sample=16)

    print(f"\n{OK} 寫出 {out}")
    if changed:
        for c in changed:
            print(f"  - {c}")
    else:
        print("  （來源已是最佳格式，只重存為 16-bit PCM）")
    print(f"\n接下來把 config.yaml 的 reference.audio 指到那個檔，")
    print(f"並確認 reference.text 是這段音訊的逐字稿。")
    return 0


# --------------------------------------------------------------- merge-lora


def cmd_merge_lora(args) -> int:
    """把 v3/v4 的 LoRA 權重與底模預先 merge。

    v3/v4 的角色權重只含 LoRA 增量（檔頭 03 / 04），server 每次啟動都要
    載底模、套 LoRA、再 merge_and_unload()（TTS.py:560-581）。預先做完
    可以省掉這些，同時也不用再帶 769MB 的底模。

    v2Pro 不需要這個 —— 它的權重是完整的（檔頭 05）。
    """
    import torch

    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    settings = load(args.config)
    vits = settings.resolve(settings.models.vits)
    if not os.path.isfile(vits):
        print(f"{BAD} 找不到 {vits}")
        return 1
    text_version, model_version, is_lora = get_sovits_version_from_path_fast(vits)
    print(f"{settings.models.vits}")
    print(f"  檔頭判定: text={text_version} model={model_version} is_lora={is_lora}")
    if not is_lora:
        print(f"\n{OK} 這不是 LoRA 權重，不需要 merge。")
        print("   （v2Pro 的權重本來就是完整的，啟動時直接載入）")
        return 0

    print(f"\n{WARN} 這是 {model_version} 的 LoRA 權重。預先 merge 需要底模。")
    print("   目前尚未實作 —— 本專案主力是 v2Pro，它不需要這個步驟。")
    print("   若你要走 v4，請先確認底模存在，再回報需求。")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="prepare.py", description="把模型與參考音訊處理成最快啟動的形式"
    )
    ap.add_argument("--config", default=None, help="YAML 設定檔（預設 server/config.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bert", help="離線剪裁 BERT 並轉 fp16 safetensors")
    b.add_argument("--out", required=True)
    b.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_bert)

    r = sub.add_parser("reference", help="檢查並轉換參考音訊")
    r.add_argument("input")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_reference)

    m = sub.add_parser("merge-lora", help="預先 merge v3/v4 的 LoRA 權重")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_merge_lora)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
