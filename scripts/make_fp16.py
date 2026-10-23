#!/usr/bin/env python3
"""把 BERT 權重轉成 fp16 存一份，省一半磁碟。

為什麼這是安全的
----------------
`TTS.py:490-491` 在 GPU 上本來就會把 BERT 轉成 fp16：

    if self.configs.is_half and str(self.configs.device) != "cpu":
        self.bert_model = self.bert_model.half()

所以 GPU 模式的流程是「硬碟 fp32 -> 載入 -> 轉 fp16 -> 拿去算」。
先轉好存起來，流程變成「硬碟 fp16 -> 載入 -> 轉 fp16（無動作）-> 拿去算」。

fp32 -> fp16 是確定性的轉換：同一個 fp32 數字不管轉幾次、在哪裡轉，都得到
同一個 fp16 數字（實測包含 inf / nan / 次正規數 / overflow 邊界）。所以兩條
路徑的權重**逐位元相同** —— 不是「差不多」，是完全一樣，音質不可能變。

為什麼 CPU 不能用
-----------------
CPU 上 `is_half` 是 False，那個 `.half()` 不會執行，整個 BERT 用 fp32 跑。
這時若給它 fp16 的檔案，transformers 會把它上轉回 fp32 —— 數值是「fp16 精度
裝在 fp32 容器裡」，和真正的 fp32 權重不同。那就是實實在在的精度損失。

所以兩份都留著，由 `engine.preflight()` 依裝置挑。給錯會直接失敗，不會安靜
地降精度。

用法
----
    python3 scripts/make_fp16.py                 # 產生 <bert>-fp16/
    python3 scripts/make_fp16.py --verify        # 只驗證，不產生
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BERT = REPO / "server/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

WEIGHT_NAMES = ("pytorch_model.bin", "model.safetensors")


def fp16_dir(src: Path) -> Path:
    return src.with_name(src.name + "-fp16")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def convert(src: Path, dst: Path) -> tuple[int, int]:
    import torch

    weight = next((src / n for n in WEIGHT_NAMES if (src / n).exists()), None)
    if weight is None:
        raise SystemExit(f"{src} 下找不到 {' 或 '.join(WEIGHT_NAMES)}")
    if weight.suffix == ".safetensors":
        raise SystemExit("目前只處理 pytorch_model.bin；safetensors 請先轉成 .bin")

    dst.mkdir(parents=True, exist_ok=True)
    # 設定與 tokenizer 原樣複製
    for f in src.iterdir():
        if f.is_file() and f.name not in WEIGHT_NAMES:
            shutil.copy2(f, dst / f.name)

    state = torch.load(weight, map_location="cpu")
    converted = 0
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dtype == torch.float32:
            out[k] = v.half()
            converted += 1
        else:
            out[k] = v
    torch.save(out, dst / "pytorch_model.bin")
    return converted, len(state)


def verify(src: Path, dst: Path) -> bool:
    """逐位元比對：fp32 的 .half() 必須等於 fp16 檔案本身。"""
    import torch

    a = torch.load(src / "pytorch_model.bin", map_location="cpu")
    b = torch.load(dst / "pytorch_model.bin", map_location="cpu")
    if set(a) != set(b):
        print(f"  鍵不一致：只在 fp32 有 {sorted(set(a) - set(b))[:5]}，"
              f"只在 fp16 有 {sorted(set(b) - set(a))[:5]}")
        return False

    bad = []
    for k in a:
        va, vb = a[k], b[k]
        if not torch.is_tensor(va):
            continue
        if va.is_floating_point() and va.dtype == torch.float32:
            # GPU 上 TTS.py:491 會做的就是這個 .half()
            if not torch.equal(va.half().view(torch.int16), vb.view(torch.int16)):
                bad.append(k)
        elif not torch.equal(va, vb):
            bad.append(k)
    if bad:
        print(f"  {len(bad)} 個張量不相同，例如：{bad[:5]}")
        return False
    print(f"  {len(a)} 個張量全部逐位元相同")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bert", type=Path, default=DEFAULT_BERT)
    ap.add_argument("--verify", action="store_true", help="只驗證既有的 fp16 目錄")
    args = ap.parse_args()

    src = args.bert
    dst = fp16_dir(src)
    if not src.is_dir():
        raise SystemExit(f"找不到 {src}")

    if not args.verify:
        if dst.exists():
            print(f"{dst} 已存在，先刪掉再重做")
            shutil.rmtree(dst)
        print(f"轉換 {src.name} -> {dst.name}")
        converted, total = convert(src, dst)
        print(f"  {converted}/{total} 個張量轉成 fp16")

    if not dst.is_dir():
        raise SystemExit(f"找不到 {dst}，先不加 --verify 跑一次")

    before = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    after = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"  {human(before)} -> {human(after)}（省 {human(before - after)}）")

    print("逐位元驗證（fp32 的 .half() vs fp16 檔案）")
    if not verify(src, dst):
        print("驗證失敗 —— 不要用這份 fp16 權重")
        return 1
    print("通過：GPU（is_half=True）上兩者給出完全相同的權重")
    return 0


if __name__ == "__main__":
    sys.exit(main())
