#!/usr/bin/env python3
"""驗證「等價優化」真的等價。需要 GPU 與權重，在目標機器上執行。

README §4 把優化分成三類，類別一宣稱「輸出張量在數值上完全相同」。
宣稱本身沒有價值，要能驗。這支腳本逐項實測：

  1. BERT 截層 + 移除 MLM 頭
     載入兩份模型，一份剪裁一份不剪，比對 TextPreprocessor 實際取用的
     hidden_states[-3]。要求逐元素完全相同（不是近似）。

  2. 文字前端快取
     同一段文字連續兩次，比對未命中與命中的 (phones, bert, norm_text)。

  3. 參考音訊衍生張量的暖機
     算兩次 prompt_semantic / refer_spec / sv_emb，比對是否相同。

  4. 端到端
     暖機後的輸出 vs 強制重算參考張量的輸出。

第 4 項在 GPU 上可能有非位元級的差異（cuDNN 的非決定性 kernel），
所以它回報最大絕對誤差而不是要求完全相同；前三項要求完全相同。

用法:
    python scripts/verify_equivalence.py [--config server/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))

from gsv_server import bootstrap  # noqa: E402

bootstrap.setup()

import torch  # noqa: E402

from gsv_server import bert_trim  # noqa: E402
from gsv_server.settings import load  # noqa: E402

TEXTS = [
    "慢慢來也沒關係，只要沒有停下來，你其實一直都在前進。",
    "今天天氣不錯。",
    "這個專案用 v2Pro 模型，sample rate 是 32000 Hz。",
    "A",
    "測試 mixed language 的 BERT 特徵是否一致。",
    "一二三四五六七八九十，1234567890，百分之五十。",
]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def check_bert_trim(settings) -> bool:
    """載入兩份 BERT，一份剪裁，比對實際被取用的那一層輸出。"""
    print("\n=== 1. BERT 截層 + 移除 MLM 頭 ===")
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    path = settings.resolve(settings.models.bert)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = settings.is_half and device != "cpu"

    tok = AutoTokenizer.from_pretrained(path)

    def build():
        m = AutoModelForMaskedLM.from_pretrained(path).eval().to(device)
        return m.half() if half else m

    original = build()
    trimmed_src = build()
    wrapper, stats = bert_trim.trim(trimmed_src)

    print(
        f"  層數 {stats['num_layers_original']} → {stats['num_layers_kept']}，"
        f"參數 {stats['params_before']:,} → {stats['params_after']:,}"
        f"（省 {stats['params_removed']:,}，{stats['removed_pct']}%）"
    )

    all_ok = True
    with torch.no_grad():
        for text in TEXTS:
            inputs = {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}
            # 完全複製 TextPreprocessor.py:196-197 的取值方式
            ref = torch.cat(original(**inputs, output_hidden_states=True)["hidden_states"][-3:-2], -1)
            got = torch.cat(wrapper(**inputs, output_hidden_states=True)["hidden_states"][-3:-2], -1)
            same = ref.shape == got.shape and torch.equal(ref, got)
            diff = (ref.float() - got.float()).abs().max().item() if ref.shape == got.shape else float("nan")
            all_ok &= _report(f"{text[:24]!r}", same, f"maxdiff={diff:.3e} shape={tuple(got.shape)}")

    del original, trimmed_src, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_ok


def check_engine_caches(settings) -> bool:
    """在真實引擎上檢查文字前端快取與參考張量暖機。"""
    from gsv_server.engine import Engine

    print("\n=== 2-4. 引擎層（文字快取 / 參考張量 / 端到端）===")
    print("  載入模型中…")
    eng = Engine(settings)
    tts = eng.tts
    ok = True

    # --- 2. 文字前端快取：未命中 vs 命中 ---
    print("\n--- 2. 文字前端快取 ---")
    pre = tts.text_preprocessor
    eng.text_cache.clear()
    for text in TEXTS[:4]:
        miss = pre.get_phones_and_bert(text, settings.text_lang, tts.configs.version)
        hit = pre.get_phones_and_bert(text, settings.text_lang, tts.configs.version)
        same = (
            list(miss[0]) == list(hit[0])
            and miss[1].shape == hit[1].shape
            and torch.equal(miss[1], hit[1])
            and miss[2] == hit[2]
        )
        ok &= _report(f"{text[:24]!r}", same, f"phones={len(miss[0])}")
    print(f"  快取統計: {eng.text_cache.stats()}")

    # --- 3. 參考張量暖機 ---
    print("\n--- 3. 參考音訊衍生張量 ---")
    ref_path = settings.resolve(settings.reference.audio)
    first_sem = tts.prompt_cache["prompt_semantic"].clone()
    first_spec = [s[0].clone() if isinstance(s, (list, tuple)) else s.clone()
                  for s in tts.prompt_cache["refer_spec"]]
    tts.set_ref_audio(ref_path)
    second_sem = tts.prompt_cache["prompt_semantic"]
    second_spec = [s[0] if isinstance(s, (list, tuple)) else s for s in tts.prompt_cache["refer_spec"]]
    ok &= _report("prompt_semantic", torch.equal(first_sem, second_sem),
                  f"shape={tuple(second_sem.shape)}")
    spec_same = len(first_spec) == len(second_spec) and all(
        a.shape == b.shape and torch.equal(a, b) for a, b in zip(first_spec, second_spec)
    )
    ok &= _report("refer_spec", spec_same, f"n={len(second_spec)}")

    # --- 4. 端到端 ---
    print("\n--- 4. 端到端（暖機 vs 強制重算參考張量）---")
    text = TEXTS[0]
    warm = [c for _, c in eng.synthesize(text, seed=1234)]
    # 讓 TTS.py:1133 的路徑比對失敗，強迫它重算參考張量
    tts.prompt_cache["ref_audio_path"] = None
    cold = [c for _, c in eng.synthesize(text, seed=1234)]
    import numpy as np

    a, b = np.concatenate(warm), np.concatenate(cold)
    if a.shape != b.shape:
        ok &= _report("波形", False, f"長度不同 {a.shape} vs {b.shape}")
    else:
        maxdiff = int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())
        exact = maxdiff == 0
        _report("波形", exact,
                f"maxdiff={maxdiff} (int16 刻度；GPU 非決定性 kernel 可能造成非零)")
        if not exact:
            print("        注意：非零不必然代表暖機有錯。固定 seed 下 cuDNN 仍可能")
            print("        選到非決定性 kernel。前三項完全相同即表示快取本身正確。")
    print(f"\n  延遲量測: {eng.metrics()['ttfa']}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-engine", action="store_true", help="只驗 BERT 截層，不載入完整引擎")
    args = parser.parse_args()

    settings = load(args.config)
    print(f"裝置: {'cuda' if torch.cuda.is_available() else 'cpu'}  "
          f"version: {settings.version}  is_half: {settings.is_half}")

    ok = check_bert_trim(settings)
    if not args.skip_engine:
        ok &= check_engine_caches(settings)

    print("\n" + ("全部通過" if ok else "有項目未通過 —— 等價性不成立，請勿使用該優化"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
