#!/usr/bin/env python3
"""延遲基準測試。主要指標是 TTFA（time-to-first-audio）。

為什麼不是總體 RTF：AR 解碼每秒音訊要生 25 個 semantic token，每個 token
一次 24 層 × 512 維的 forward，天生序列化、無法平行。所以總時間壓不下來，
能壓的是「第一塊音訊多快送出」。詳見 README §5.1。

這支腳本量的是從客戶端看到的真實 TTFA —— 串流端點第一個 byte 到達的時間，
不是伺服器自己回報的數字。只用標準庫，不需要額外依賴。

用法:
    python scripts/bench.py --url http://localhost:8000
    python scripts/bench.py --profiles quality,low --runs 5
    python scripts/bench.py --split-methods cut1,cut3,cut5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

TEXTS = [
    ("短", "你好。"),
    ("中", "慢慢來也沒關係，只要沒有停下來，你其實一直都在前進。"),
    (
        "長",
        "明天的天氣預報是：早上會有一點懶洋洋的微風，適合慢慢醒來；"
        "中午陽光剛好，不刺眼，但足夠讓人想出去走走；"
        "下午可能會飄來幾朵心情很好的雲，把天空點綴得剛剛好；"
        "晚上氣溫微涼，適合聽音樂或發呆。",
    ),
    ("中英混合", "這個 server 用 v2Pro 模型，TTFA 大約 200 ms。"),
]


def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def measure_stream(
    base: str, text: str, profile: str | None, timeout: float
) -> tuple[float, float, int, int]:
    """回傳 (首位元組秒數, 總秒數, 位元組數, 取樣率)。

    首位元組的時間就是客戶端實際等待的 TTFA。
    """
    params = {"text": text}
    if profile:
        params["profile"] = profile
    url = f"{base}/tts/stream?" + urllib.parse.urlencode(params)

    start = time.perf_counter()
    first: float | None = None
    total_bytes = 0
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        sr = int(resp.headers.get("X-Sample-Rate", 0))
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter() - start
            total_bytes += len(chunk)
    return first or 0.0, time.perf_counter() - start, total_bytes, sr


def fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    ms = sorted(v * 1000 for v in values)
    med = statistics.median(ms)
    return f"中位 {med:7.1f} ms   最小 {ms[0]:7.1f}   最大 {ms[-1]:7.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=3, help="每種組合跑幾次（取中位數）")
    ap.add_argument("--profiles", default="quality", help="逗號分隔，例如 quality,low")
    ap.add_argument("--split-methods", default="", help="逗號分隔，會透過 /model 之外的方式提示；留空則用 server 現值")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=1, help="正式計時前先跑幾次（丟棄）")
    args = ap.parse_args()

    base = args.url.rstrip("/")

    try:
        health = get_json(f"{base}/health")
    except urllib.error.URLError as exc:
        print(f"連不上 {base}: {exc}")
        return 1
    if not health.get("ready"):
        print(f"server 尚未就緒: {health}")
        return 1

    print("=== server 狀態 ===")
    for k, v in health.items():
        print(f"  {k:20s} {v}")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if args.split_methods:
        print(f"\n注意：text_split_method 需在 server 端設定（config.yaml 的 latency.text_split_method）。")
        print(f"      本腳本不會替你改它。要比較 {args.split_methods} 請分別啟動 server 後各跑一次。")

    if args.warmup:
        print(f"\n暖機 {args.warmup} 次（不計入）…")
        for _ in range(args.warmup):
            try:
                measure_stream(base, TEXTS[1][1], profiles[0], args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"  暖機失敗: {exc}")
                return 1

    print(f"\n=== TTFA（客戶端收到第一個 byte 的時間），每組 {args.runs} 次 ===")
    rows = []
    for profile in profiles:
        print(f"\n--- profile = {profile} ---")
        print(f"{'文字':12s} {'字數':>5s}  {'TTFA':38s} {'音訊長度':>10s} {'即時倍率':>9s}")
        for label, text in TEXTS:
            ttfas, totals, audio_secs = [], [], []
            for _ in range(args.runs):
                try:
                    first, total, nbytes, sr = measure_stream(base, text, profile, args.timeout)
                except Exception as exc:  # noqa: BLE001
                    print(f"{label:12s} 失敗: {type(exc).__name__}: {exc}")
                    break
                ttfas.append(first)
                totals.append(total)
                audio_secs.append(nbytes / 2 / sr if sr else 0.0)
            if not ttfas:
                continue
            audio = statistics.median(audio_secs)
            total_med = statistics.median(totals)
            # 即時倍率 > 1 表示生成速度快於播放速度，可持續串流不中斷
            rtx = audio / total_med if total_med else 0.0
            print(f"{label:12s} {len(text):5d}  {fmt(ttfas):38s} {audio:9.2f}s {rtx:8.2f}x")
            rows.append(
                {
                    "profile": profile,
                    "label": label,
                    "chars": len(text),
                    "ttfa_median_ms": round(statistics.median(ttfas) * 1000, 1),
                    "total_median_ms": round(total_med * 1000, 1),
                    "audio_seconds": round(audio, 3),
                    "realtime_factor": round(rtx, 3),
                }
            )

    try:
        print("\n=== server 端量測 ===")
        print(json.dumps(get_json(f"{base}/metrics"), ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"  取得 /metrics 失敗: {exc}")

    print("\n=== 判讀 ===")
    print("  即時倍率 > 1 代表生成快於播放，串流可以不中斷。")
    print("  TTFA 是使用者實際感受到的延遲；總時間只決定能不能跟上播放。")
    if any(r["realtime_factor"] < 1 for r in rows):
        print("  有組合的即時倍率 < 1 —— 該長度的文字無法持續即時播放。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
