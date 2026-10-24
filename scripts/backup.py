#!/usr/bin/env python3
"""備份與還原「你自己的東西」。

為什麼需要
----------
這個 repo 裡有三種檔案，只有第三種是你的：

1. vendored 的 upstream 程式碼 —— `vendor_upstream.sh` 隨時可以重建
2. 下載來的權重 —— `fetch_models.sh` 隨時可以重抓（約 2.85 GB）
3. **你自己做出來的東西** —— 改過的設定、自訂發音、用 `add_voice.py` 裝的
   模型與參考音訊。這些重做不回來。

第 3 類全部被 `.gitignore` 擋在版控之外（權重太大、設定是本機的），所以
不備份就是真的沒了。

預設只收小檔案（設定 + 字典 + 一份清單），**不**收權重 —— 那些重抓就有。
清單會記下你用的是哪些模型檔，還原時照著補即可。

用法
----
    python3 scripts/backup.py export                 # 產生 gsv-backup-<日期>.tar.gz
    python3 scripts/backup.py export --with-voices   # 連自訂模型與參考音訊一起收
    python3 scripts/backup.py list   <檔案>          # 看裡面有什麼，不解開
    python3 scripts/backup.py restore <檔案>         # 還原（既有檔案會先備份成 .bak）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "server"
WEIGHTS = SERVER / "GPT_SoVITS" / "pretrained_models"

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

# 一定會收的小檔案（相對 repo 根）
SMALL = [
    "server/config.yaml",
    "server/pronunciation.rep",
]

# 這些是 fetch_models.sh 抓得回來的底模，不進備份
STOCK = {
    "chinese-roberta-wwm-ext-large",
    "chinese-roberta-wwm-ext-large-fp16",
    "chinese-hubert-base",
    "fast_langdetect",
    "sv",
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def load_config() -> dict:
    import yaml

    with open(SERVER / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def referenced_paths(cfg: dict) -> dict[str, str]:
    """config 裡指到的模型與參考音訊（相對 server/）。"""
    models = cfg.get("models") or {}
    ref = cfg.get("reference") or {}
    out = {k: v for k, v in models.items() if isinstance(v, str) and v}
    if isinstance(ref.get("audio"), str):
        out["reference.audio"] = ref["audio"]
    return out


def custom_voice_files() -> list[Path]:
    """`add_voice.py` 裝進 pretrained_models/<名字>/ 的東西。"""
    if not WEIGHTS.is_dir():
        return []
    out: list[Path] = []
    for item in sorted(WEIGHTS.iterdir()):
        if item.name in STOCK or item.name.startswith("."):
            continue
        if item.is_dir() and item.name != "reference":
            out.extend(sorted(f for f in item.rglob("*") if f.is_file()))
        elif item.is_dir() and item.name == "reference":
            out.extend(sorted(f for f in item.rglob("*") if f.is_file()))
    return out


def do_export(args: argparse.Namespace) -> int:
    cfg = load_config()
    refs = referenced_paths(cfg)

    files: list[tuple[Path, str]] = []
    for rel in SMALL:
        p = REPO / rel
        if p.is_file():
            files.append((p, rel))
        else:
            print(f"  {WARN} 找不到 {rel}，略過")

    # 一律算出來，才能在沒收的時候提醒使用者有東西沒進備份
    voices = custom_voice_files()
    if args.with_voices:
        for p in voices:
            files.append((p, str(p.relative_to(REPO))))

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "with_voices": bool(args.with_voices),
        # 還原到新機器時，照這份清單確認權重齊不齊
        "config_references": refs,
        "files": [rel for _, rel in files],
        "note": "底模（BERT / CNHubert / SV / fast_langdetect）不在備份裡，"
                "用 scripts/fetch_models.sh 重抓。",
    }

    out = Path(args.out) if args.out else REPO / f"gsv-backup-{dt.date.today()}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        info = tarfile.TarInfo("MANIFEST.json")
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        info.size = len(data)
        import io

        tar.addfile(info, io.BytesIO(data))
        for p, rel in files:
            tar.add(p, arcname=rel)

    total = sum(p.stat().st_size for p, _ in files)
    print(f"{OK} {out}")
    print(f"  {len(files)} 個檔案，原始 {human(total)}，壓縮後 {human(out.stat().st_size)}")
    if not args.with_voices and voices:
        print(f"  {WARN} 有 {len(voices)} 個自訂模型／參考音訊沒收進來"
              f"（{human(sum(f.stat().st_size for f in voices))}）")
        print(f"      要一起備份請加 --with-voices")
    print(f"\n  config 目前指到：")
    for k, v in refs.items():
        exists = (SERVER / v).exists()
        print(f"    {'  ' if exists else WARN} {k:18s} {v}")
    return 0


def do_list(args: argparse.Namespace) -> int:
    with tarfile.open(args.archive, "r:gz") as tar:
        try:
            man = json.loads(tar.extractfile("MANIFEST.json").read())  # type: ignore[union-attr]
        except Exception:
            print(f"{BAD} 這個檔案裡沒有 MANIFEST.json，可能不是本工具產生的")
            return 1
        print(f"建立時間 {man['created']}  含自訂語音：{'是' if man['with_voices'] else '否'}")
        print(f"\n檔案（{len(man['files'])}）：")
        for m in tar.getmembers():
            if m.name == "MANIFEST.json":
                continue
            print(f"  {human(m.size):>10s}  {m.name}")
        print("\nconfig 當時指到：")
        for k, v in man.get("config_references", {}).items():
            print(f"  {k:18s} {v}")
    return 0


def do_restore(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name != "MANIFEST.json"]
        # 不信任壓縮檔裡的路徑：只還原到 repo 底下
        for m in members:
            target = (REPO / m.name).resolve()
            if not str(target).startswith(str(REPO.resolve())):
                print(f"{BAD} 壓縮檔裡有指向 repo 之外的路徑：{m.name}")
                return 1

        restored, skipped = 0, 0
        for m in members:
            target = REPO / m.name
            if target.exists():
                if not args.force:
                    bak = target.with_suffix(target.suffix + ".bak")
                    shutil.copy2(target, bak)
                    print(f"  現有的 {m.name} 已備份成 {bak.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                skipped += 1
                continue
            target.write_bytes(src.read())
            restored += 1
        print(f"{OK} 還原 {restored} 個檔案" + (f"，略過 {skipped}" if skipped else ""))
    print("\n改完設定要重啟 server 才生效。")
    print("底模不在備份裡 —— 若是新機器，先跑 scripts/fetch_models.sh。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="打包你自己的設定與字典")
    e.add_argument("--out", help="輸出檔名")
    e.add_argument("--with-voices", action="store_true",
                   help="連 add_voice.py 裝的模型與參考音訊一起收（會很大）")
    e.set_defaults(func=do_export)

    l = sub.add_parser("list", help="看備份裡有什麼")
    l.add_argument("archive")
    l.set_defaults(func=do_list)

    r = sub.add_parser("restore", help="還原")
    r.add_argument("archive")
    r.add_argument("--force", action="store_true", help="直接覆蓋，不留 .bak")
    r.set_defaults(func=do_restore)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
