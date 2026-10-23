#!/usr/bin/env python3
"""從推論進入點計算可達的 upstream 模組，並回報無法解析的 import。

upstream 把 repo 根與 GPT_SoVITS/ 都放進 sys.path，所以同一個模組可能被寫成
`tools.i18n.i18n` 或 `module.models`，兩個根都要搜。

用途有二：
  1. 決定哪些檔案可以剪掉（剪枝前跑在完整的 upstream 上）
  2. 驗證剪枝沒有造成斷鏈（剪枝後跑在 server/ 上，--verify）

注意：以 sys.path.append 在執行期動態載入的模組（sv.py 的 eres2net、
audio_sr.py 的 AP_BWE_main）靜態分析追不到，會出現在未解析清單中。
這些是已知且預期的，列在 DYNAMIC 裡。
"""
import argparse, ast, collections, os, sys

# 文字前端經 cleaner.py 的 __import__("text." + lang) 動態載入，
# 中英文的模組必須在此明確列出。
ENTRIES = [
    "TTS_infer_pack.TTS",
    "process_ckpt",
    "sv",
    "text.cleaner",
    "text.chinese2",
    "text.english",
    "text.symbols2",
    "text.LangSegmenter",
]

# 執行期以 sys.path 動態載入，靜態分析追不到 —— 預期會未解析
DYNAMIC = {
    "ERes2NetV2": "sv.py:5 → GPT_SoVITS/eres2net/ERes2NetV2.py",
    "kaldi": "sv.py:6 → GPT_SoVITS/eres2net/kaldi.py",
    "datasets1": "tools/audio_sr.py:6 → tools/AP_BWE_main/datasets1/",
    "models": "tools/audio_sr.py:6 → tools/AP_BWE_main/models/",
    "env": "BigVGAN 內部相對載入",
    "utils0": "BigVGAN 內部相對載入",
    "activations": "BigVGAN 內部相對載入",
    "alias_free_activation": "BigVGAN 內部相對載入",
    "meldataset": "BigVGAN 內部相對載入",
    "bigvgan": "BigVGAN 內部相對載入",
}

# 已剪除的死分支：靜態分析看得到這個 import，但它在執行期永遠走不到。
# 每一筆都要寫出判定依據，不是為了讓驗證器安靜而加的。
DEAD_BRANCHES = {
    "BigVGAN.alias_free_activation.cuda.activation1d":
        "bigvgan.py:93 僅在 use_cuda_kernel=True 時於方法內載入，"
        "而 TTS.py:626 寫死 use_cuda_kernel=False"
        "（upstream 註解：True 會因缺 Ninja 而 RuntimeError）",
}

STDLIB = set(sys.stdlib_module_names)


def resolve(mod, roots):
    """回傳模組對應的檔案路徑。

    除了 X.py 與 X/__init__.py，也接受沒有 __init__.py 的純目錄 ——
    那是 PEP 420 namespace package，upstream 的 BigVGAN/ 就是這種，
    執行時 import 得到，不算斷鏈。
    """
    rel = mod.replace(".", os.sep)
    for root in roots:
        for cand in (os.path.join(root, rel + ".py"), os.path.join(root, rel, "__init__.py")):
            if os.path.isfile(cand):
                return cand
        pkg_dir = os.path.join(root, rel)
        if os.path.isdir(pkg_dir):
            return pkg_dir
    return None


def imports_of(path, mod):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as e:
        print(f"  !! 語法錯誤 {path}: {e}", file=sys.stderr)
        return [], []
    pkg = mod if path.endswith("__init__.py") else (mod.rsplit(".", 1)[0] if "." in mod else "")
    required, optional = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            required += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = [p for p in pkg.split(".") if p]
                if node.level > 1:
                    parts = parts[: len(parts) - node.level + 1]
                prefix = ".".join(parts)
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            if target:
                required.append(target)
                # `from X import n` 的 n 可能是子模組也可能只是符號，
                # 只有真的解析得到檔案時才當成一條邊。
                optional += [f"{target}.{a.name}" for a in node.names]
    return required, optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="含 GPT_SoVITS/ 與 tools/ 的目錄")
    ap.add_argument("--verify", action="store_true", help="只檢查斷鏈，成功則靜默")
    ap.add_argument("--manifest", action="store_true",
                    help="只印出可達的 .py 路徑，供 scripts/manifest.txt 的 [reachable] 組使用")
    args = ap.parse_args()

    roots = [args.root, os.path.join(args.root, "GPT_SoVITS")]
    seen, queue, unresolved = {}, collections.deque(ENTRIES), {}

    while queue:
        mod = queue.popleft()
        if mod in seen:
            continue
        path = resolve(mod, roots)
        seen[mod] = path
        if path is None:
            unresolved.setdefault(mod.split(".")[0], set()).add(mod)
            continue
        if os.path.isdir(path):
            # namespace package 本身沒有程式碼可解析
            continue
        required, optional = imports_of(path, mod)
        for dep in required:
            if dep in DEAD_BRANCHES:
                continue
            if dep not in seen:
                queue.append(dep)
        for dep in optional:
            if dep not in seen and resolve(dep, roots):
                queue.append(dep)

    kept = sorted({os.path.relpath(p, args.root) for p in seen.values() if p and p.endswith(".py")})

    # 未解析的頂層名稱，扣掉標準庫與已知的動態載入
    external = {k: v for k, v in unresolved.items() if k not in STDLIB and k not in DYNAMIC}
    internal_tops = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            name = entry[:-3] if entry.endswith(".py") else entry
            if name not in ("__pycache__",):
                internal_tops.add(name)
    suspicious = {k: v for k, v in external.items() if k in internal_tops}

    if args.manifest:
        for k in kept:
            print(k)
        return 0

    if args.verify:
        if suspicious:
            print("斷鏈！以下 upstream 內部模組無法解析：")
            for k, mods in sorted(suspicious.items()):
                for m in sorted(mods):
                    print(f"  {m}")
            return 1
        print(f"OK — 可達 {len(kept)} 個檔案，無內部斷鏈")
        print(f"   動態載入（預期未解析）: {', '.join(sorted(k for k in unresolved if k in DYNAMIC))}")
        for mod, why in sorted(DEAD_BRANCHES.items()):
            print(f"   已剪除的死分支: {mod}\n     依據: {why}")
        return 0

    print(f"=== 可達的 .py ({len(kept)}) ===")
    for k in kept:
        print(" ", k)
    print(f"\n=== 未解析的頂層名稱 ===")
    for k in sorted(unresolved):
        tag = "標準庫" if k in STDLIB else (f"動態: {DYNAMIC[k]}" if k in DYNAMIC else "外部套件")
        print(f"  {k:28s} {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
