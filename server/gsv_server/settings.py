"""設定。全部可由 YAML 檔與環境變數覆寫，程式碼中不寫死任何路徑。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import VENDOR_ROOT

ENV_PREFIX = "GSV_"

# upstream 的 v2_languages 還含 ja / ko / yue / auto / auto_yue，但那些前端
# 模組已在 vendoring 時移除，傳進去會 ImportError。這裡是實際可用的集合。
ALLOWED_LANGS = frozenset({"zh", "en"})


@dataclass
class ModelPaths:
    """四個模型權重。相對路徑以 server/ 為基準。"""

    t2s: str = "GPT_SoVITS/pretrained_models/洛神-e15.ckpt"
    vits: str = "GPT_SoVITS/pretrained_models/洛神_e8_v2Pro.pth"
    bert: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    cnhubert: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base"


@dataclass
class Reference:
    """參考音訊。固定不變 → 衍生張量可在啟動時算完快取（見 engine.warmup）。"""

    audio: str = "GPT_SoVITS/pretrained_models/reference/19.wav"
    text: str = "我想，你们不必担忧我们的成长，会给你们带来威胁或阻碍。"
    # 用 "zh" 而非 "all_zh" / "auto"，原因見 Settings.text_lang 的說明。
    lang: str = "zh"


@dataclass
class Sampling:
    """取樣參數。

    預設值取自 upstream 執行時預設（TTS.py:1011-1013、api_v2.py:30-32），
    而非洛神 ckpt 裡的 config.inference.top_k=15 —— 後者是訓練 config
    的欄位，不是推論預設。詳見 README §9。
    """

    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed_factor: float = 1.0
    seed: int = -1


@dataclass
class Latency:
    """延遲相關設定。

    profile 決定串流策略，兩個選項的品質代價取自 upstream 在
    TTS.py:1026-1033 對自家參數的標註：

      quality  return_fragment=True（upstream 標 "Best Quality"）
               + parallel_infer=True → AR 走 infer_panel_batch_infer，
               與 upstream 預設同一條路徑。逐「句組」吐出。
               TTFA 下限由 text_split_method 的第一段長度決定。

      low      streaming_mode=True（upstream 標 "Medium quality"）
               會強制 parallel_infer=False → AR 改走 infer_panel_naive，
               是另一條程式碼路徑。TTFA 更低，但 upstream 自己說品質較差。

    預設 quality。要用 low 必須明確設定 —— 不為了延遲偷偷犧牲音質。
    """

    profile: str = "quality"

    # 單一請求是否可以用 profile 參數要求 low。
    #
    # 預設 false：low 會啟用 upstream 自評 Medium quality 的 streaming_mode，
    # 若開放給呼叫端，任何客戶端都能繞過「不犧牲音質」這個約束，而且不會
    # 留下任何痕跡。要讓呼叫端有這個選擇，必須由 server 端明確開啟。
    #
    # 若 profile 本身就設成 low，那是 server 擁有者的明確選擇，不受此限。
    allow_low_profile: bool = False

    # upstream 預設 cut1 = 每 4 句一組。cut3 = 逐句（TTFA 更低，但改變
    # 送進 GPT 的文本 → 改變韻律，屬於會影響輸出的調整）。
    text_split_method: str = "cut1"
    fragment_interval: float = 0.3

    # 僅 profile="low" 時生效
    overlap_length: int = 2
    min_chunk_length: int = 16
    fixed_length_chunk: bool = False  # upstream 標 "lower quality"

    # 等價優化（README §4 類別一），預設開啟
    trim_bert: bool = True
    warmup_dummy_run: bool = True
    text_cache_size: int = 512


@dataclass
class Server:
    host: str = "0.0.0.0"
    port: int = 8000
    # 送出的 PCM 區塊大小（樣本數）。只影響網路傳輸，不影響合成。
    pcm_chunk_samples: int = 4096


@dataclass
class Settings:
    version: str = "v2Pro"
    device: str = "auto"  # auto | cuda | cuda:N | cpu
    is_half: bool = True
    # "zh" 在 upstream 的註解是「按中英混合识别」，正是本專案要的中英混合。
    #
    # 不可用 "auto"：TextPreprocessor.py:148-150 的 auto 分支會把偵測到的
    # 語言原樣採用（langlist.append(tmp["lang"])），一旦偵測成日文或韓文
    # 就會去 import text/japanese.py 或 korean.py —— 這兩個模組已在
    # vendoring 時移除（本專案只支援中英文），會直接 ImportError。
    #
    # "zh" 走的是 TextPreprocessor.py:157-169 的 else 分支，該分支對非
    # 英文段落一律以傳入的 language 覆蓋（註解：「因无法区别中日韩文汉字,
    # 以用户输入为准」），因此只會載入 chinese2 與 english 兩個模組。
    text_lang: str = "zh"

    models: ModelPaths = field(default_factory=ModelPaths)
    reference: Reference = field(default_factory=Reference)
    sampling: Sampling = field(default_factory=Sampling)
    latency: Latency = field(default_factory=Latency)
    server: Server = field(default_factory=Server)

    # ---- 衍生 ----

    def resolve(self, rel: str) -> str:
        """把相對路徑解成絕對路徑（以 server/ 為基準）。"""
        p = Path(rel)
        return str(p if p.is_absolute() else VENDOR_ROOT / p)

    def validate(self) -> None:
        if self.version not in ("v2Pro", "v2ProPlus", "v4"):
            raise ValueError(f"不支援的 version: {self.version}（本專案支援 v2Pro / v2ProPlus / v4）")
        if self.text_lang not in ALLOWED_LANGS:
            raise ValueError(
                f"text_lang 只接受 {sorted(ALLOWED_LANGS)}（本專案只 vendored 中英文前端），"
                f"收到 {self.text_lang!r}"
            )
        if self.reference.lang not in ALLOWED_LANGS:
            raise ValueError(f"reference.lang 只接受 {sorted(ALLOWED_LANGS)}，收到 {self.reference.lang!r}")
        if self.latency.profile not in ("quality", "low"):
            raise ValueError(f"latency.profile 必須是 quality 或 low，收到 {self.latency.profile}")
        if self.version == "v4" and self.latency.profile == "low":
            # TTS.py:1071-1073：v3/v4 的 use_vocoder=True，streaming_mode
            # 會被 upstream 自動退回 return_fragment 模式。
            raise ValueError(
                "v4 不支援 streaming_mode（TTS.py:1071 會自動退回分段返回），"
                "請改用 latency.profile=quality"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, data: dict[str, Any], path: str = "") -> None:
    """把 dict 套到 dataclass 上，未知的鍵直接報錯而不是安靜忽略。"""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"設定檔有未知的鍵: {path}{key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} 需要是一個區塊（mapping）")
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def _env_overrides(settings: Settings) -> None:
    """GSV_LATENCY__PROFILE=low 這種形式的環境變數覆寫（雙底線代表層級）。"""
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        parts = env_key[len(ENV_PREFIX):].lower().split("__")
        target = settings
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"{env_key}: 設定中沒有 {part} 這個區塊")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"{env_key}: 設定中沒有 {leaf} 這個項目")
        old = getattr(target, leaf)
        if isinstance(old, bool):
            value: Any = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(old, int):
            value = int(raw)
        elif isinstance(old, float):
            value = float(raw)
        else:
            value = raw
        setattr(target, leaf, value)


def load(config_path: str | os.PathLike[str] | None = None) -> Settings:
    settings = Settings()
    path = Path(config_path) if config_path else VENDOR_ROOT / "config.yaml"
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} 的最外層必須是 mapping")
        _apply(settings, raw)
    _env_overrides(settings)
    settings.validate()
    return settings
