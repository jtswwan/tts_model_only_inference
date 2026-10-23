"""把 BERT 剪到推論實際用得到的部分。等價優化，輸出數值完全相同。

upstream 只取 BERT 的一個中間層：

    TextPreprocessor.py:196-197
        res = self.bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

於是有三處白做的工：

1. hidden_states[-3] 是倒數第三個元素。hidden_states 有 num_layers + 1 個
   元素（embeddings 輸出 + 每層輸出），所以 [-3] 等於索引 num_layers - 2，
   也就是「倒數第二層」的輸出 —— 最後兩層算完從來沒被讀取。

2. 模型以 AutoModelForMaskedLM 載入，但 MLM 輸出頭（1024→1024 的 transform
   加上 1024→vocab 的投影）的結果從頭到尾沒被使用。那個 vocab 投影是每次
   請求都要做的大矩陣乘法。

3. BertModel 預設會算 pooler_output，同樣沒被使用。

**關於負索引的陷阱**：單純把 encoder 截到 num_layers - 2 層是不等價的。
截完以後 hidden_states 只有 num_layers - 1 個元素，[-3] 會指到更前面的層，
輸出就變了。所以截層之後必須把 hidden_states 補回原本的長度（重複最後一個
元素），讓 [-3] 仍然落在同一層的輸出上。下面的 _FeatureOnlyBert 就是做這件事。

本模組不修改任何 upstream 檔案，只在 TTS 初始化完成後替換掉 tts.bert_model。
等價性由 scripts/verify_equivalence.py 實際比對驗證。
"""

from __future__ import annotations

import torch
from torch import nn


class _FeatureOnlyBert(nn.Module):
    """只跑到需要的那一層，並把 hidden_states 補回原長度。

    對呼叫端（TextPreprocessor.get_bert_feature）而言行為與原模型一致：
    接受同樣的 keyword 參數，回傳的物件可用 ["hidden_states"] 取值，
    且 [-3] 指到同一層的輸出。
    """

    def __init__(self, bert_base: nn.Module, original_hidden_states_len: int):
        super().__init__()
        self.bert = bert_base
        self.original_hidden_states_len = original_hidden_states_len

    def forward(self, output_hidden_states: bool = True, **inputs):
        # output_hidden_states 由呼叫端傳入且恆為 True；這裡一律要求輸出，
        # 因為我們唯一需要的就是它。
        out = self.bert(**inputs, output_hidden_states=True)
        hidden = list(out.hidden_states)
        # 補到原本的長度，使負索引語意不變
        missing = self.original_hidden_states_len - len(hidden)
        if missing > 0:
            hidden.extend([hidden[-1]] * missing)
        return {"hidden_states": tuple(hidden)}


def needed_layer_index(num_layers: int) -> int:
    """upstream 的 hidden_states[-3] 實際對應的層索引（1-based 層編號）。

    hidden_states 長度 = num_layers + 1，[-3] = 索引 num_layers - 2，
    索引 0 是 embeddings 輸出，索引 i 是第 i 層的輸出。
    """
    return num_layers - 2


PROBE_TEXTS = (
    "慢慢來也沒關係，只要沒有停下來，你其實一直都在前進。",
    "今天天氣不錯。",
    "A",
    "一二三四五六七八九十，1234567890，百分之五十。",
    "這個專案用 v2Pro 模型，sample rate 是 32000 Hz。",
)


def probe(model: nn.Module, tokenizer, device, texts=PROBE_TEXTS) -> list[torch.Tensor]:
    """完全照 TextPreprocessor.py:196-197 的方式取值，回傳每段文字的結果。

    用於截層前後的等價性自檢：取法必須和真正會被使用的那條路徑一模一樣，
    否則驗了也不算驗。
    """
    out = []
    with torch.no_grad():
        for text in texts:
            inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
            res = model(**inputs, output_hidden_states=True)
            out.append(torch.cat(res["hidden_states"][-3:-2], -1).clone())
    return out


class TrimEquivalenceError(RuntimeError):
    """截層後的輸出與原模型不一致。"""


def assert_equivalent(before: list[torch.Tensor], after: list[torch.Tensor]) -> None:
    """要求逐位元相同。不是近似，不給容差。

    這兩次前向是同樣的輸入、同樣的權重、同樣的運算順序，結果本來就該完全
    相同。有任何差異就代表剪裁邏輯不成立（最可能的原因是 hidden_states 的
    負索引補齊沒對上），而不是浮點誤差。
    """
    if len(before) != len(after):
        raise TrimEquivalenceError(f"探針數量不符: {len(before)} vs {len(after)}")
    for i, (ref, got) in enumerate(zip(before, after)):
        if ref.shape != got.shape:
            raise TrimEquivalenceError(
                f"探針 {i} 形狀不符: {tuple(ref.shape)} vs {tuple(got.shape)}"
            )
        if not torch.equal(ref, got):
            diff = (ref.float() - got.float()).abs().max().item()
            raise TrimEquivalenceError(
                f"探針 {i} 數值不符，最大絕對誤差 {diff:.3e}。\n"
                "BERT 剪裁不等價 —— 這會默默影響音質。\n"
                "請在 config.yaml 設 latency.trim_bert: false 後重新啟動，"
                "並回報 transformers 版本。"
            )


def trim(mlm_model: nn.Module) -> tuple[nn.Module, dict]:
    """原地剪裁並回傳 (替換用的模組, 說明用的統計資料)。

    傳入的是 TTS.init_bert_weights 建出來的 AutoModelForMaskedLM，
    此時它已經搬到目標裝置且套用了 half()。
    """
    base = getattr(mlm_model, "bert", None)
    if base is None:
        raise TypeError(
            f"預期是 BERT 系列的 MaskedLM 模型（需有 .bert），收到 {type(mlm_model).__name__}"
        )

    layers = base.encoder.layer
    num_layers = len(layers)
    keep = needed_layer_index(num_layers)
    if keep < 1:
        raise ValueError(f"層數太少（{num_layers}），無法剪裁")

    original_hidden_states_len = num_layers + 1
    before = sum(p.numel() for p in mlm_model.parameters())

    # 1. 丟掉用不到的最後兩層
    base.encoder.layer = nn.ModuleList(list(layers)[:keep])
    if hasattr(base.config, "num_hidden_layers"):
        base.config.num_hidden_layers = keep

    # 2. 丟掉 pooler（BertModel.forward 對 None 有判斷）
    if getattr(base, "pooler", None) is not None:
        base.pooler = None

    wrapper = _FeatureOnlyBert(base, original_hidden_states_len)

    # 3. MLM 輸出頭不再被引用 —— 包裝只持有 base。
    #    注意詞表投影的權重與 input embeddings 綁定（tie_word_embeddings），
    #    所以省下的是 transform 那層與 bias，不是整個詞表矩陣。
    after = sum(p.numel() for p in wrapper.parameters())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "num_layers_original": num_layers,
        "num_layers_kept": keep,
        "needed_layer_index": keep,
        "hidden_states_len_original": original_hidden_states_len,
        "params_before": before,
        "params_after": after,
        "params_removed": before - after,
        "removed_pct": round(100.0 * (before - after) / before, 2) if before else 0.0,
    }
    return wrapper, stats
