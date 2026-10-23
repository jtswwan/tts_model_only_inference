"""進入點。

    python -m gsv_server                     # 用 server/config.yaml
    python -m gsv_server --config other.yaml
    GSV_LATENCY__PROFILE=low python -m gsv_server

不要用 `uvicorn gsv_server.api:app` —— app 需要帶著設定建構，
而且 bootstrap 必須在 import upstream 之前完成。前一版的
inference_server.py 正是因為「用什麼指令啟動」會改變實際執行的程式碼
而出問題（README §1.1），這裡只留一條啟動路徑。
"""

from __future__ import annotations

import argparse
import logging

from . import bootstrap

bootstrap.setup()

from .settings import load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="gsv_server", description="GPT-SoVITS 純推論 server")
    parser.add_argument("--config", default=None, help="YAML 設定檔（預設 server/config.yaml）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = load(args.config)
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    import uvicorn

    from .api import create_app

    # workers 固定為 1：模型常駐在記憶體，GPU 操作由 Engine 內的鎖序列化。
    # 多 worker 會讓每個行程各載一份模型並在同一張 GPU 上互相爭搶。
    uvicorn.run(create_app(settings), host=host, port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
