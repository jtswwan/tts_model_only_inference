"""GPT-SoVITS 純推論 server。

vendored 的 upstream 程式碼在 server/GPT_SoVITS 與 server/tools 底下，
一律原封不動。本套件只負責外層：設定、暖機、快取、介面、量測。
"""

__all__ = ["bootstrap", "settings", "engine", "api"]
