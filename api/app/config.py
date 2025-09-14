import os
from pathlib import Path
from dotenv import load_dotenv

# 專案根目錄 (包含 analysis/)
ROOT = Path(__file__).resolve().parents[2]

# 讀專案根目錄的 .env
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------
# 路徑與輸出位置
# ---------------------------------------------------------
MODELS_DIR = ROOT / "api" / "assets" / "models" / "latest"
MODELS_DIR.mkdir(parents=True, exist_ok=True)