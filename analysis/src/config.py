# analysis/src/config.py
from __future__ import annotations
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

# ---------------------------------------------------------
# DB 連線：優先使用 PG_DSN，否則用分段參數組裝
#   PG_DSN 例：postgresql+psycopg://user:pwd@host:5432/dbname
# ---------------------------------------------------------
def get_pg_dsn() -> str:
    dsn = os.getenv("PG_DSN")
    if dsn:
        return dsn

    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "youbike")
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "postgres")

    return f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db}"

# ---------------------------------------------------------
# 訓練查詢預設條件（可由 .env 覆蓋）
#   TRAIN_TABLE 或 TRAIN_SQL 選一種；有 TRAIN_SQL 時會優先生效
# ---------------------------------------------------------
TRAIN_TABLE = os.getenv("TRAIN_TABLE", "youbike_training_labels")
TRAIN_SQL   = os.getenv("TRAIN_SQL")  # 若提供會覆蓋 TRAIN_TABLE 的預設 SQL

TRAIN_CITY  = os.getenv("TRAIN_CITY", "TPE")  # 留空代表全部城市
TRAIN_START = os.getenv("TRAIN_START")        # 例：2025-08-25 00:00
TRAIN_END   = os.getenv("TRAIN_END")          # 例：2025-09-01 00:00

# 其他舊設定（保留相容；純時間特徵版其實用不到）
HORIZON_MIN   = int(os.getenv("HORIZON_MIN", "5"))
LOOKBACK_MIN  = int(os.getenv("LOOKBACK_MIN", "15"))
DEFAULT_CITY  = os.getenv("CITY", "TPE")
DEFAULT_SNO   = os.getenv("SNO") or None
