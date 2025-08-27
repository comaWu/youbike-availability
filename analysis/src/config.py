import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PG_DSN = (
    f"postgresql://{os.getenv('PG_USER','postgres')}:{os.getenv('PG_PASSWORD','postgres')}"
    f"@{os.getenv('PG_HOST','localhost')}:{os.getenv('PG_PORT','5432')}/{os.getenv('PG_DB','youbike')}"
)
HORIZON_MIN  = int(os.getenv("HORIZON_MIN", "5"))
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "15"))
DEFAULT_CITY = os.getenv("CITY", "TPE")
DEFAULT_SNO  = os.getenv("SNO") or None