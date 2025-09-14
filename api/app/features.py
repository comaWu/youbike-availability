# analysis/src/features.py
# --- Time-only features (no lags / no recent state) --------------------------
from __future__ import annotations
import numpy as np
import pandas as pd

# 時區處理
try:
    from zoneinfo import ZoneInfo
    TPE_TZ = ZoneInfo("Asia/Taipei")
except Exception:  # 老版本 Python 備援
    TPE_TZ = "Asia/Taipei"

# 若要更精準可填入台灣國定假日：{"2025-01-01", "2025-02-28", ...}
TW_HOLIDAYS: set[str] = set()

# 本模型用到的特徵欄位（訓練與推論要一致）
FEATS_TIMEONLY = [
    "tgt_hour", "tgt_dow",
    "tgt_is_weekend", "tgt_is_holiday",
    "tgt_is_peak", "tgt_is_night", "tgt_is_lunch",
    "tgt_hour_sin", "tgt_hour_cos",
    "tgt_dow_sin", "tgt_dow_cos",
    # 如需把站點/城市也做成特徵，可自行 one-hot / embedding；此處不強制
]

def _to_tpe(ts) -> pd.Timestamp:
    """把字串或 timestamp 轉成台北時間的 timezone-aware Timestamp。"""
    t = pd.to_datetime(ts)
    if isinstance(TPE_TZ, str):
        return (t.tz_localize(TPE_TZ) if t.tzinfo is None else t.tz_convert(TPE_TZ))
    else:
        return (t.tz_localize(TPE_TZ) if t.tzinfo is None else t.tz_convert(TPE_TZ))

def _row_time_features(ts_local: pd.Timestamp) -> dict:
    """給定(台北時間) timestamp，產生一列時間特徵。"""
    hour = int(ts_local.hour)
    dow  = int(ts_local.dayofweek)      # 0=Mon, 6=Sun
    is_weekend = int(dow in (5, 6))
    is_holiday = int(ts_local.date().isoformat() in TW_HOLIDAYS)

    # 你可以依需求調整時段旗標
    is_peak  = int(hour in (7, 8, 9, 17, 18, 19))
    is_night = int(hour >= 23 or hour < 6)
    is_lunch = int(hour in (12, 13))

    hr_rad  = 2 * np.pi * hour / 24.0
    dow_rad = 2 * np.pi * dow  / 7.0

    return {
        "tgt_hour": hour,
        "tgt_dow": dow,
        "tgt_is_weekend": is_weekend,
        "tgt_is_holiday": is_holiday,
        "tgt_is_peak": is_peak,
        "tgt_is_night": is_night,
        "tgt_is_lunch": is_lunch,
        "tgt_hour_sin": float(np.sin(hr_rad)),
        "tgt_hour_cos": float(np.cos(hr_rad)),
        "tgt_dow_sin": float(np.sin(dow_rad)),
        "tgt_dow_cos": float(np.cos(dow_rad)),
    }

# --------------------------------------------------------------------------- #
#  A) 推論用：只給 (city, sno, target_local_iso) → 回一列特徵 DataFrame
# --------------------------------------------------------------------------- #
def build_timeonly_features_for_target(
    *, city: str, sno: str | None, target_local_iso: str
) -> pd.DataFrame:
    """
    生成一筆『只靠時間』的特徵列，用於「絕對時間」推論。
    不讀任何即時/歷史狀態，因此可預測資料以後的日期。
    """
    ts = _to_tpe(target_local_iso)
    row = {
        "city": city,
        "sno": sno,
        "ts_local": ts,   # 方便除錯，模型不一定要用
        **_row_time_features(ts),
    }
    return pd.DataFrame([row])


# 便利函式：取訓練用的特徵欄位清單
def get_feature_columns_timeonly() -> list[str]:
    return FEATS_TIMEONLY[:]
