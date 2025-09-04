# analysis/src/features.py
# -*- coding: utf-8 -*-
"""
YouBike 特徵工程（單站 / 多站皆可）
- 從 DB 撈取分鐘資料
- 逐站補齊時間軸 (1min)
- 時間特徵、滯後特徵、移動平均
- 目標 y = 未來 horizon 分鐘後的 available
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import psycopg
from typing import Iterable, List, Optional, Tuple
from .config import PG_DSN, HORIZON_MIN, LOOKBACK_MIN, DEFAULT_CITY, DEFAULT_SNO

# === 可調參數 ===
DEFAULT_LAGS = (1, 2, 3, 5, 10)      # 滯後特徵
DEFAULT_MA   = (3, 5, 10)            # 移動平均視窗
USE_MA       = True                  # 是否啟用移動平均特徵

# ---------------------------------------------------------------------
# DB 載入（請確保 station_minute 欄位至少有：ts, available, empty, tot, city, sno）
# ---------------------------------------------------------------------
def load_series(city: str, sno: Optional[str], days: int = 14) -> pd.DataFrame:
    interval_expr = f"{days} days"
    q = f"""
    SELECT sm.ts, sm.available, sm.empty, s.tot, sm.city, sm.sno
    FROM station_minute sm
    JOIN station s ON s.city=sm.city AND s.sno=sm.sno
    WHERE sm.ts >= now() - interval '{interval_expr}'
    AND sm.city = %s
    """
    params = [city]
    if sno:
        q += " AND sm.sno = %s"
        params.append(sno)
    q += " ORDER BY sm.city, sm.sno, sm.ts"

    with psycopg.connect(PG_DSN) as conn:
        df = pd.read_sql(q, conn, params=params)

    # datetime 處理
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    else:
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        else:
            df["ts"] = df["ts"].dt.tz_convert("UTC")

    return df


# ---------------------------------------------------------------------
# 逐站重採樣（解決多站時 ts 重覆 reindex 會報錯）
# ---------------------------------------------------------------------
def resample_per_station(raw: pd.DataFrame) -> pd.DataFrame:
    """
    以 (city, sno) 分組，逐站補齊分鐘索引並做基本缺值處理。
    """
    raw = raw.sort_values(["city", "sno", "ts"])

    def _resample_one(g: pd.DataFrame) -> pd.DataFrame:
        # 同站若有重覆時間戳，保留最後一筆
        g = g.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
        idx = pd.date_range(g["ts"].min(), g["ts"].max(), freq="1min", tz="UTC")
        out = (
            g.set_index("ts")
             .reindex(idx)              # 僅針對此站自己的時間軸
             .rename_axis("ts")
             .reset_index()
        )
        # 補欄位
        out["city"] = out["city"].ffill().bfill()
        out["sno"]  = out["sno"].ffill().bfill()
        out["tot"]  = out["tot"].ffill()
        out["available"] = out["available"].fillna(0)
        out["empty"]     = out["empty"].fillna(0)
        return out

    out = (
        raw.groupby(["city", "sno"], group_keys=False)
           .apply(_resample_one)
           .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------
# 特徵函式
# ---------------------------------------------------------------------
def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    產生時間特徵（皆以 UTC 為基準；若你要轉台灣時間，可在這裡加 .dt.tz_convert('Asia/Taipei')）
    """
    out = df.copy()
    out["ts_local"] = out["ts"]  # 如需本地時間可改 tz
    out["hour"] = out["ts_local"].dt.hour.astype(np.int16)
    out["dow"]  = out["ts_local"].dt.weekday.astype(np.int8)  # Mon=0
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)
    return out


def make_lag_features(df: pd.DataFrame, lags: Iterable[int] = DEFAULT_LAGS) -> pd.DataFrame:
    """
    對 available 產生滯後特徵 lag_k（每個站獨立計算）
    """
    out = df.copy()
    out = out.sort_values(["city", "sno", "ts"])
    for k in lags:
        out[f"lag_{k}"] = out.groupby(["city", "sno"])["available"].shift(k)
    return out


def make_ma_features(df: pd.DataFrame, windows: Iterable[int] = DEFAULT_MA) -> pd.DataFrame:
    """
    對 available 產生移動平均 ma_w（每個站獨立計算）
    """
    out = df.copy()
    out = out.sort_values(["city", "sno", "ts"])
    for w in windows:
        out[f"ma_{w}"] = (
            out.groupby(["city", "sno"])["available"]
               .rolling(window=w, min_periods=max(1, w // 2))
               .mean()
               .reset_index(level=[0,1], drop=True)
        )
    return out


def attach_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    y = 未來 horizon 分鐘後的 available（每站獨立對齊）
    """
    out = df.copy()
    out = out.sort_values(["city", "sno", "ts"])
    out["y"] = out.groupby(["city", "sno"])["available"].shift(-horizon)
    return out


# ---------------------------------------------------------------------
# 主流程：build_features
# ---------------------------------------------------------------------
def build_features(
    city: str = DEFAULT_CITY,
    sno: Optional[str] = DEFAULT_SNO,
    days: int = 14,
    horizon: int = HORIZON_MIN,
    lags: Iterable[int] = DEFAULT_LAGS,
    ma_windows: Iterable[int] = DEFAULT_MA,
) -> pd.DataFrame:
    """
    建構訓練用特徵表：
    - 逐站重採樣 → 時間特徵 → lag → (optional) 移動平均 → y
    - 類別特徵：city/sno 以字串形式輸出（LightGBM 會吃 category）
    """
    print(f"=== 開始建構特徵 city={city}, sno={sno}, days={days}, horizon={horizon} ===")

    # 1) 撈資料
    raw = load_series(city, sno, days=days)
    print(f"[RAW] shape={raw.shape}")
    if raw.empty:
        return raw

    # 2) 逐站重採樣（解決多站重覆 ts 問題）
    raw = resample_per_station(raw)
    print(f"[RESAMPLED] shape={raw.shape}")

    # 3) 時間特徵
    feat = make_time_features(raw)
    # 4) 滯後特徵
    feat = make_lag_features(feat, lags=lags)
    # 5) 移動平均（可關閉）
    if USE_MA and ma_windows:
        feat = make_ma_features(feat, windows=ma_windows)

    # 6) 目標 y
    feat = attach_target(feat, horizon=horizon)

    # 類別特徵（轉字串，推論時會轉 category）
    feat["city"] = feat["city"].astype(str)
    feat["sno"]  = feat["sno"].astype(str)

    # 7) 清理不可用樣本：至少要有 available / lag_x / ma_x（若開啟）與 y
    need_cols = ["available"] + [c for c in feat.columns if c.startswith("lag_")]
    if USE_MA:
        need_cols += [c for c in feat.columns if c.startswith("ma_")]

    before = len(feat)
    feat = feat.dropna(subset=need_cols + ["y"])
    after = len(feat)
    print(f"[FINAL] shape={feat.shape}  dropped={before - after}")

    return feat


# ---------------------------------------------------------------------
# 便利 CLI：直接在終端測試
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = build_features()
    print(df.head())
