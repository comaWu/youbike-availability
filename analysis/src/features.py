import pandas as pd
import numpy as np
import psycopg
from .config import PG_DSN, HORIZON_MIN, LOOKBACK_MIN

def load_series(city: str, sno: str | None, days: int = 14) -> pd.DataFrame:
    """讀取最近 N 天資料；若 sno=None 則回多站資料（含 city, sno）"""
    q = f"""
      SELECT sm.ts, sm.available, sm.empty, s.tot, sm.city, sm.sno
      FROM station_minute sm
      JOIN station s ON s.city=sm.city AND s.sno=sm.sno
      WHERE sm.ts >= now() - interval '{days} days'
        AND sm.city = %s
        { "AND sm.sno = %s" if sno else "" }
      ORDER BY sm.ts
    """
    with psycopg.connect(PG_DSN) as conn:
        df = pd.read_sql(q, conn, params=[city] + ([sno] if sno else []))
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 基本防守：available/empty 不應為負或超過 tot；不合邏輯的直接丟
    df = df[(df["available"] >= 0) & (df["empty"] >= 0)]
    df = df[(df["available"] <= df["tot"]) & (df["empty"] <= df["tot"])]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_local"] = df["ts"].dt.tz_convert("Asia/Taipei")
    df["hour"] = df["ts_local"].dt.hour
    df["dow"] = df["ts_local"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def make_lag_features(df: pd.DataFrame, lags=(1,2), ma_windows=(2,3)) -> pd.DataFrame:
    df = df.sort_values(["city","sno","ts"]).copy()
    def _per_station(g):
        for k in lags:
            g[f"lag_{k}"] = g["available"].shift(k)
        for w in ma_windows:
            g[f"ma_{w}"] = g["available"].rolling(w, min_periods=1).mean()
        return g
    return df.groupby(["city","sno"], group_keys=False).apply(_per_station)

def attach_target(df: pd.DataFrame, horizon: int = HORIZON_MIN) -> pd.DataFrame:
    df = df.sort_values(["city","sno","ts"]).copy()
    def _per_station(g):
        g["y"] = g["available"].shift(-horizon)
        # 若 tot 偶爾缺，避免 clip 產生 NaN
        up = g["tot"].fillna(g["tot"].median())
        g["y"] = g["y"].clip(lower=0, upper=up)
        return g
    return df.groupby(["city","sno"], group_keys=False).apply(_per_station)

def build_features(city: str, sno: str | None, days=14, horizon=HORIZON_MIN) -> pd.DataFrame:
    raw = load_series(city, sno, days=days)
    if raw.empty:
        raise RuntimeError(f"資料為空：city={city}, sno={sno}，請確認此站在最近 {days} 天是否有資料")

    raw = clean_df(raw)
    if raw.empty:
        raise RuntimeError(f"清洗後無資料：city={city}, sno={sno}；可能資料異常被過濾，請換站或縮短 days")

    # 單站：用 index + 完整分鐘補齊（避免 lag 斷裂）
    if sno:
        min_ts = pd.to_datetime(raw["ts"]).min()
        max_ts = pd.to_datetime(raw["ts"]).max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            raise RuntimeError(f"此站缺少有效時間戳：city={city}, sno={sno}")
        idx = pd.date_range(min_ts, max_ts, freq="1min", tz="UTC")
        raw = (raw.set_index("ts")
                  .reindex(idx)
                  .rename_axis("ts").reset_index())
        # 補回靜態欄位
        for col, val in [("city", city), ("sno", sno), ("tot", raw["tot"].dropna().median())]:
            raw[col] = raw[col].fillna(val)
        raw["available"] = raw["available"]  # 保持 NaN 以便後續 dropna
    # 特徵
    feat = make_time_features(raw)
    feat = make_lag_features(feat)
    feat = attach_target(feat, horizon=horizon)
    # 丟掉 lag/目標需要的起末端缺值
    need_cols = ["available"] + [c for c in feat.columns if c.startswith("lag_")]
    feat = feat.dropna(subset=need_cols + ["y"])
    if feat.empty:
        raise RuntimeError("特徵集為空：lag/目標對齊後沒有可用樣本，請換站或延長觀測天數")
    return feat