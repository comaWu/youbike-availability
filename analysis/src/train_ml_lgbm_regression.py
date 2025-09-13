# analysis/src/train_ml_lgbm_regression.py
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from .config import MODELS_DIR, get_pg_dsn, TRAIN_CITY, TRAIN_START, TRAIN_END
from .features import add_timeonly_features_for_training, get_feature_columns_timeonly


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train_regression")


MODEL_PATH   = MODELS_DIR / "model_anytime_reg.pkl"
FEATS_PATH   = MODELS_DIR / "feature_columns_any_reg.json"
SCATTER_PNG  = MODELS_DIR / "reg_scatter.png"
DIST_PNG     = MODELS_DIR / "reg_dist.png"
EVAL_SUMMARY = MODELS_DIR / "eval_any_reg_summary.json"


def load_training_df() -> pd.DataFrame:
    """從 PostgreSQL 讀取訓練資料（迴歸用）。"""
    if not (TRAIN_START and TRAIN_END):
        raise ValueError("請在 .env 設定 TRAIN_START / TRAIN_END（例：2025-08-25 00:00）")

    dsn = get_pg_dsn()
    log.info(f"[STEP] 連線資料庫")
    engine = create_engine(dsn)

    log.info(f"[STEP] 執行 SQL 讀取資料 (city={TRAIN_CITY or 'ALL'}, "
             f"start={TRAIN_START}, end={TRAIN_END})")

    # 內建 SQL：station_minute + station → city, sno, ts_local, available
    sql = text("""
        SELECT
            sm.city,
            sm.sno,
            sm.ts AT TIME ZONE 'Asia/Taipei' AS ts_local,
            sm.available
        FROM station_minute sm
        JOIN station s ON s.city = sm.city AND s.sno = sm.sno
        WHERE sm.ts >= CAST(:start_ts AS timestamptz)
          AND sm.ts <  CAST(:end_ts   AS timestamptz)
          AND sm.is_active = TRUE
          AND sm.city = COALESCE(CAST(:city AS text), sm.city)
        ORDER BY sm.ts, sm.city, sm.sno
    """)

    params = {
        "city": TRAIN_CITY if TRAIN_CITY else None,
        "start_ts": TRAIN_START,
        "end_ts": TRAIN_END,
    }

    t0 = time.time()
    df = pd.read_sql(sql, engine, params=params)
    log.info(f"[OK] 讀到 {len(df):,} 筆資料，用時 {time.time()-t0:.2f}s")

    required = {"city", "sno", "ts_local", "available"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DB 回傳缺少欄位：{missing}")

    # tz-aware
    log.info("[STEP] 時間時區處理 → Asia/Taipei (tz-aware)")
    ts = pd.to_datetime(df["ts_local"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"ts_local 有 {int(ts.isna().sum())} 筆無法轉時間")
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Taipei")
        ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    except Exception:
        ts = ts.dt.tz_localize("Asia/Taipei") if ts.dt.tz is None else ts.dt.tz_convert("Asia/Taipei")

    df = df.copy()
    df["ts_local"] = ts
    df["available"] = pd.to_numeric(df["available"], errors="coerce").fillna(0).astype(int)

    log.info(df.head(3).to_string(index=False))
    return df


def train_lgbm_reg(X: np.ndarray, y: np.ndarray, seed: int = 42):
    log.info("[STEP] 開始訓練 LightGBM（迴歸）")
    params = {
        "objective": "regression",
        "metric": ["l2", "l1"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "random_state": seed,
        "verbosity": -1,
    }
    dtrain = lgb.Dataset(X, label=y)
    t0 = time.time()
    model = lgb.train(params, dtrain, num_boost_round=500)
    log.info(f"[OK] 訓練完成（{time.time()-t0:.2f}s, trees=500）")
    return model


def plot_scatter(y_true, y_pred, out_png: Path, title="Pred vs True"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    m = max(float(np.max(y_true)), float(np.max(y_pred))) if len(y_true) else 1.0
    plt.plot([0, m], [0, m])
    plt.xlabel("True available"); plt.ylabel("Pred available")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


def plot_dist(y_true, y_pred, out_png: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    plt.hist(y_true, bins=30, alpha=0.6, label="True")
    plt.hist(y_pred, bins=30, alpha=0.6, label="Pred")
    plt.legend()
    plt.xlabel("available"); plt.ylabel("count"); plt.title("Distribution")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Train regression (time-only) to predict available count at absolute time")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("==== 迴歸訓練（純時間特徵 / 絕對時間預測）====")
    log.info(f"輸出資料夾：{MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 讀 DB
    df_raw = load_training_df()

    # 2) 時間特徵
    log.info("[STEP] 建立時間特徵欄位")
    df = add_timeonly_features_for_training(df_raw)
    feats = get_feature_columns_timeonly()
    log.info(f"[OK] 特徵欄位：{feats}")

    X = df[feats].values
    y = df["available"].astype(float).values
    log.info(f"[INFO] X shape={X.shape}, y range=({np.min(y):.1f}, {np.max(y):.1f})")

    # 3) 切 train/val
    log.info("[STEP] 切分訓練/驗證集")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.seed)
    log.info(f"[OK] train={len(X_tr):,}  valid={len(X_val):,}")

    # 4) 訓練
    model = train_lgbm_reg(X_tr, y_tr, seed=args.seed)

    # 5) 評估
    log.info("[STEP] 驗證集評估")
    pred_val = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    r2 = float(r2_score(y_val, pred_val))
    log.info(f"[METRIC] MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")

    # 6) 繪圖
    log.info("[STEP] 產生散點/分佈圖")
    plot_scatter(y_val, pred_val, SCATTER_PNG, "Available: Pred vs True")
    plot_dist(y_val, pred_val, DIST_PNG)
    log.info(f"[OK] 圖片輸出：{SCATTER_PNG.name}, {DIST_PNG.name}")

    # 7) 輸出
    log.info("[STEP] 存模型與特徵欄位")
    joblib.dump(model, MODEL_PATH)
    FEATS_PATH.write_text(json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[OK] 模型：{MODEL_PATH.name}，特徵欄位：{FEATS_PATH.name}")

    # 8) 總結
    summary = {
        "samples": int(len(df)),
        "train_samples": int(len(X_tr)),
        "valid_samples": int(len(X_val)),
        "features": feats,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "scatter_png": str(SCATTER_PNG),
        "dist_png": str(DIST_PNG),
        "model_path": str(MODEL_PATH),
        "feature_list_path": str(FEATS_PATH),
        "train_city": TRAIN_CITY,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
    }
    EVAL_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[DONE] 訓練完成，摘要：{EVAL_SUMMARY}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
