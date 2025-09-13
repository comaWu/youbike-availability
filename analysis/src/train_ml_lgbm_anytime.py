# analysis/src/train_ml_lgbm_anytime.py
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
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
import lightgbm as lgb

from .config import MODELS_DIR, get_pg_dsn, TRAIN_CITY, TRAIN_START, TRAIN_END
from .features import add_timeonly_features_for_training, get_feature_columns_timeonly


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train_classification")


MODEL_PATH   = MODELS_DIR / "model_anytime.pkl"
FEATS_PATH   = MODELS_DIR / "feature_columns_any.json"
PR_PNG       = MODELS_DIR / "pr_any.png"
ROC_PNG      = MODELS_DIR / "roc_any.png"
EVAL_SUMMARY = MODELS_DIR / "eval_any_summary.json"


def load_training_df() -> pd.DataFrame:
    """從 PostgreSQL 讀取訓練資料（分類用）。"""
    if not (TRAIN_START and TRAIN_END):
        raise ValueError("請在 .env 設定 TRAIN_START / TRAIN_END（例：2025-08-25 00:00）")

    dsn = get_pg_dsn()
    log.info(f"[STEP] 連線資料庫")
    engine = create_engine(dsn)

    log.info(f"[STEP] 執行 SQL 讀取資料 (city={TRAIN_CITY or 'ALL'}, "
             f"start={TRAIN_START}, end={TRAIN_END})")

    # 內建 SQL：station_minute + station → city, sno, ts_local, y
    sql = text("""
        SELECT
            sm.city,
            sm.sno,
            sm.ts AT TIME ZONE 'Asia/Taipei' AS ts_local,
            CASE WHEN sm.available > 0 THEN 1 ELSE 0 END AS y
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

    required = {"city", "sno", "ts_local", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DB 回傳缺少欄位：{missing}")

    # tz-aware
    log.info("[STEP] 時區處理 → Asia/Taipei (tz-aware)")
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
    df["y"] = df["y"].astype(int)

    log.info(df.head(3).to_string(index=False))
    return df


def train_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 42):
    log.info("[STEP] 開始訓練 LightGBM（二分類）")
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
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


def plot_pr(y_true, y_prob, title, out_png: Path):
    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title} AP={ap:.3f}")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return float(ap)


def plot_roc(y_true, y_prob, title, out_png: Path):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title} AUC={auc:.3f}")
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return float(auc)


def main():
    ap = argparse.ArgumentParser(description="Train classification (time-only) for absolute-time can-rent probability")
    ap.add_argument("--val_size", type=float, default=0.2, help="驗證集比例(0~1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("==== 分類訓練（純時間特徵 / 絕對時間預測）====")
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
    y = df["y"].astype(int).values
    log.info(f"[INFO] X shape={X.shape}, y positives={int(y.sum()):,}/{len(y):,}")

    # 3) 切 train/val
    log.info("[STEP] 切分訓練/驗證集")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y
    )
    log.info(f"[OK] train={len(X_tr):,}  valid={len(X_val):,}")

    # 4) 訓練
    model = train_lgbm(X_tr, y_tr, seed=args.seed)

    # 5) 評估
    log.info("[STEP] 驗證集評估")
    prob_val = model.predict(X_val)
    ap = average_precision_score(y_val, prob_val)
    auc = roc_auc_score(y_val, prob_val)
    log.info(f"[METRIC] AP={ap:.4f}, AUC={auc:.4f}")

    # 6) 繪圖
    log.info("[STEP] 產生 PR/ROC 圖")
    ap_plot  = plot_pr(y_val, prob_val, "Anytime Model (Time-only)", PR_PNG)
    auc_plot = plot_roc(y_val, prob_val, "Anytime Model (Time-only)", ROC_PNG)
    log.info(f"[OK] 圖片輸出：{PR_PNG.name}, {ROC_PNG.name}")

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
        "ap_valid": float(ap),
        "auc_valid": float(auc),
        "pr_curve_png": str(PR_PNG),
        "roc_curve_png": str(ROC_PNG),
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
