# analysis/src/train_ml_lgbm.py
import json
from pathlib import Path
import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    # 視資料量自訂：先用 3~7 天；資料更大可拉到 30 天
    df = build_features(city, sno, days=7, horizon=HORIZON_MIN)

    # --- 特徵欄位設定 ---
    # 數值型特徵
    num_feats = ["hour", "dow", "is_weekend", "tot"] + \
                [c for c in df.columns if c.startswith("lag_") or c.startswith("ma_")]
    # 類別特徵：讓 LGBM 原生處理（不用 one-hot）
    cat_feats = ["city", "sno"]

    # X / y
    X = df[num_feats + cat_feats].copy()
    y = df["y"].astype(float)

    # 轉成 pandas 的 category dtype，讓 LGBM 吃到類別資訊
    for c in cat_feats:
        X[c] = X[c].astype("category")

    # 時序切分（最後 20% 當 test）
    split = int(len(df) * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    # --- LightGBM 模型 ---
    # 這組參數偏保守（避免過擬合）；樣本更多時可適度調大 num_leaves / n_estimators
    model = LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.9,          # bagging_fraction
        subsample_freq=1,
        colsample_bytree=0.9,   # feature_fraction
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr, y_tr,
        categorical_feature=cat_feats,
        eval_set=[(X_te, y_te)],
        eval_metric="l1"       # MAE

    )

    yhat = model.predict(X_te)
    mae = mean_absolute_error(y_te, yhat)
    r2 = r2_score(y_te, yhat)

    # 保存模型與特徵欄位
    joblib.dump(model, OUT_DIR / "model_lgbm.pkl")
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(num_feats + cat_feats, ensure_ascii=False, indent=2))
    meta = {
        "model": "LightGBM",
        "mae_test": float(mae),
        "r2_test": float(r2),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "horizon_min": int(HORIZON_MIN),
        "city": city, "sno": sno,
        "cat_features": cat_feats,
        "params": {
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "random_state": 42
        }
    }
    (OUT_DIR / "meta_lgbm.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[LGBM TRAIN] MAE: {mae:.4f}  R2: {r2:.4f}  (horizon={HORIZON_MIN}m)")
    print("saved to:", OUT_DIR)

if __name__ == "__main__":
    run()
