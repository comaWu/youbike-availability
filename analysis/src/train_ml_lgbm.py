# analysis/src/train_ml_lgbm.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from sklearn.metrics import mean_absolute_error, r2_score

from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO, days=7, horizon=HORIZON_MIN):
    # 1) 構建特徵
    df = build_features(city, sno, days=days, horizon=horizon)
    if df.empty:
        raise RuntimeError("特徵集為空，請增加 days 或降低 horizon")

    # 2) 拆 X/y；指定類別特徵
    cat_feats = ["city", "sno"]
    y = df["y"].astype(float)

    # 找出所有 datetime 欄位（例如 ts, ts_local）
    datetime_cols = [c for c in df.columns
                    if str(df[c].dtype).startswith("datetime64")]

    # 建立 X：拿掉 y + 所有 datetime 欄位
    X = df.drop(columns=["y"] + datetime_cols).copy()

    # 將布林轉成小整數，避免某些版本出現型別不符
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype("int8")

    # 類別特徵轉 category（保證存在才轉）
    for c in cat_feats:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # 3) 時序切分（最後 20% 為 valid，用於 early stopping）
    split = int(len(df) * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]
    if len(X_va) == 0:
        raise RuntimeError("資料太短，無法切出驗證集；請增加 days")

    # 4) 建立 LightGBM + Early Stopping
    #    將 n_estimators 設大一點，交給 early_stopping_rounds 主動停止
    model = LGBMRegressor(
        objective="regression",
        n_estimators=2000,      # 大一點，讓 early stopping 來決定最佳 iter
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.9,          # bagging_fraction
        subsample_freq=1,
        colsample_bytree=0.9,   # feature_fraction
        random_state=42,
        n_jobs=-1,
    )

    # 5) 訓練（無 verbose 參數；用 callbacks 控制輸出與 early stopping）
    callbacks = [
        log_evaluation(50),                 # 每 50 次列印一次 valid 指標；想更安靜可拿掉
        early_stopping(stopping_rounds=100) # 100 次沒有進步就停止
    ]
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",          # 用 MAE 當監控指標
        categorical_feature=[c for c in cat_feats if c in X.columns],
        callbacks=callbacks
    )

    # 6) 用 best_iteration_ 做預測與評估
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None:
        yhat_tr = model.predict(X_tr, num_iteration=best_iter)
        yhat_va = model.predict(X_va, num_iteration=best_iter)
    else:
        yhat_tr = model.predict(X_tr)
        yhat_va = model.predict(X_va)

    mae_tr = mean_absolute_error(y_tr, yhat_tr)
    r2_tr  = r2_score(y_tr, yhat_tr)
    mae_va = mean_absolute_error(y_va, yhat_va)
    r2_va  = r2_score(y_va, yhat_va)

    print(f"[LGBM TRAIN] best_iter={best_iter}  "
          f"Train MAE={mae_tr:.4f} R2={r2_tr:.4f}  "
          f"Valid MAE={mae_va:.4f} R2={r2_va:.4f}  (horizon={horizon}m)")

    # 7) 特徵重要性（存檔給報表用）
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # 8) 存檔：模型、特徵欄位、meta
    joblib.dump(model, OUT_DIR / "model_lgbm.pkl")
    (OUT_DIR / "feature_columns.json").write_text(
        json.dumps(list(X.columns), ensure_ascii=False, indent=2)
    )
    meta = {
        "model": "LightGBM",
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "horizon_min": int(horizon),
        "city": city, "sno": sno,
        "n_train": int(len(X_tr)), "n_valid": int(len(X_va)),
        "mae_train": float(mae_tr), "r2_train": float(r2_tr),
        "mae_valid": float(mae_va), "r2_valid": float(r2_va),
        "params": {
            "n_estimators": 2000,
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
    fi.to_csv(OUT_DIR / "feature_importances.csv", index=False)

    print("saved to:", OUT_DIR)

if __name__ == "__main__":
    run()
