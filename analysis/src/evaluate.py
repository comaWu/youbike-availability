from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"

def load_latest_model_and_features():
    # 優先用 LGBM；沒有就用 GBR
    if (OUT_DIR / "model_lgbm.pkl").exists():
        model_path = OUT_DIR / "model_lgbm.pkl"
    else:
        model_path = OUT_DIR / "model_gbr.pkl"

    model = joblib.load(model_path)

    # 特徵欄位（訓練時已寫入）
    feats_path = OUT_DIR / "feature_columns.json"
    feats = json.loads(feats_path.read_text())
    return model, feats

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    # 與訓練一致的樣本切分與特徵建構
    df = build_features(city, sno, days=14, horizon=HORIZON_MIN)

    model, feats = load_latest_model_and_features()

    # 只取訓練用到的欄位；若含 city/sno，轉成 category 以利 LGBM
    X = df[feats].copy()
    y = df["y"]

    for c in ("city", "sno"):
        if c in X.columns:
            X[c] = X[c].astype("category")

    split = int(len(df) * 0.8)
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    yhat = model.predict(X_te)
    print("MAE:", mean_absolute_error(y_te, yhat))
    print("R2 :", r2_score(y_te, yhat))

if __name__ == "__main__":
    run()
