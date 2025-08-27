import json
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    df = build_features(city, sno, days=30, horizon=HORIZON_MIN)
    feats = ["hour","dow","is_weekend","tot"] + \
            [c for c in df.columns if c.startswith("lag_") or c.startswith("ma_")]
    X, y = df[feats], df["y"]
    split = int(len(df)*0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    mae = mean_absolute_error(y_te, yhat)

    # 保存
    joblib.dump(model, OUT_DIR/"model_gbr.pkl")
    (OUT_DIR/"feature_columns.json").write_text(json.dumps(feats, ensure_ascii=False, indent=2))
    meta = {
        "model":"GradientBoostingRegressor",
        "mae_test": float(mae),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "horizon_min": HORIZON_MIN,
        "city": city, "sno": sno
    }
    (OUT_DIR/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("[TRAIN] MAE:", mae, "saved to:", OUT_DIR)

if __name__ == "__main__":
    run()
