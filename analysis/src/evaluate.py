from pathlib import Path
import json, joblib
from sklearn.metrics import mean_absolute_error, r2_score
from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    df = build_features(city, sno, days=14, horizon=HORIZON_MIN)
    feats = json.loads((OUT_DIR/"feature_columns.json").read_text())
    model = joblib.load(OUT_DIR/"model_gbr.pkl")

    X, y = df[feats], df["y"]
    split = int(len(df)*0.8)
    X_te, y_te = X.iloc[split:], y.iloc[split:]
    yhat = model.predict(X_te)
    print("MAE:", mean_absolute_error(y_te, yhat))
    print("R2 :", r2_score(y_te, yhat))

if __name__ == "__main__":
    run()
