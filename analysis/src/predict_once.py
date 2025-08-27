import json, joblib
import numpy as np
from pathlib import Path
from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"

def prob_rentable(pred_available, tot):
    # 簡單轉換：預估剩餘量 / 總車位，上限 1
    return max(0.0, min(1.0, float(pred_available) / max(1, float(tot))))

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    df = build_features(city, sno, days=2, horizon=HORIZON_MIN)
    feats = json.loads((OUT_DIR/"feature_columns.json").read_text())
    model = joblib.load(OUT_DIR/"model_gbr.pkl")

    # 取最後一筆特徵推論
    row = df.iloc[-1]
    x = row[feats].values.reshape(1, -1)
    yhat = model.predict(x)[0]
    yhat = np.clip(yhat, 0, row["tot"])
    p = prob_rentable(yhat, row["tot"])
    print({
        "city": city, "sno": row["sno"],
        "horizon_min": HORIZON_MIN,
        "pred_available": float(yhat),
        "prob_rentable": float(p),
        "now_ts": str(row["ts"]),
    })

if __name__ == "__main__":
    run()
