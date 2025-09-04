# analysis/src/predict_once.py
from pathlib import Path
import json
import joblib
import pandas as pd
from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"

def load_latest_model_and_features():
    """優先載入 LGBM；沒有就載 GBR；同時讀取訓練時保存的特徵欄位。"""
    model_path = OUT_DIR / "model_lgbm.pkl"
    if not model_path.exists():
        model_path = OUT_DIR / "model_gbr.pkl"
    model = joblib.load(model_path)

    feats_path = OUT_DIR / "feature_columns.json"
    feats = json.loads(feats_path.read_text())
    return model, feats

def predict_once(city=DEFAULT_CITY, sno=DEFAULT_SNO, horizon=HORIZON_MIN, days=14):
    """
    讀 DB 建特徵 → 載模型 → 取最後一筆做單點預測
    回傳：
    {
      city, sno, horizon_min, now_ts, tot, available_now,
      pred_available, prob_rentable
    }
    """
    # 1) 構建與訓練一致的特徵
    df = build_features(city, sno, days=days, horizon=horizon)

    # 沒資料就直接回報
    if df.empty:
        return {
            "city": city, "sno": sno, "horizon_min": int(horizon),
            "error": "no data to predict"
        }

    # 2) 載模型與特徵欄位
    model, feats = load_latest_model_and_features()

    # 3) 對齊欄位；LGBM 需要類別特徵為 category dtype
    X = df[feats].copy()
    for c in ("city", "sno"):
        if c in X.columns:
            X[c] = X[c].astype("category")

    # 4) 取最後一筆做預測（對應最新時間點）
    x_last = X.iloc[[-1]]
    yhat = float(model.predict(x_last)[0])

    row = df.iloc[-1]
    tot = int(row.get("tot", 0))
    now_avail = float(row.get("available", 0.0))

    # 一個簡單的機率近似：預估可借 / 車樁總數（可再換成二元分類模型）
    prob = max(0.0, min(1.0, yhat / max(1, tot)))

    return {
        "city": str(row.get("city", city)),
        "sno": str(row.get("sno", sno)),
        "horizon_min": int(horizon),
        "now_ts": str(row.get("ts")),
        "tot": tot,
        "available_now": now_avail,
        "pred_available": round(yhat, 3),
        "prob_rentable": round(prob, 3),
    }

def main():
    res = predict_once()
    print(res)

if __name__ == "__main__":
    main()
