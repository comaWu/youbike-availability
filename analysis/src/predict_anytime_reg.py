# analysis/src/predict_anytime_reg.py
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .features import build_timeonly_features_for_target, get_feature_columns_timeonly

MODEL_PATH = MODELS_DIR / "model_anytime_reg.pkl"
FEATS_PATH = MODELS_DIR / "feature_columns_any_reg.json"

def _load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def _load_feature_list():
    if FEATS_PATH.exists():
        try:
            return json.loads(FEATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return get_feature_columns_timeonly()

def _align_features(X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    return X[feats]

def predict_available_at_time(
    *, city: str, sno: str, target_local_iso: str, clip_min: int = 0
) -> dict:
    """
    用迴歸模型預測『絕對時間』的可借車輛數。
    回傳 pred_available（四捨五入為非負整數；不做上限剪裁）。
    """
    model = _load_model()
    if model is None:
        return {"ok": False, "msg": f"regression model not found: {MODEL_PATH}"}

    feats = _load_feature_list()

    X = build_timeonly_features_for_target(
        city=city, sno=sno, target_local_iso=target_local_iso
    )
    X_model = _align_features(X, feats)

    try:
        y_hat = float(model.predict(X_model)[0])
    except Exception as e:
        return {"ok": False, "msg": f"predict failed: {e}"}

    # 後處理：不為負 & 取整數
    y_hat = max(clip_min, round(y_hat))

    return {
        "ok": True,
        "city": city,
        "sno": sno,
        "target_local": str(X.loc[0, "ts_local"]),
        "pred_available": int(y_hat),
        "used_features": feats,
    }

# 命令列測試
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="TPE")
    ap.add_argument("--sno", required=True)
    ap.add_argument("--target", required=True, help="YYYY-MM-DD HH:MM (Asia/Taipei)")
    args = ap.parse_args()
    res = predict_available_at_time(city=args.city, sno=args.sno, target_local_iso=args.target)
    print(json.dumps(res, ensure_ascii=False, indent=2))
