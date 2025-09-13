# analysis/src/predict_anytime.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd
import numpy as np

from .config import MODELS_DIR, ROOT
from .features import (
    build_timeonly_features_for_target,
    get_feature_columns_timeonly,
)

# ---- 檔案位置 ----
CLS_MODEL_PATH   = MODELS_DIR / "model_anytime.pkl"
CLS_FEATS_PATH   = MODELS_DIR / "feature_columns_any.json"
REG_MODEL_PATH   = MODELS_DIR / "model_anytime_reg.pkl"
REG_FEATS_PATH   = MODELS_DIR / "feature_columns_any_reg.json"

STATIONS_JSON = ROOT / "analysis" / "src" / "stations.json"

# ---- 站點索引 ----
_st_idx: dict[str, dict[str, dict]] | None = None
def _ensure_station_index():
    global _st_idx
    if _st_idx is not None:
        return
    _st_idx = {}
    if STATIONS_JSON.exists():
        try:
            arr = json.loads(STATIONS_JSON.read_text(encoding="utf-8"))
            for it in arr:
                city = it.get("city") or it.get("sarea") or "UNK"
                sno = str(it.get("sno"))
                _st_idx.setdefault(city, {})[sno] = it
        except Exception:
            _st_idx = {}

def _get_station(city: str, sno: str) -> dict | None:
    _ensure_station_index()
    return _st_idx.get(city, {}).get(str(sno)) if _st_idx else None

# ---- 小工具 ----
def _load_json(path: Path, fallback: Optional[list] = None):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return fallback if fallback is not None else []

def _align_features(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    return X[feats]

def _safe_round_nonneg(x: float) -> int:
    try:
        return int(max(0, round(float(x))))
    except Exception:
        return 0

# ---- 載入模型/特徵 ----
def _load_cls():
    return joblib.load(CLS_MODEL_PATH) if CLS_MODEL_PATH.exists() else None

def _load_reg():
    return joblib.load(REG_MODEL_PATH) if REG_MODEL_PATH.exists() else None

def _load_cls_feats():
    return _load_json(CLS_FEATS_PATH, get_feature_columns_timeonly())

def _load_reg_feats():
    return _load_json(REG_FEATS_PATH, get_feature_columns_timeonly())

# ---- 主功能 ----
def predict_one_anytime(
    *,
    city: str,
    sno: str,
    target_local_iso: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    同時回傳：
      - proba_can_rent（分類機率；若沒模型則 None）
      - pred_available（預估可借數量；若沒模型則 None）
    會自動從 stations.json 注入訓練時用到的站點特徵（tot/lat/lng）。
    """
    # 1) 時間特徵
    base = build_timeonly_features_for_target(
        city=city, sno=sno, target_local_iso=target_local_iso
    )

    # 2) 站點特徵：tot/lat/lng
    st = _get_station(city, sno) or {}
    base = base.copy()
    base["tot"] = float(st.get("tot") or 0)
    base["lat"] = float(st.get("lat") or 0)
    base["lng"] = float(st.get("lng") or 0)

    res: Dict[str, Any] = {
        "ok": True,
        "city": city,
        "sno": sno,
        "target_local": str(base.loc[0, "ts_local"]),
        "proba_can_rent": None,
        "pred_available": None,
        "decision": None,
        "msg": None,
    }

    # 3) 分類
    cls = _load_cls()
    if cls is not None:
        feats_c = _load_cls_feats()
        Xc = _align_features(base, feats_c)
        try:
            proba = float(cls.predict(Xc)[0])
            res["proba_can_rent"] = proba
            if threshold is not None:
                res["decision"] = bool(proba >= float(threshold))
        except Exception as e:
            res["ok"] = False
            res["msg"] = f"classification predict failed: {e}"

    # 4) 迴歸
    reg = _load_reg()
    if reg is not None:
        feats_r = _load_reg_feats()
        Xr = _align_features(base, feats_r)
        try:
            yhat = float(reg.predict(Xr)[0])
            res["pred_available"] = _safe_round_nonneg(yhat)
        except Exception as e:
            res["ok"] = False
            res["msg"] = f"regression predict failed: {e}"

    if cls is None and reg is None:
        res["ok"] = False
        res["msg"] = (f"no model found under {MODELS_DIR} "
                      f"({CLS_MODEL_PATH.name} / {REG_MODEL_PATH.name})")

    return res

# ---- CLI ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict probability and available count at absolute time (with station features)")
    ap.add_argument("--city", default="TPE")
    ap.add_argument("--sno", required=True)
    ap.add_argument("--target", required=True, help="YYYY-MM-DD HH:MM")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    out = predict_one_anytime(city=args.city, sno=args.sno, target_local_iso=args.target, threshold=args.threshold)
    print(json.dumps(out, ensure_ascii=False, indent=2))
