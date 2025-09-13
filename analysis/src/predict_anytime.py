# analysis/src/predict_anytime.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd
import numpy as np

from .config import MODELS_DIR
from .features import (
    build_timeonly_features_for_target,
    get_feature_columns_timeonly,
)

# ---- 檔案位置（與訓練腳本一致）----
CLS_MODEL_PATH   = MODELS_DIR / "model_anytime.pkl"
CLS_FEATS_PATH   = MODELS_DIR / "feature_columns_any.json"

REG_MODEL_PATH   = MODELS_DIR / "model_anytime_reg.pkl"
REG_FEATS_PATH   = MODELS_DIR / "feature_columns_any_reg.json"


# ----------------- 小工具 -----------------
def _load_json(path: Path, fallback: Optional[list] = None):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return fallback if fallback is not None else []


def _align_features(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """將資料欄位對齊到訓練時的特徵欄位（缺的補 0，多的忽略）。"""
    X = df.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = 0
    # 嚴格照訓練順序
    return X[feats]


def _safe_round_nonneg(x: float) -> int:
    try:
        return int(max(0, round(float(x))))
    except Exception:
        return 0


# ----------------- 載入模型 -----------------
def _load_cls():
    return joblib.load(CLS_MODEL_PATH) if CLS_MODEL_PATH.exists() else None


def _load_reg():
    return joblib.load(REG_MODEL_PATH) if REG_MODEL_PATH.exists() else None


def _load_cls_feats():
    return _load_json(CLS_FEATS_PATH, get_feature_columns_timeonly())


def _load_reg_feats():
    return _load_json(REG_FEATS_PATH, get_feature_columns_timeonly())


# ----------------- 主功能 -----------------
def predict_one_anytime(
    *,
    city: str,
    sno: str,
    target_local_iso: str,
    threshold: Optional[float] = None,  # 若想同時給出 0/1 判斷，可傳入閾值（例如 0.5）
) -> Dict[str, Any]:
    """
    以『絕對時間』預測：
      - proba_can_rent（分類機率；若沒模型則為 None）
      - pred_available（預測可借數量；若沒模型則為 None）
    """
    # 1) 準備單筆特徵（純時間）
    base = build_timeonly_features_for_target(
        city=city, sno=sno, target_local_iso=target_local_iso
    )
    # base 內含 ts_local 與時間特徵；下面會各自對齊欄位

    result: Dict[str, Any] = {
        "ok": True,
        "city": city,
        "sno": sno,
        "target_local": str(base.loc[0, "ts_local"]),
        "proba_can_rent": None,
        "pred_available": None,
        "decision": None,            # 若提供 threshold 會回傳 True/False
        "used_features_cls": None,   # 方便除錯
        "used_features_reg": None,
        "msg": None,
    }

    # 2) 分類：機率
    cls = _load_cls()
    if cls is not None:
        feats_cls = _load_cls_feats()
        Xc = _align_features(base, feats_cls)
        try:
            proba = float(cls.predict(Xc)[0])  # LightGBM 的 binary 預設輸出正類機率
            result["proba_can_rent"] = proba
            result["used_features_cls"] = feats_cls
            if threshold is not None:
                result["decision"] = bool(proba >= float(threshold))
        except Exception as e:
            result["ok"] = False
            result["msg"] = f"classification predict failed: {e}"
    # 若沒有分類模型，保留為 None

    # 3) 迴歸：數量
    reg = _load_reg()
    if reg is not None:
        feats_reg = _load_reg_feats()
        Xr = _align_features(base, feats_reg)
        try:
            yhat = float(reg.predict(Xr)[0])
            result["pred_available"] = _safe_round_nonneg(yhat)
            result["used_features_reg"] = feats_reg
        except Exception as e:
            result["ok"] = False
            result["msg"] = f"regression predict failed: {e}"

    # 若兩個模型都不存在
    if cls is None and reg is None:
        result["ok"] = False
        result["msg"] = (
            f"no model found under {MODELS_DIR}. "
            f"expected: {CLS_MODEL_PATH.name} / {REG_MODEL_PATH.name}"
        )

    return result


# ----------------- 命令列介面 -----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict probability and/or available count at an absolute time (time-only features)")
    ap.add_argument("--city", default="TPE")
    ap.add_argument("--sno", required=True)
    ap.add_argument("--target", required=True, help="YYYY-MM-DD HH:MM（台北時間）")
    ap.add_argument("--threshold", type=float, default=None, help="若提供將回傳 decision=True/False")
    args = ap.parse_args()

    res = predict_one_anytime(
        city=args.city,
        sno=args.sno,
        target_local_iso=args.target,
        threshold=args.threshold,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
