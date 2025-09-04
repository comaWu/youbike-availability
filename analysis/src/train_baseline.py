# analysis/src/train_baseline.py
import json
import numpy as np
from pathlib import Path
from .features_noRes import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick_ma_col(columns, horizon):
    """從現有欄位挑一個 ma_*，優先選跟 horizon 最接近的；沒有就回 None"""
    ma_cols = [c for c in columns if c.startswith("ma_")]
    if not ma_cols:
        return None
    ranked = sorted(
        ((abs(int(c.split("_")[1]) - horizon), int(c.split("_")[1]), c) for c in ma_cols),
        key=lambda x: (x[0], -x[1])  # 跟 horizon 差距小優先；差距相同選較大的視窗
    )
    return ranked[0][2]  # 欄位名

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO):
    # 視你的資料量調整天數；NTP 初期資料少可先用 1~3 天
    df = build_features(city, sno, days=3, horizon=HORIZON_MIN)

    # 特徵欄位（不含 ma_* / lag_* 之外的都可保留）
    feat_cols = ["hour", "dow", "is_weekend", "tot"] + \
                [c for c in df.columns if c.startswith("lag_") or c.startswith("ma_")]

    # 選一個可用的移動平均欄位作為基線；沒有就退回 lag_1
    ma_col = pick_ma_col(df.columns, HORIZON_MIN)
    base_series = df[ma_col] if ma_col else df.get("lag_1", df["available"])

    # 時序切分（最後 20% 當測試）
    split = int(len(df) * 0.8)
    # 訓練區的預測值只是佔位，不影響我們看測試區 MAE
    y_pred = np.concatenate([base_series.iloc[:split].values, base_series.iloc[split:].values])

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(df["y"].iloc[split:], base_series.iloc[split:])

    meta = {
        "model": f"moving_average_baseline({ma_col or 'lag_1'})",
        "mae_test": float(mae),
        "horizon_min": int(HORIZON_MIN),
        "features": feat_cols,
        "city": city, "sno": sno
    }
    (OUT_DIR / "baseline_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[BASELINE] MAE: {mae:.4f}  (using {ma_col or 'lag_1'})")
    print("meta saved at:", OUT_DIR / "baseline_meta.json")

if __name__ == "__main__":
    run()
