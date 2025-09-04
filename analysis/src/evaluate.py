# analysis/src/evaluate.py
from pathlib import Path
import json, joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from .features import build_features
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
EVAL_JSON   = OUT_DIR / "eval_summary.json"
TEST_CSV    = OUT_DIR / "test_predictions.csv"
MAE_PNG     = OUT_DIR / "metrics_mae.png"
R2_PNG      = OUT_DIR / "metrics_r2.png"
SCATTER_PNG = OUT_DIR / "true_vs_pred.png"

def load_latest_model_and_features():
    """優先載入 LGBM；沒有就用 GBR；並讀取訓練時保存的特徵欄位。"""
    if (OUT_DIR / "model_lgbm.pkl").exists():
        model_path = OUT_DIR / "model_lgbm.pkl"
    else:
        model_path = OUT_DIR / "model_gbr.pkl"
    model = joblib.load(model_path)

    feats = json.loads((OUT_DIR / "feature_columns.json").read_text())
    return model, feats

def run(city=DEFAULT_CITY, sno=DEFAULT_SNO, days=14, horizon=HORIZON_MIN):
    # 1) 構建與訓練一致的特徵
    df = build_features(city, sno, days=days, horizon=horizon)
    if df.empty:
        raise RuntimeError("特徵集為空，請增加 days 或確認資料是否持續擷取")

    # 2) 載入模型與特徵欄位
    model, feats = load_latest_model_and_features()

    # 3) 對齊特徵；LGBM 的類別特徵轉 category
    #    若 feats 中有缺欄位（例如你關掉某些 lag/ma），此處可視需要補 0：
    # for c in feats:
    #     if c not in df.columns:
    #         df[c] = 0
    X = df[feats].copy()
    y = df["y"]
    for c in ("city", "sno"):
        if c in X.columns:
            X[c] = X[c].astype("category")

    # 4) 80/20 切分並評估
    split = int(len(df) * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)

    mae_tr = float(mean_absolute_error(y_tr, yhat_tr))
    r2_tr  = float(r2_score(y_tr, yhat_tr))
    mae_te = float(mean_absolute_error(y_te, yhat_te))
    r2_te  = float(r2_score(y_te, yhat_te))

    print(f"[Train] MAE={mae_tr:.4f}  R2={r2_tr:.4f}   n={len(y_tr)}")
    print(f"[Test ] MAE={mae_te:.4f}  R2={r2_te:.4f}   n={len(y_te)}")

    # 5) 保存測試集對照（給散點圖用）
    compare = pd.DataFrame({
        "y_true": y_te.reset_index(drop=True),
        "y_pred": pd.Series(yhat_te)
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare.to_csv(TEST_CSV, index=False)

    # 6) 保存摘要 JSON
    summary = {
        "mae_train": mae_tr, "r2_train": r2_tr,
        "mae_test": mae_te, "r2_test": r2_te,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te))
    }
    EVAL_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # 7) 出圖：MAE / R² / 真實 vs 預測
    # MAE
    plt.figure(figsize=(4,4))
    plt.bar(["Train","Test"], [mae_tr, mae_te])
    plt.title("MAE")
    plt.ylabel("Error (bikes)")
    plt.tight_layout()
    plt.savefig(MAE_PNG)
    plt.close()

    # R²
    plt.figure(figsize=(4,4))
    plt.bar(["Train","Test"], [r2_tr, r2_te])
    plt.title("R²")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(R2_PNG)
    plt.close()

    # 散點（最多 1000 筆，避免太密）
    head_n = min(1000, len(compare))
    plt.figure(figsize=(5,5))
    plt.scatter(compare["y_true"].head(head_n), compare["y_pred"].head(head_n), alpha=0.5)
    m = max(compare["y_true"].max(), compare["y_pred"].max())
    plt.plot([0, m], [0, m], "--")
    plt.xlabel("True (test)")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted (Test)")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG)
    plt.close()

    print("Saved:", EVAL_JSON, TEST_CSV, MAE_PNG, R2_PNG, SCATTER_PNG)

if __name__ == "__main__":
    run()
