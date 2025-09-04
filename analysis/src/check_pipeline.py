# analysis/src/check_pipeline.py
from pathlib import Path
import json, argparse, time
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from .config import DEFAULT_CITY, DEFAULT_SNO, HORIZON_MIN
from .features import build_features
from .train_ml_lgbm import run as lgbm_train_run
from .predict_once import predict_once
import joblib

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "latest"
MD_REPORT = OUT_DIR / "pipeline_report.md"
EVAL_JSON = OUT_DIR / "eval_summary.json"
FI_CSV    = OUT_DIR / "feature_importances.csv"
PREVIEW_CSV = OUT_DIR / "feature_preview.csv"

def tnow(): return time.perf_counter()

def header(t): print(f"\n=== {t} ===")

def stage_build_features(city, sno, days, horizon, mode, head_n):
    header("Stage 1: BUILD FEATURES")
    t0 = tnow()
    df = build_features(city=city, sno=sno, days=days, horizon=horizon)
    t1 = tnow()
    info = {
        "shape": list(df.shape),
        "build_sec": round(t1 - t0, 3)
    }
    print(f"[FEATURES] shape={df.shape}  ({info['build_sec']}s)")

    if df.empty:
        print("⚠️ 特徵集為空（請增加 days、降低 horizon 或確認資料抓取）")
        return df, info

    if mode in ("normal", "verbose"):
        print(df.head(head_n).to_string(index=False))
    if mode == "verbose":
        na = df.isna().sum()
        na = na[na > 0]
        if not na.empty:
            print("\n[NaN columns]")
            print(na.sort_values(ascending=False).to_string())
        print("\n[Columns]")
        print(", ".join(df.columns.tolist()))

    # 存一份小預覽（給報表用）
    try:
        df.head(min(head_n, 10)).to_csv(PREVIEW_CSV, index=False)
    except Exception:
        pass

    return df, info

def stage_train_lgbm(city, sno, days, horizon, mode):
    header("Stage 2: TRAIN (LightGBM)")
    t0 = tnow()
    lgbm_train_run(city=city, sno=sno, days=days, horizon=horizon)  # 內部會印 MAE/R2
    t1 = tnow()
    sec = round(t1 - t0, 3)
    print(f"[TRAIN] done in {sec}s")

    # 顯示 Top-K feature importance
    model_path = OUT_DIR / "model_lgbm.pkl"
    feats_path = OUT_DIR / "feature_columns.json"
    top = 15
    fi_list = []
    if model_path.exists() and feats_path.exists():
        model = joblib.load(model_path)
        feats = json.loads(feats_path.read_text())
        try:
            fi = pd.DataFrame({"feature": feats, "importance": model.feature_importances_}) \
                    .sort_values("importance", ascending=False)
            fi.to_csv(FI_CSV, index=False)
            if mode in ("normal", "verbose"):
                print("\n[Top Feature Importances]")
                print(fi.head(top).to_string(index=False))
            fi_list = fi.head(top).to_dict(orient="records")
        except Exception as e:
            print(f"(skip FI: {e})")
    else:
        print("⚠️ 未找到 model/feats 輸出，可能訓練失敗")
    return {"train_sec": sec, "fi_top": fi_list}

def stage_evaluate(city, sno, days, horizon, mode, head_n):
    header("Stage 3: EVALUATE")
    # build features again to keep split coherent with latest
    df = build_features(city=city, sno=sno, days=days, horizon=horizon)
    if df.empty:
        print("⚠️ 無法評估：特徵集為空")
        return None

    model_path = OUT_DIR / "model_lgbm.pkl"
    feats_path = OUT_DIR / "feature_columns.json"
    if not (model_path.exists() and feats_path.exists()):
        print("⚠️ 找不到 LGBM 模型或特徵欄位，跳過評估")
        return None

    model = joblib.load(model_path)
    feats = json.loads(feats_path.read_text())

    X = df[feats].copy()
    y = df["y"]
    for c in ("city", "sno"):
        if c in X.columns:
            X[c] = X[c].astype("category")

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

    if mode in ("normal", "verbose"):
        cmp = pd.DataFrame({"y_true": y_te.reset_index(drop=True),
                            "y_pred": pd.Series(yhat_te).round(3)})
        print("\n[Test head]")
        print(cmp.head(head_n).to_string(index=False))
    else:
        cmp = None

    summary = {"mae_train": mae_tr, "r2_train": r2_tr,
               "mae_test": mae_te, "r2_test": r2_te,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te))}
    Path(EVAL_JSON).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

def stage_predict_once(city, sno, horizon, days):
    header("Stage 4: PREDICT ONCE")
    res = predict_once(city=city, sno=sno, horizon=horizon, days=days)
    print(res)
    return res

def write_markdown(city, sno, days, horizon, bf_info, tr_info, eval_info, pred_info):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = []
    md.append(f"# YouBike Pipeline Report\n")
    md.append(f"- city: **{city}**  sno: **{sno}**  days: **{days}**  horizon: **{horizon}m**\n")
    md.append(f"## Build Features\n")
    md.append(f"- shape: `{bf_info.get('shape')}`  · build_sec: `{bf_info.get('build_sec')}`\n")
    md.append(f"- 預覽：`feature_preview.csv`\n")
    md.append(f"## Train (LightGBM)\n")
    md.append(f"- train_sec: `{tr_info.get('train_sec')}`\n")
    if tr_info.get("fi_top"):
        md.append(f"- Top Feature Importances（完整見 `feature_importances.csv`）:\n")
        md.append("| feature | importance |\n|---|---|")
        for r in tr_info["fi_top"]:
            md.append(f"| {r['feature']} | {r['importance']} |")
        md.append("")
    md.append(f"## Evaluate\n")
    if eval_info:
        md.append(f"- Train: MAE={eval_info['mae_train']:.4f}, R2={eval_info['r2_train']:.4f}, n={eval_info['n_train']}")
        md.append(f"- Test : MAE={eval_info['mae_test']:.4f}, R2={eval_info['r2_test']:.4f}, n={eval_info['n_test']}")
        md.append(f"- JSON：`eval_summary.json`\n")
    else:
        md.append("- （無法評估）\n")
    md.append(f"## Predict Once\n")
    md.append(f"```json\n{json.dumps(pred_info, indent=2, ensure_ascii=False)}\n```")
    MD_REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n📄 報告已輸出：{MD_REPORT}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--sno", default=DEFAULT_SNO)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=HORIZON_MIN)
    parser.add_argument("--mode", choices=["brief","normal","verbose"], default="normal")
    parser.add_argument("--head", type=int, default=5, help="每段顯示的列數")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, bf_info = stage_build_features(args.city, args.sno, args.days, args.horizon, args.mode, args.head)
    if df is None or df.empty:
        return

    tr_info = stage_train_lgbm(args.city, args.sno, args.days, args.horizon, args.mode)
    eval_info = stage_evaluate(args.city, args.sno, args.days, args.horizon, args.mode, args.head)
    pred_info = stage_predict_once(args.city, args.sno, args.horizon, args.days)

    write_markdown(args.city, args.sno, args.days, args.horizon, bf_info, tr_info, eval_info, pred_info)

if __name__ == "__main__":
    main()
