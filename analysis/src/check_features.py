# analysis/src/check_features.py
import pandas as pd
from analysis.src.features import build_features

def check_samples(city="TPE", sno=None, days=1, horizon=5):
    """檢查在不同階段剩下多少樣本"""
    df = build_features(city, sno, days=days, horizon=horizon)

    print(f"總樣本數: {len(df)}")
    if df.empty:
        print("⚠️ 特徵集為空，可能原因：資料不足、lag/horizon 太大、或 dropna 全刪光")
        return

    # 統計 NaN 狀況
    nan_counts = df.isna().sum()
    print("\n每個欄位缺值數：")
    print(nan_counts[nan_counts > 0])

    # 計算 drop 的比例
    total = len(df)
    valid = df.dropna().shape[0]
    print(f"\n完整樣本數: {valid} / {total} ({valid/total:.1%})")

    # 頭尾幾筆樣本
    print("\n--- 前 5 筆 ---")
    print(df.head(5)[["ts","available"]+[c for c in df.columns if c.startswith("lag_")]+["y"]])
    print("\n--- 後 5 筆 ---")
    print(df.tail(5)[["ts","available"]+[c for c in df.columns if c.startswith("lag_")]+["y"]])

if __name__ == "__main__":
    check_samples()
