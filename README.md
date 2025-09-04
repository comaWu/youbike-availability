# YouBike Availability Prediction 🚲

這是一個 **台北/新北 YouBike 站點即時可借數量預測系統**，包含資料抓取、特徵工程、機器學習訓練與前端地圖展示。

## 功能特色
- ⏱ **爬蟲**：每 5 分鐘抓取 YouBike API，存入 PostgreSQL
- 🗂 **資料處理**：逐站補齊分鐘數據，生成時間特徵、滯後特徵、移動平均
- 🤖 **模型**：使用 LightGBM，MAE < 0.2，R² > 0.95
- 📊 **報表**：`check_pipeline.py` 自動輸出完整報告與圖表
- 🗺 **前端**：React + Leaflet，地圖視覺化顯示預測結果




## 🔄 專案流程

1. **資料蒐集**

  - crawler.py 每 5 分鐘呼叫 YouBike API，存進 PostgreSQL (station\_records)。

2. **特徵工程** (features.py)

  - 加入 lag (lag\_1, lag\_5, ...)、移動平均 (ma\_3, ma\_5, ma\_10)、時間特徵。

3. **模型訓練**

  - train\_baseline.py → 移動平均基準

  - train\_ml.py → GradientBoosting

  - train\_ml\_lgbm.py → LightGBM

4. **模型評估** (evaluate.py)

  - 輸出 MAE、R²

  - 比較真實值 vs 預測值

5. **即時預測** (predict\_once.py)

  - 查 DB 最新狀態 → 載入模型 → 預測未來 X 分鐘可借數 & 機率


# 訓練（含 early stopping）
python -m analysis.src.train_ml_lgbm

# 評估
python -m analysis.src.evaluate

# 單次預測
python -m analysis.src.predict_once

python -m analysis.src.check_pipeline --mode brief --days 3 --horizon 5
python -m analysis.src.check_pipeline --mode brief --days 3 --horizon 5
python -m analysis.src.check_pipeline




