# YouBike Availability Prediction 🚲

這是一個 **台北/新北 YouBike 站點即時可借數量預測系統**，包含資料抓取、特徵工程、機器學習訓練與前端地圖展示。

## 🔄 專案總覽
資料來源：每 5 分鐘抓取 YouBike API，存入 PostgreSQL
目標：在地圖上點選任一站點，指定「絕對時間（YYYY-MM-DD HH:MM）」後，回傳：
可租機率（分類模型）
預估可借數量（迴歸模型）
路徑：B – 單一模型 + 站點特徵（tot/lat/lng），確保不同站點在同時刻會得到不同預測
後端：FastAPI
前端：Leaflet（React），可點站點、選日期時間、即時顯示結果


1) 資料來源與 DB 設計

表 station（靜態）：city, sno, sna, lat, lng, tot, is_active...
表 station_minute（動態）：city, sno, ts, available, ...
訓練標籤
分類：y = (available > 0 ? 1 : 0)
迴歸：available（整數）

# 訓練
# 機率（分類）
python -m analysis.src.train_ml_lgbm_anytime --val_size 0.2
# 數量（迴歸）
python -m analysis.src.train_ml_lgbm_regression --val_size 0.2

# FASTAPI
uvicorn analysis.src.server_fastapi:app --reload --host 0.0.0.0 --port 8000
健康檢查：GET http://localhost:8000/healthz
站點清單：GET http://localhost:8000/api/stations?city=TPE
站點資訊：GET http://localhost:8000/api/stations/TPE/500101001
單點推論：GET http://localhost:8000/api/predict_one?city=TPE&sno=500101001&target=2025-09-10%2008:00

# VUE
#npm run dev

