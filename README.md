\# 🚲 YouBike Availability AI Prediction



> 利用 AI/ML 預測 YouBike 站點未來幾分鐘的可借數，並結合 React + Leaflet 做互動地圖展示。



---



\## ✨ 專案特色



\- \*\*即時資料蒐集\*\*：定時從 YouBike 官方 API 抓取站點狀態。

\- \*\*時序資料庫\*\*：PostgreSQL 儲存完整時序資料。

\- \*\*特徵工程\*\*：lag、移動平均、時間特徵（hour, dow, weekend）。

\- \*\*AI 預測模型\*\*：

&nbsp; - Baseline：移動平均

&nbsp; - GradientBoosting (scikit-learn)

&nbsp; - LightGBM（支援 categorical features: city, sno）

\- \*\*地圖前端\*\*：

&nbsp; - React + Leaflet

&nbsp; - 彩色 marker 顯示可租機率

&nbsp; - Popup 顯示可借/預測數

&nbsp; - Legend（圖例）+ 比例尺

&nbsp; - 使用者輸入地址或地圖點擊起/終點 → 自動規劃路線（步行/自行車/汽車）

\- \*\*部署\*\*：

&nbsp; - GitHub 管理版本

&nbsp; - Vercel 部署前端

&nbsp; - 後端 API 可用 Render / Railway / 自架伺服器



---



\## 🏗️ 專案架構





---



\## 🔄 專案流程



1\. \*\*資料蒐集\*\*  

&nbsp;  - `crawler.py` 每 5 分鐘呼叫 YouBike API，存進 PostgreSQL (`station\_records`)。



2\. \*\*特徵工程\*\* (`features.py`)  

&nbsp;  - 加入 lag (`lag\_1, lag\_5, ...`)、移動平均 (`ma\_3, ma\_5, ma\_10`)、時間特徵。



3\. \*\*模型訓練\*\*  

&nbsp;  - `train\_baseline.py` → 移動平均基準

&nbsp;  - `train\_ml.py` → GradientBoosting

&nbsp;  - `train\_ml\_lgbm.py` → LightGBM



4\. \*\*模型評估\*\* (`evaluate.py`)  

&nbsp;  - 輸出 MAE、R²

&nbsp;  - 比較真實值 vs 預測值



5\. \*\*即時預測\*\* (`predict\_once.py`)  

&nbsp;  - 查 DB 最新狀態 → 載入模型 → 預測未來 X 分鐘可借數 \& 機率



6\. \*\*前端展示\*\* (`web/`)  

&nbsp;  - Leaflet 地圖顯示站點可借狀態與預測

&nbsp;  - Legend（右上角圖例）、比例尺

&nbsp;  - 起/終點輸入 \& 點擊 → 規劃路徑



---



\## ⚙️ 安裝與使用



\### 1. 後端 (資料蒐集 + 訓練)



```bash

\# 建立虛擬環境或用 docker

pip install -r requirements.txt



\# 抓取資料

python -m analysis.src.crawler



\# 特徵檢查

python -m analysis.src.check\_features --city NTP --sno 500202005 --days 1 --horizon 5



\# 訓練基線模型

python -m analysis.src.train\_baseline



\# 訓練 ML 模型

python -m analysis.src.train\_ml\_lgbm



\# 評估

python -m analysis.src.evaluate



cd web

npm install

npm run dev   # 本地開發

npm run build # 打包





