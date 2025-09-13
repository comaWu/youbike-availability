# /api/index.py
from fastapi import FastAPI, Query
# 直接沿用你 analysis 的邏輯
from analysis.src import predict_anytime  # 依你的模組路徑調整
from analysis.src import server_fastapi   # 若你已寫好 handler 也可直接 import 使用

app = FastAPI()

@app.get("/stations")
def stations(city: str = Query(...)):
    # TODO: 回傳你現有的站點清單（可先接到你現有的讀檔/DB 函式）
    return server_fastapi.list_stations(city)  # 例：若你那邊有現成方法

@app.get("/predict_one")
def predict_one(city: str, sno: str, target: str):
    # 舉例：呼叫你的預測函式
    r = predict_anytime.run_prediction_one(city=city, sno=sno, target=target)
    # 依照前端期望欄位回傳
    return {
        "ok": True,
        "proba_can_rent": r["proba"],    # 0~1
        "pred_available": r["available"] # 整數
    }
