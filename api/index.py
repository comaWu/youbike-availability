# /api/index.py
from fastapi import FastAPI, Query, HTTPException
from typing import Optional

# 同資料夾匯入（已搬進 /api）
from predict_anytime import run_prediction_one   # 依你的函式名調整
import server_fastapi  # 若裡面有工具函式可照用

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict_one")
def predict_one(city: str = Query(...), sno: str = Query(...), target: str = Query(...)):
    try:
        r = run_prediction_one(city=city, sno=sno, target=target)
        return {
            "ok": True,
            "proba_can_rent": r["proba"],
            "pred_available": r["available"]
        }
    except Exception as e:
        # 保底錯誤處理
        raise HTTPException(status_code=500, detail=str(e))
