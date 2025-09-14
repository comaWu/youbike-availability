from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 成功後改白名單：["https://你的-web.vercel.app", "http://localhost:5173"]
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", BASE_DIR / "assets"))

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict_one")
def predict_one(city: str, sno: str, target: str):
    try:
        # 路由內 lazy import，避免冷啟動時就載入 heavy 套件
        from app.predict_anytime import run_prediction_one
        result = run_prediction_one(
            city=city, sno=sno, target=target, assets_dir=str(ASSETS_DIR)
        )
        return {"ok": True,
                "proba_can_rent": result["proba"],
                "pred_available": result["available"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
