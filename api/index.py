# api/index.py
# --- A) 若你已有 FastAPI app 定義（建議） ---
# 假設你的專案已有 analysis/src/server_fastapi.py 且內有 `app`
# from analysis.src.server_fastapi import app

# --- B) 若沒有現成 app，先用最小可跑版本 ---
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json

app = FastAPI(title="YouBike API")

# 假設你的模型檔放在 api/assets/models/latest/ 之下
MODEL_DIR = Path(__file__).parent / "assets" / "models" / "latest"

# 這只是示範：實務上你會載入真實的模型（例如 joblib.load / lightgbm.Booster.load_model）
def load_dummy_model():
    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"model": "not_found"}

MODEL_META = load_dummy_model()

class PredictOneRequest(BaseModel):
    city: str
    sno: str
    when: str  # "YYYY-MM-DD HH:MM"

@app.get("/healthz")
def healthz():
    return {"ok": True, "model_loaded": bool(MODEL_META)}

@app.post("/predict_one")
def predict_one(req: PredictOneRequest):
    # TODO: 在這裡呼叫你的預測邏輯
    # e.g. y = model.predict(features)
    # 這裡先回傳假資料
    return {
        "city": req.city,
        "sno": req.sno,
        "when": req.when,
        "availability_prob": 0.73,
        "meta": MODEL_META,
    }
