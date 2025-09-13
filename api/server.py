from fastapi import FastAPI
from analysis.src import predict_anytime, server_fastapi

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/stations")
def stations_endpoint(city: str = Query(...)):
    # 假設 server_fastapi.list_stations(city) 是正確的
    return server_fastapi.list_stations(city)

@app.get("/api/predict_one")
def predict_one_endpoint(city: str, sno: str, target: str):
    r = predict_anytime.run_prediction_one(city=city, sno=sno, target=target)
    return {
        "ok": True,
        "proba_can_rent": r["probas"], # 根據您的實際返回值調整鍵名
        "pred_available": r["available"] # 根據您的實際返回值調整鍵名
    }