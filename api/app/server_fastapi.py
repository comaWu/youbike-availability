from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ROOT, MODELS_DIR
from .predict_anytime import predict_one_anytime

# -----------------------------------------------------------------------------
# 基本設定 & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("server")

app = FastAPI(title="YouBike Anytime Prediction API", version="1.0.0")

# 開發用 CORS（可視需要在 .env 設 FRONTEND_ORIGIN 或改成 *）
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:5174", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 站點資料載入（stations.json）
# -----------------------------------------------------------------------------
DATA_DIR = ROOT / "analysis" / "src"
STATIONS_JSON = DATA_DIR / "stations.json"

_stations_index: Dict[str, Dict[str, dict]] = {}   # { city: { sno: station_dict } }
_last_loaded: Optional[str] = None


def _load_stations() -> int:
    """從 stations.json 載入並建立索引。"""
    global _stations_index, _last_loaded
    if not STATIONS_JSON.exists():
        log.warning(f"stations.json not found: {STATIONS_JSON}")
        _stations_index = {}
        _last_loaded = None
        return 0

    with STATIONS_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)

    idx: Dict[str, Dict[str, dict]] = {}
    for it in items:
        city = it.get("city") or it.get("sarea") or "UNK"
        sno = str(it.get("sno"))
        idx.setdefault(city, {})[sno] = it

    _stations_index = idx
    _last_loaded = str(STATIONS_JSON)
    total = sum(len(v) for v in _stations_index.values())
    log.info(f"Loaded stations: {total} from {STATIONS_JSON}")
    return total


# 啟動即載入一次
_load_stations()


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictOneIn(BaseModel):
    city: str = Field(..., description="城市代碼，如 TPE")
    sno: str = Field(..., description="站點編號")
    target: str = Field(..., description="目標時間（台北時間）YYYY-MM-DD HH:MM")
    threshold: Optional[float] = Field(None, description="可選的決策閾值（0~1）")


class PredictOneOut(BaseModel):
    ok: bool
    city: str
    sno: str
    target_local: str
    proba_can_rent: Optional[float] = None
    pred_available: Optional[int] = None
    decision: Optional[bool] = None
    msg: Optional[str] = None


class PredictManyIn(BaseModel):
    items: List[PredictOneIn]


class PredictAllAtOutItem(BaseModel):
    city: str
    sno: str
    sna: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    proba_can_rent: Optional[float] = None
    pred_available: Optional[int] = None


# -----------------------------------------------------------------------------
# Health & Admin
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    total = sum(len(v) for v in _stations_index.values())
    return {
        "ok": True,
        "time": __import__("datetime").datetime.utcnow().isoformat(),
        "models_dir": str(MODELS_DIR),
        "stations_indexed": total,
        "stations_path": str(STATIONS_JSON),
    }


@app.post("/admin/reload_stations")
def admin_reload_stations():
    total = _load_stations()
    return {"ok": True, "stations_indexed": total, "path": str(STATIONS_JSON)}


# -----------------------------------------------------------------------------
# Stations APIs
# -----------------------------------------------------------------------------
@app.get("/api/stations")
def list_stations(city: Optional[str] = Query(None, description="城市代碼，如 TPE")):
    """
    取站點清單；若給 city 僅回該城市，否則回全部。
    """
    if not _stations_index:
        _load_stations()
    if city:
        data = list(_stations_index.get(city, {}).values())
    else:
        # 扁平化所有城市
        data: List[dict] = []
        for c in _stations_index.values():
            data.extend(c.values())
    return data


@app.get("/api/stations/{city}/{sno}")
def get_station(city: str, sno: str):
    if not _stations_index:
        _load_stations()
    s = _stations_index.get(city, {}).get(str(sno))
    if not s:
        raise HTTPException(status_code=404, detail="station not found")
    return s


# -----------------------------------------------------------------------------
# Prediction APIs
# -----------------------------------------------------------------------------
@app.get("/api/predict_one", response_model=PredictOneOut)
def api_predict_one(
    city: str = Query(...),
    sno: str = Query(...),
    target: str = Query(..., description="YYYY-MM-DD HH:MM (台北時間)"),
    threshold: Optional[float] = Query(None),
):
    """
    以「絕對時間」推論單一站點：
    - 同時回傳：租得到機率 (classification) ＋ 預估可借數量 (regression)
    - 兩種模型缺一也能運作（缺的欄位為 null）
    """
    log.info(f"[predict_one] city={city} sno={sno} target={target} thr={threshold}")
    res = predict_one_anytime(city=city, sno=str(sno), target_local_iso=target, threshold=threshold)

    # 壓平只保留對前端有用的欄位
    return {
        "ok": bool(res.get("ok")),
        "city": res.get("city"),
        "sno": res.get("sno"),
        "target_local": res.get("target_local"),
        "proba_can_rent": res.get("proba_can_rent"),
        "pred_available": res.get("pred_available"),
        "decision": res.get("decision"),
        "msg": res.get("msg"),
    }


@app.post("/api/predict_many")
def api_predict_many(payload: PredictManyIn):
    """
    批次推論多個站點／不同時間。
    輸入 items=[{city,sno,target,threshold?}, ...]
    """
    out = []
    for it in payload.items:
        res = predict_one_anytime(city=it.city, sno=it.sno, target_local_iso=it.target, threshold=it.threshold)
        out.append({
            "city": res.get("city"),
            "sno": res.get("sno"),
            "target_local": res.get("target_local"),
            "proba_can_rent": res.get("proba_can_rent"),
            "pred_available": res.get("pred_available"),
            "decision": res.get("decision"),
            "ok": res.get("ok"),
            "msg": res.get("msg"),
        })
    return {"ok": True, "items": out, "n": len(out)}


@app.get("/api/predict_all_at")
def api_predict_all_at(
    city: str = Query(..., description="城市代碼，如 TPE"),
    target: str = Query(..., description="YYYY-MM-DD HH:MM (台北時間)"),
    limit: int = Query(0, ge=0, description=">0 時只取前 N 個站點（調試用）"),
):
    """
    給定城市 + 目標時間，對該城市所有站點進行推論（可能較久）。
    回傳陣列，每個元素含：sno, sna, lat, lng, proba_can_rent, pred_available
    """
    if not _stations_index:
        _load_stations()

    stations = list(_stations_index.get(city, {}).values())
    if limit and limit > 0:
        stations = stations[:limit]

    out: List[PredictAllAtOutItem] = []
    for s in stations:
        sno = str(s.get("sno"))
        sna = s.get("sna")
        lat = s.get("lat")
        lng = s.get("lng")

        res = predict_one_anytime(city=city, sno=sno, target_local_iso=target)
        out.append(PredictAllAtOutItem(
            city=city, sno=sno, sna=sna, lat=lat, lng=lng,
            proba_can_rent=res.get("proba_can_rent"),
            pred_available=res.get("pred_available")
        ))

    # 直接轉成可序列化
    return [o.model_dump() for o in out]
