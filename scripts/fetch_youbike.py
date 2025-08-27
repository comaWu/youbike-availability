import os, time, math, logging, datetime as dt
import requests
import psycopg
from dotenv import load_dotenv

load_dotenv()

# ---------- 設定 ----------
PG_DSN = (
    f"postgresql://{os.getenv('PG_USER','postgres')}:{os.getenv('PG_PASSWORD','postgres')}"
    f"@{os.getenv('PG_HOST','localhost')}:{os.getenv('PG_PORT','5432')}/{os.getenv('PG_DB','youbike')}"
)

URL_TPE = os.getenv("TPE_YOUBIKE_URL",
    "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json")
URL_NTP = os.getenv("NTP_YOUBIKE_URL",
    "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json")

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.getenv("HTTP_BACKOFF_BASE", "1.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

# logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ingest")

# ---------- 工具 ----------
def to_int(x, d=0):
    try:
        return int(x)
    except Exception:
        return d

def to_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

def parse_src_time(mday):
    if not mday:
        return dt.datetime.utcnow()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(mday, fmt)
        except Exception:
            pass
    return dt.datetime.utcnow()

def http_get_json(url: str):
    """帶指數退避的 GET+JSON"""
    last_err = None
    for i in range(HTTP_RETRIES):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            # 對 5xx/429 等做重試；4xx 多半是請求問題，直接拋出
            if r.status_code >= 500 or r.status_code in (408, 429):
                raise requests.HTTPError(f"retryable status={r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            backoff = (HTTP_BACKOFF_BASE ** i)
            log.warning(f"GET {url} failed (try {i+1}/{HTTP_RETRIES}): {e}; backoff {backoff:.1f}s")
            time.sleep(backoff)
    # 超過重試次數，拋最後一次錯
    raise RuntimeError(f"GET {url} failed after {HTTP_RETRIES} retries: {last_err}")

# ---------- 欄位對齊 ----------
def normalize_tpe(rec):
    """台北一筆 → 統一 schema"""
    return {
        "city": "TPE",
        "sno": rec.get("sno"),
        "sna": rec.get("sna"),
        "sarea": rec.get("sarea"),
        "ar": rec.get("ar"),
        "lat": to_float(rec.get("latitude") or rec.get("lat")),
        "lng": to_float(rec.get("longitude") or rec.get("lng")),
        "tot": to_int(rec.get("Quantity") or rec.get("tot")),
        "available": to_int(rec.get("available_rent_bikes") or rec.get("sbi")),
        "empty": to_int(rec.get("available_return_bikes") or rec.get("bemp")),
        "is_active": str(rec.get("act","1")) == "1",
        "src_update": parse_src_time(rec.get("mday")),
    }

def normalize_ntp(rec):
    """新北一筆 → 統一 schema"""
    return {
        "city": "NTP",
        "sno": rec.get("sno"),
        "sna": rec.get("sna"),
        "sarea": rec.get("sarea"),
        "ar": rec.get("ar"),
        "lat": to_float(rec.get("lat")),
        "lng": to_float(rec.get("lng")),
        "tot": to_int(rec.get("tot")),
        "available": to_int(rec.get("sbi")),
        "empty": to_int(rec.get("bemp")),
        "is_active": str(rec.get("act","1")) == "1",
        "src_update": parse_src_time(rec.get("mday")),
    }

# ---------- SQL ----------
UPSERT_STATION_SQL = """
INSERT INTO station(city, sno, sna, sarea, ar, lat, lng, tot)
VALUES (%(city)s, %(sno)s, %(sna)s, %(sarea)s, %(ar)s, %(lat)s, %(lng)s, %(tot)s)
ON CONFLICT (city, sno) DO UPDATE
  SET sna=EXCLUDED.sna, sarea=EXCLUDED.sarea, ar=EXCLUDED.ar,
      lat=EXCLUDED.lat, lng=EXCLUDED.lng, tot=EXCLUDED.tot;
"""

INSERT_SNAPSHOT_SQL = """
INSERT INTO station_minute(ts, city, sno, available, empty, is_active, src_update)
VALUES (date_trunc('minute', NOW()), %(city)s, %(sno)s, %(available)s, %(empty)s, %(is_active)s, %(src_update)s)
ON CONFLICT (ts, city, sno) DO NOTHING;
"""

# ---------- DB 批次寫入 ----------
def chunks(seq, n):
    """把序列切成大小為 n 的小段"""
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def upsert_batch(records):
    total = len(records)
    if total == 0:
        log.info("no records to upsert")
        return
    with psycopg.connect(PG_DSN, autocommit=False) as conn, conn.cursor() as cur:
        done = 0
        # 先 upsert station（靜態），再寫 snapshot（動態）
        # 這裡用簡單的迴圈；若量更大可考慮 COPY 或 execute_batch
        for batch in chunks(records, BATCH_SIZE):
            for r in batch:
                # 基本防呆
                if not r["sno"] or r["lat"] == 0 or r["lng"] == 0:
                    continue
                try:
                    cur.execute(UPSERT_STATION_SQL, r)
                    cur.execute(INSERT_SNAPSHOT_SQL, r)
                    done += 1
                except Exception as e:
                    # 單筆失敗不影響整批（log 之後繼續）
                    log.error(f"upsert failed for {r['city']}:{r['sno']} - {e}")
            conn.commit()
            log.info(f"committed batch: {done}/{total}")
    log.info(f"upsert finished: {done}/{total}")

# ---------- 主流程 ----------
def main():
    log.info("fetching TPE...")
    tpe = http_get_json(URL_TPE)
    log.info("fetching NTP...")
    ntp = http_get_json(URL_NTP)

    if not isinstance(tpe, list) or not isinstance(ntp, list):
        raise RuntimeError("Unexpected API response format (not list)")

    norm_tpe = [normalize_tpe(x) for x in tpe]
    norm_ntp = [normalize_ntp(x) for x in ntp]
    all_recs = norm_tpe + norm_ntp

    log.info(f"normalized: TPE={len(norm_tpe)}, NTP={len(norm_ntp)}, total={len(all_recs)}")

    upsert_batch(all_recs)
    log.info("done.")

if __name__ == "__main__":
    main()