from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 成功後可改白名單
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is alive"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/echo")
def echo(q: str = Query("hello")):
    return {"echo": q}
