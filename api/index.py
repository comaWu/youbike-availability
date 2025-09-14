from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is alive"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/echo")
def echo(q: str = Query("hello")):
    return {"echo": q}
