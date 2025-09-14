from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is alive"}

@app.get("/health")
def health():
    return {"ok": True}