import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from schemas import PredictionResponse
from inference import predict, load_model

app = FastAPI(title="Pneumonia Classification API", version="1.0.0")

# CORS (open for now; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def warmup():
    # Preload model on startup so first request is fast
    try:
        load_model()
    except Exception as e:
        # Don't crash startup; raise on first request instead
        print(f"[WARN] Model preload failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # Basic content-type guard
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only JPEG or PNG images are supported.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        result = predict(img)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
