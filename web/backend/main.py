"""
FastAPI Backend for Pneumonia Classification
Author: John Mark
"""

import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.staticfiles import StaticFiles
from PIL import Image
import importlib.resources as pkg_resources

from schemas import PredictionResponse
from inference import predict, load_model


# =====================================================
# FastAPI app config
# =====================================================
app = FastAPI(title="Pneumonia Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Mount LOCAL swagger-ui assets
# =====================================================
try:
    import swagger_ui_bundle

    # Locate the package’s static folder
    with pkg_resources.as_file(pkg_resources.files(swagger_ui_bundle)) as swagger_path:
        app.mount("/swagger-ui", StaticFiles(directory=swagger_path), name="swagger-ui")
        print(f"✅ Serving Swagger UI assets locally from: {swagger_path}")
except Exception as e:
    print(f"[WARN] Could not mount local Swagger UI assets: {e}")


# =====================================================
# Override /docs to use local JS & CSS
# =====================================================
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Local Swagger UI (no CDN calls).
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
        swagger_js_url="/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/swagger-ui/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_redirect():
    return get_swagger_ui_oauth2_redirect_html()


# =====================================================
# Startup: preload model
# =====================================================
@app.on_event("startup")
def warmup():
    try:
        load_model()
        print("✅ Model preloaded successfully.")
    except Exception as e:
        print(f"[WARN] Model preload failed: {e}")


# =====================================================
# Endpoints
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
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


# =====================================================
# Run directly
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
