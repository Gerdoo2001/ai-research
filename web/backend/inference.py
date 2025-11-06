import os
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 224

# Compute absolute base directory (repo root)
# backend -> web -> ai-research
BASE_DIR = Path(__file__).resolve().parents[2]  # -> ai-research/
MODEL_PATH = os.path.join(BASE_DIR, "model", "outputs", "densenet121_pneumonia.h5")

_model = None


def load_model():
    """Load DenseNet121 model (cached after first call)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        print(f"🧠 Loading model from {MODEL_PATH} ...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    return _model


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize to model input size, normalize, add batch dimension."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict(pil_img: Image.Image, threshold: float = 0.5):
    """Predict pneumonia from chest X-ray."""
    model = load_model()
    x = preprocess_image(pil_img)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    return {"label": label, "probability": prob, "threshold": threshold}
