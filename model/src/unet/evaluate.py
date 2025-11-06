# src/model/unet/evaluate.py
import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from model.unet.model import U_Net

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "outputs" / "unet_lung_segmentation.best.hdf5"
INPUT_DIR = BASE_DIR / "model" / "datasets" / "chest_xray" / "test"
OUTPUT_DIR = BASE_DIR / "model" / "segmented" / "test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 256

model = U_Net()
model.load_weights(str(MODEL_PATH))
print("✅ Loaded model weights from", MODEL_PATH)

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".png")):
        continue

    img = cv2.imread(str(INPUT_DIR / file))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_tensor = np.expand_dims(img / 255.0, axis=0)

    pred_mask = model.predict(input_tensor)[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite(str(OUTPUT_DIR / file), pred_mask)
    print("🫁 Saved mask for:", file)
