# src/model/unet/data_generator.py
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]  # -> ai-research/
DATASET_ROOT = BASE_DIR / "model" / "datasets" / "lung_segmentation"

# Folder structure should be:
# model/datasets/lung_segmentation/
# ├── images/
# │   ├── 0001.png
# │   ├── 0002.png
# │   └── ...
# └── masks/
#     ├── 0001_mask.png
#     ├── 0002_mask.png
#     └── ...

IMAGE_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks"

# ==========================================
# MAIN DATA LOADER
# ==========================================
def getData(img_size=256, flag="train"):
    """
    Loads lung segmentation dataset (images + masks)
    Args:
        img_size (int): Target resize for image and mask
        flag (str): "train" or "test"
    Returns:
        (X, Y): Tuple of numpy arrays of shape
                X -> (N, img_size, img_size, 3)
                Y -> (N, img_size, img_size, 1)
    """
    print(f"📂 Loading data from {IMAGE_DIR} and {MASK_DIR}")

    X, Y = [], []
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    for fname in image_files:
        # Match mask filename (same base name)
        mask_name = fname.replace(".jpg", "_mask.png").replace(".jpeg", "_mask.png").replace(".png", "_mask.png")
        img_path = IMAGE_DIR / fname
        mask_path = MASK_DIR / mask_name

        if not mask_path.exists():
            print(f"⚠️ Missing mask for {fname}, skipping...")
            continue

        # Load image (BGR -> RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

        # Load mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size, img_size))
        mask = np.expand_dims(mask, axis=-1)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        X.append(img)
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    print(f"✅ Loaded {len(X)} samples. Shape: {X.shape}, {Y.shape}")
    return X, Y


# ==========================================
# TEST RUN
# ==========================================
if __name__ == "__main__":
    X, Y = getData(img_size=256, flag="train")
    print("Sample data shapes:", X.shape, Y.shape)
