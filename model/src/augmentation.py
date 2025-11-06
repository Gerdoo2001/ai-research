import os
import cv2
from pathlib import Path

# ==========================================
# CONFIGURATION (absolute paths)
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]  # -> ai-research/
SRC_ROOT = BASE_DIR / "model" / "datasets" / "chest_xray" / "train"
DST_ROOT = BASE_DIR / "model" / "augmented_data" / "train"

CATEGORIES = ["NORMAL", "PNEUMONIA"]


# ==========================================
# UTILITIES
# ==========================================
def ensure_dir(p: Path):
    """Create directory if it doesn’t exist."""
    p.mkdir(parents=True, exist_ok=True)


def flip_horizontal(img):
    """Flip image horizontally."""
    return cv2.flip(img, 1)


def apply_clahe(img):
    """Apply CLAHE enhancement to improve local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ==========================================
# MAIN AUGMENTATION LOGIC
# ==========================================
def augment_images():
    for cat in CATEGORIES:
        src = SRC_ROOT / cat
        dst = DST_ROOT / cat
        ensure_dir(dst)

        print(f"Processing {cat} images...")
        if not src.exists():
            print(f"⚠️  Missing folder: {src}")
            continue

        for name in os.listdir(src):
            if not name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = str(src / name)
            img = cv2.imread(path)
            if img is None:
                continue

            base = Path(name).stem

            # 1️⃣ CLAHE enhancement
            clahe_img = apply_clahe(img)
            clahe_path = dst / f"{base}_clahe.jpg"
            cv2.imwrite(str(clahe_path), clahe_img)

            # 2️⃣ Horizontal flip (after CLAHE)
            flipped = flip_horizontal(clahe_img)
            flip_path = dst / f"{base}_flip.jpg"
            cv2.imwrite(str(flip_path), flipped)

            # 3️⃣ (Future) Lung segmentation
            # ------------------------------------------------
            # from your future implementation:
            # segmented = segment_lungs(your_model, clahe_img)
            # seg_path = dst / f"{base}_seg.jpg"
            # cv2.imwrite(str(seg_path), segmented)
            # ------------------------------------------------

    print(f"\n✅ Augmentation complete. Results saved to: {DST_ROOT}")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    augment_images()
