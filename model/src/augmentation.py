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
def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def apply_clahe(img):
    """Apply CLAHE enhancement to improve local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def flip_horizontal(img):
    """Flip image horizontally."""
    return cv2.flip(img, 1)

# ==========================================
# MAIN AUGMENTATION LOGIC
# ==========================================
def augment_images():
    total = 0
    for category in CATEGORIES:
        src_folder = SRC_ROOT / category
        dst_folder = DST_ROOT / category
        ensure_dir(dst_folder)

        if not src_folder.exists():
            print(f"⚠️ Missing folder: {src_folder}")
            continue

        print(f"\n🔄 Processing category: {category}")

        for file in os.listdir(src_folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = src_folder / file
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Skipped invalid image: {file}")
                continue

            base = Path(file).stem

            # 1️⃣ CLAHE version
            clahe_img = apply_clahe(img)
            clahe_out = dst_folder / f"{base}_clahe.jpg"
            cv2.imwrite(str(clahe_out), clahe_img)

            # 2️⃣ Flipped CLAHE version
            flipped_img = flip_horizontal(clahe_img)
            flip_out = dst_folder / f"{base}_clahe_flip.jpg"
            cv2.imwrite(str(flip_out), flipped_img)

            total += 2

        print(f"✅ Done: {category} → saved in {dst_folder}")

    print(f"\n✨ Augmentation complete! Total new images: {total}")
    print(f"📁 Saved to: {DST_ROOT}")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    augment_images()
