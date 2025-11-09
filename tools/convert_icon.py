# tools/convert_icon.py
# Convert PNG to multi-size ICO for Windows shortcuts.
# Requires: pip install pillow

from PIL import Image
from pathlib import Path

# Input PNG (update path if needed)
png_path = Path(__file__).resolve().parents[1] / "assets" / "icons" / "lungai.png"
ico_path = png_path.with_suffix(".ico")

if not png_path.exists():
    raise FileNotFoundError(f"PNG not found: {png_path}")

img = Image.open(png_path)
img.save(ico_path, format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
print(f"✅ Converted to ICO: {ico_path}")
