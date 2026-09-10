"""Rewrite embedded PNGs in a DOCX as opaque RGB images.

Microsoft Word can stall while exporting some python-docx packages that embed
large RGBA plots.  Flattening those plots against white preserves their visible
appearance while removing the alpha channel that triggers the export problem.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def flatten_png(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        if image.mode == "RGB":
            return data
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        output = io.BytesIO()
        background.save(output, format="PNG", optimize=True)
        return output.getvalue()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    with zipfile.ZipFile(source, "r") as incoming, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as outgoing:
        for item in incoming.infolist():
            data = incoming.read(item.filename)
            if item.filename.startswith("word/media/") and item.filename.lower().endswith(".png"):
                flattened = flatten_png(data)
                converted += int(flattened != data)
                data = flattened
            outgoing.writestr(item, data)
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"normalized DOCX failed ZIP integrity at {bad}")
    print(f"Normalized {converted} embedded PNGs: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
