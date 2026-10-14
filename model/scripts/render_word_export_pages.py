"""Use the canonical DOCX rasterizer with a verified native Word PDF export.

Windows lacks bundled LibreOffice. Only the PDF-conversion function is replaced;
the bundled renderer still selects DPI; bundled PDFium produces the page PNGs.
"""
import argparse
import importlib.util
import shutil
from pathlib import Path

import pypdfium2 as pdfium

parser = argparse.ArgumentParser()
parser.add_argument("--renderer", type=Path, required=True)
parser.add_argument("--docx", type=Path, required=True)
parser.add_argument("--pdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not args.pdf.is_file() or args.pdf.stat().st_size == 0:
    raise SystemExit("A completed Word PDF export is required")
spec = importlib.util.spec_from_file_location("canonical_renderer", args.renderer)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def convert_word_export(input_path, user_profile, convert_tmp_dir, stem, verbose=False):
    if Path(input_path).resolve() != args.docx.resolve():
        raise RuntimeError("Word export does not match the requested DOCX")
    target = Path(convert_tmp_dir) / (stem + ".pdf")
    shutil.copy2(args.pdf, target)
    return str(target), "Native Microsoft Word PDF export supplied; no LibreOffice was invoked."


renderer.convert_to_pdf = convert_word_export


def rasterize_pdfium(pdf_path, dpi, output_folder, **kwargs):
    paths = []
    with pdfium.PdfDocument(pdf_path) as document:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=dpi / 72)
            path = Path(output_folder) / f"page0000-{index+1}.png"
            bitmap.to_pil().save(path)
            paths.append(str(path))
            bitmap.close()
            page.close()
    return paths


# Use the selected runtime's bundled PDFium; no system Poppler is required.
renderer.convert_from_path = rasterize_pdfium
dpi = renderer.calc_dpi_via_ooxml_docx(str(args.docx), 1600, 2000)
renderer.rasterize(str(args.docx), str(args.output), dpi, verbose=True, emit_pdf=False)
print("Verified Word export page PNGs: " + str(args.output))
