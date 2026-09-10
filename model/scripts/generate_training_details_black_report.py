"""Generate an all-black Word report for the live pneumonia experiment.

The report is a presentation layer over the immutable dataset audit and the
actual run artifacts. It deliberately excludes smoke-test metrics from model
selection claims and never treats an in-progress run as a completed result.

Design system: ``standard_business_brief`` with an ``editorial_cover`` first
page. Named color override: every visible text and rule color is black; tables
use white or grayscale fills only. US Letter, 1-inch margins, 9360-DXA tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

from document_io import save_docx_with_fallback


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
RESULTS_ROOT = MODEL_ROOT / "results"
DOCS_ROOT = MODEL_ROOT / "docs"
DEFAULT_OUTPUT = DOCS_ROOT / "Pneumonia_Model_Training_Details_Black.docx"

BLACK = RGBColor(0, 0, 0)
WHITE_HEX = "FFFFFF"
LIGHT_GRAY_HEX = "EFEFEF"
MID_GRAY_HEX = "D9D9D9"
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
CELL_MARGIN_TOP_BOTTOM_DXA = 80
CELL_MARGIN_START_END_DXA = 120

MODEL_ORDER = [
    ("baseline_densenet121_s42", "DenseNet121"),
    ("baseline_densenet169_s42", "DenseNet169"),
    ("baseline_densenet201_s42", "DenseNet201"),
    ("baseline_resnet50_s42", "ResNet50"),
    ("baseline_resnet101_s42", "ResNet101"),
    ("baseline_resnet152_s42", "ResNet152"),
    ("baseline_yolov5_cls_s42", "YOLOv5 classification"),
    ("baseline_yolov8_cls_s42", "YOLOv8 classification"),
    ("baseline_yolov10_cls_s42", "YOLOv10 classification"),
]


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def set_run_font(run, *, size: float | None = None, bold: bool | None = None,
                 italic: bool | None = None, name: str = "Arial") -> None:
    run.font.name = name
    run.font.color.rgb = BLACK
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), name)
    rpr.rFonts.set(qn("w:hAnsi"), name)
    rpr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_style_font(style, size: float, *, bold: bool = False, name: str = "Arial") -> None:
    style.font.name = name
    style.font.size = Pt(size)
    style.font.color.rgb = BLACK
    style.font.bold = bold
    rpr = style._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), name)
    rpr.rFonts.set(qn("w:hAnsi"), name)
    rpr.rFonts.set(qn("w:eastAsia"), name)


def configure_styles(doc: Document) -> None:
    styles = doc.styles

    normal = styles["Normal"]
    set_style_font(normal, 10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    title = styles["Title"]
    set_style_font(title, 28, bold=True)
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(8)
    title.paragraph_format.line_spacing = 1.0

    subtitle = styles["Subtitle"]
    set_style_font(subtitle, 14)
    subtitle.paragraph_format.space_before = Pt(0)
    subtitle.paragraph_format.space_after = Pt(8)
    subtitle.paragraph_format.line_spacing = 1.0

    heading_tokens = {
        "Heading 1": (16, 16, 8),
        "Heading 2": (13, 12, 6),
        "Heading 3": (11.5, 8, 4),
    }
    for name, (size, before, after) in heading_tokens.items():
        style = styles[name]
        set_style_font(style, size, bold=True)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        set_style_font(style, 10.5)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.10
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.tab_stops.add_tab_stop(Inches(0.5))

    for name in ("Caption", "Header", "Footer"):
        if name in styles:
            set_style_font(styles[name], 8.5)

    if "Hyperlink" in styles:
        hyperlink = styles["Hyperlink"]
        hyperlink.font.color.rgb = BLACK
        hyperlink.font.underline = True


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    display = OxmlElement("w:t")
    display.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, display, end])
    set_run_font(run, size=8.5)


def paragraph_bottom_border(paragraph, *, color: str = "000000", size: str = "8") -> None:
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.append(pbdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)


def configure_sections(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.header_distance = Inches(0.492)
        section.footer_distance = Inches(0.492)
        section.different_first_page_header_footer = True

        header_p = section.header.paragraphs[0]
        header_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        header_p.paragraph_format.space_after = Pt(0)
        header_run = header_p.add_run("Pneumonia Three-Class Classification | Training Details")
        set_run_font(header_run, size=8.5, bold=True)
        paragraph_bottom_border(header_p, size="4")

        footer_p = section.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer_p.paragraph_format.space_before = Pt(0)
        footer_p.paragraph_format.space_after = Pt(0)
        prefix = footer_p.add_run("Page ")
        set_run_font(prefix, size=8.5)
        add_page_field(footer_p)


def shade_cell(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcpr.append(shd)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    tcmar = tcpr.find(qn("w:tcMar"))
    if tcmar is None:
        tcmar = OxmlElement("w:tcMar")
        tcpr.append(tcmar)
    for side, width in (
        ("top", CELL_MARGIN_TOP_BOTTOM_DXA),
        ("bottom", CELL_MARGIN_TOP_BOTTOM_DXA),
        ("start", CELL_MARGIN_START_END_DXA),
        ("end", CELL_MARGIN_START_END_DXA),
    ):
        elem = tcmar.find(qn(f"w:{side}"))
        if elem is None:
            elem = OxmlElement(f"w:{side}")
            tcmar.append(elem)
        elem.set(qn("w:w"), str(width))
        elem.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tblpr = table._tbl.tblPr
    borders = tblpr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblpr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")


def set_table_geometry(table, widths_dxa: Sequence[int]) -> None:
    if sum(widths_dxa) != TABLE_WIDTH_DXA:
        raise ValueError(f"Table widths must total {TABLE_WIDTH_DXA}: {widths_dxa}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tblpr = table._tbl.tblPr

    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:w"), str(TABLE_WIDTH_DXA))
    tblw.set(qn("w:type"), "dxa")

    tblind = tblpr.find(qn("w:tblInd"))
    if tblind is None:
        tblind = OxmlElement("w:tblInd")
        tblpr.append(tblind)
    tblind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tblind.set(qn("w:type"), "dxa")

    tbllayout = tblpr.find(qn("w:tblLayout"))
    if tbllayout is None:
        tbllayout = OxmlElement("w:tblLayout")
        tblpr.append(tbllayout)
    tbllayout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        trpr = row._tr.get_or_add_trPr()
        if trpr.find(qn("w:cantSplit")) is None:
            trpr.append(OxmlElement("w:cantSplit"))
        for idx, cell in enumerate(row.cells):
            width = widths_dxa[min(idx, len(widths_dxa) - 1)]
            tcpr = cell._tc.get_or_add_tcPr()
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.append(tcw)
            tcw.set(qn("w:w"), str(width))
            tcw.set(qn("w:type"), "dxa")
            cell.width = Inches(width / 1440)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
    set_table_borders(table)


def format_table_text(table, *, header: bool = True, font_size: float = 8.2) -> None:
    if header:
        trpr = table.rows[0]._tr.get_or_add_trPr()
        tbl_header = OxmlElement("w:tblHeader")
        tbl_header.set(qn("w:val"), "true")
        trpr.append(tbl_header)
    for ridx, row in enumerate(table.rows):
        for cell in row.cells:
            shade_cell(cell, LIGHT_GRAY_HEX if header and ridx == 0 else WHITE_HEX)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(1.5)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    set_run_font(run, size=font_size, bold=(header and ridx == 0))


def add_table(doc: Document, headers: Sequence[str], rows: Iterable[Sequence[object]],
              widths_dxa: Sequence[int], *, font_size: float = 8.2):
    rows = list(rows)
    table = doc.add_table(rows=1, cols=len(headers))
    for idx, value in enumerate(headers):
        table.rows[0].cells[idx].text = str(value)
    for values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            cells[idx].text = str(value)
    set_table_geometry(table, widths_dxa)
    format_table_text(table, font_size=font_size)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def add_body(doc: Document, text: str, *, bold_lead: str | None = None):
    paragraph = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, size=10.5, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest, size=10.5)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=10.5)
    return paragraph


def add_bullet(doc: Document, text: str):
    paragraph = doc.add_paragraph(style="List Bullet")
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5)
    return paragraph


def add_heading(doc: Document, text: str, level: int = 1):
    paragraph = doc.add_heading(text, level=level)
    for run in paragraph.runs:
        set_run_font(run, size={1: 16, 2: 13, 3: 11.5}[level], bold=True)
    return paragraph


def add_keep_note(doc: Document, text: str):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph_bottom_border(paragraph, size="6")
    run = paragraph.add_run(text)
    set_run_font(run, size=10, bold=True)
    return paragraph


def pct(value: float | str | None, digits: int = 2) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def dec(value: float | str | None, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def mb(value: float | str | None) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) / (1024 ** 2):.1f} MB"


def minutes(value: float | str | None) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) / 60:.1f} min"


def model_artifacts() -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    ledger = {row["run_id"]: row for row in load_csv_rows(RESULTS_ROOT / "experiment_ledger.csv")}
    matrix = {row["run_id"]: row for row in load_csv_rows(RESULTS_ROOT / "run_matrix.csv")}
    metrics: dict[str, dict] = {}
    for run_id, _ in MODEL_ORDER:
        path = RESULTS_ROOT / "runs" / run_id / "metrics.json"
        if path.exists():
            metrics[run_id] = load_json(path, {})
    return ledger, matrix, metrics


def status_for(run_id: str, ledger: dict[str, dict], matrix: dict[str, dict]) -> str:
    if run_id in ledger:
        return ledger[run_id].get("status", "unknown")
    if run_id in matrix:
        return matrix[run_id].get("status", "unknown")
    return "not listed"


def add_cover(doc: Document, snapshot: str, completed_count: int) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(112)

    kicker = doc.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(16)
    run = kicker.add_run("THREE-CLASS CHEST X-RAY RESEARCH REPORT")
    set_run_font(run, size=10, bold=True)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Pneumonia Model Training Details")
    for run in title.runs:
        set_run_font(run, size=28, bold=True)

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Data preparation, augmentation, training configuration, and verified validation results")
    for run in subtitle.runs:
        set_run_font(run, size=14)

    rule = doc.add_paragraph()
    rule.paragraph_format.space_before = Pt(12)
    rule.paragraph_format.space_after = Pt(24)
    paragraph_bottom_border(rule, size="12")

    summary = doc.add_paragraph()
    summary.alignment = WD_ALIGN_PARAGRAPH.CENTER
    summary.paragraph_format.space_after = Pt(72)
    sr = summary.add_run(f"{completed_count} completed full baseline models | Later research stages pending | Locked test set sealed")
    set_run_font(sr, size=11, bold=True)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(4)
    mr = meta.add_run(f"Snapshot generated {snapshot}")
    set_run_font(mr, size=10)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    nr = note.add_run("Research use only - not validated for clinical deployment")
    set_run_font(nr, size=9.5, bold=True)

    doc.add_page_break()


def add_executive_summary(doc: Document, completed: list[tuple[str, str, dict]],
                          ledger: dict[str, dict], matrix: dict[str, dict]) -> None:
    add_heading(doc, "1. Executive Summary", 1)
    add_body(
        doc,
        f"At this snapshot, {len(completed)} full baseline architectures have completed controlled validation training: "
        + ", ".join(name for _, name, _ in completed)
        + ". Smoke tests and the one-epoch pilot are excluded from this count because they only validate the pipeline.",
    )

    if completed:
        leader = max(completed, key=lambda item: item[2].get("macro_f1", -1))
        speed = max(completed, key=lambda item: item[2].get("inference", {}).get("throughput_images_per_second", -1))
        calibrated = min(completed, key=lambda item: item[2].get("expected_calibration_error", 999))
        total_training_hours = sum(item[2].get("training_duration_seconds", 0) for item in completed) / 3600
        add_bullet(doc, f"Best validation macro F1: {leader[1]} at {pct(leader[2]['macro_f1'])}.")
        add_bullet(doc, f"Fastest measured batch-1 inference: {speed[1]} at {speed[2]['inference']['throughput_images_per_second']:.1f} images/second.")
        add_bullet(doc, f"Lowest expected calibration error among completed baselines: {calibrated[1]} at {pct(calibrated[2]['expected_calibration_error'])}.")
        add_bullet(doc, f"Recorded training time across the eight completed baselines: {total_training_hours:.2f} GPU-hours, executed sequentially on the RTX 2050.")
        add_bullet(doc, "The most difficult class is Viral Pneumonia. Completed models achieve much stronger recall for Normal and Bacterial cases than for Viral cases.")

    res152 = ledger.get("baseline_resnet152_s42", {})
    if res152.get("status") == "running":
        add_bullet(
            doc,
            f"ResNet152 remains in progress at epoch {res152.get('best_epoch', '-')}/12 for best-checkpoint selection; its provisional macro F1 is {pct(res152.get('best_validation_macro_f1'))} and is not a final result.",
        )
    yolo5_state = status_for("baseline_yolov5_cls_s42", ledger, matrix)
    yolo8_state = status_for("baseline_yolov8_cls_s42", ledger, matrix)
    if yolo5_state == "completed" and yolo8_state == "completed":
        add_bullet(doc, "YOLOv5 and YOLOv8 classification baselines completed successfully. Both were faster and smaller than the torchvision CNNs, but both produced lower validation macro F1.")
    else:
        add_bullet(doc, f"YOLO classification status: YOLOv5 is {yolo5_state}; YOLOv8 is {yolo8_state}.")
    add_bullet(doc, "YOLOv10 classification is unsupported because the installed Ultralytics release has no genuine YOLOv10 classification head/checkpoint.")
    add_bullet(doc, "The full baseline stage is complete. Preprocessing ablations, focused comparison, Optuna tuning, multi-seed robustness, and final locked-test evaluation have not run.")
    add_keep_note(doc, "Result boundary: every number in the model comparison tables comes from completed validation metrics.json artifacts. No locked-test result is reported or used for selection.")

    add_heading(doc, "Status of Requested Model Families", 2)
    rows = []
    for run_id, display_name in MODEL_ORDER:
        state = status_for(run_id, ledger, matrix)
        evidence = "Final validation metrics available" if run_id in {x[0] for x in completed} else ""
        if state == "running":
            row = ledger.get(run_id, {})
            evidence = f"Provisional best epoch {row.get('best_epoch', '-')}; not final"
        elif state == "planned":
            evidence = matrix.get(run_id, {}).get("reason", "Queued for later stage")
        elif state == "unsupported":
            evidence = matrix.get(run_id, {}).get("reason", "Unsupported")
        rows.append((display_name, state.title(), evidence))
    add_table(doc, ("Architecture", "Status", "Evidence / boundary"), rows, (2600, 1200, 5560), font_size=8.4)

    add_heading(doc, "Overall Research Pipeline Progress", 2)
    controlled_baselines = sum(
        1 for run_id, _ in MODEL_ORDER
        if run_id != "baseline_yolov10_cls_s42" and status_for(run_id, ledger, matrix) == "completed"
    )
    diagnostic_completed = sum(
        1 for run_id, row in ledger.items()
        if row.get("status") == "completed" and not run_id.startswith("baseline_")
    )
    stage_rows = [
        ("Repository, environment, and dataset audit", "Complete", "Immutable source verified; locked manifests created"),
        ("Pipeline implementation and diagnostics", "Complete", f"{diagnostic_completed} completed pilot/smoke runs; one earlier YOLOv8 smoke failure was corrected by successful retries"),
        ("Supported baseline benchmark", "Complete", f"{controlled_baselines}/8 controlled baselines completed"),
        ("Preprocessing ablation", "Not started", "0/3 supported full ablations completed"),
        ("Focused comparison and Optuna tuning", "Not started", "No focused run or tuning trial exists"),
        ("Multi-seed robustness", "Not started", "Seeds 42, 123, and 2026 remain planned"),
        ("Locked-test evaluation", "Sealed", "No test metric exists; test data has not been used for selection"),
    ]
    add_table(doc, ("Stage", "Status", "Actual evidence"), stage_rows, (3000, 1500, 4860), font_size=8.2)
    add_keep_note(doc, "Automation boundary: after the baseline stage, the orchestrator stopped because Pneumonia_Research_Progress_Report.docx could not be overwritten while locked. No later experiment was silently executed.")


def add_data_prep(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "2. Data Preparation", 1)
    add_body(doc, "The source dataset was audited without modifying the original chest_xray directory. Every readable image was inventoried, labeled from folder and filename evidence, hashed, grouped conservatively, and assigned to a locked split manifest.")

    add_heading(doc, "Source Inventory and Retention", 2)
    rows = [
        ("Physical files in source tree", "17,591", "2,483,915,296 bytes; source treated as read-only"),
        ("Readable canonical images", "5,856", "Images that passed the audit reader"),
        ("Redundant exact duplicates excluded", "32", "Exact duplicate content removed from the retained set"),
        ("Unique images retained", "5,824", "Basis of the locked train/validation/test manifests"),
        ("Unreadable retained images", "0", "No invalid image was allowed into a manifest"),
        ("Source integrity recheck", "17,591 / 17,591", "Zero hash mismatches after the audit"),
    ]
    add_table(doc, ("Audit item", "Value", "Interpretation"), rows, (3200, 1600, 4560), font_size=8.5)

    add_heading(doc, "Three-Class Label Derivation", 2)
    add_body(doc, "Labels were derived only from repository-visible folder and filename evidence. Pixel appearance was never used to assign a class.")
    rows = [
        ("0", "Normal", "NORMAL source folder", "1,579", "27.11%"),
        ("1", "Viral Pneumonia", "PNEUMONIA folder plus _virus_ filename token", "1,485", "25.50%"),
        ("2", "Bacterial Pneumonia", "PNEUMONIA folder plus _bacteria_ filename token", "2,760", "47.39%"),
    ]
    add_table(doc, ("Index", "Class", "Label evidence", "Retained", "Share"), rows, (700, 1900, 3860, 1400, 1500), font_size=8.3)

    add_heading(doc, "Why the Original Split Was Rebuilt", 2)
    add_body(doc, "The original validation folder was not suitable for three-class model selection: it contained 8 Normal and 8 Bacterial images but no Viral images. The audit also found 264 conservative group identifiers occurring across original folders. A fixed, stratified, group-aware split was therefore built with seed 42.")

    rows = [
        ("Train", "4,076", "1,105", "1,039", "1,932", "2,326"),
        ("Validation", "874", "237", "223", "414", "228"),
        ("Locked test", "874", "237", "223", "414", "235"),
        ("Total", "5,824", "1,579", "1,485", "2,760", "2,789"),
    ]
    add_table(doc, ("Split", "Total", "Normal", "Viral", "Bacterial", "Groups"), rows, (1500, 1300, 1400, 1400, 1600, 2160), font_size=8.4)
    add_bullet(doc, "Exact-hash leakage across assigned splits: 0 pairs.")
    add_bullet(doc, "Conservative group leakage across assigned splits: 0 groups.")
    add_bullet(doc, "Manifest version: three_class_seed42_3b815cf769c8.")
    add_bullet(doc, "Manifest SHA-256: 3b815cf769c85658a04748da5eb2e15b981b89d9c9b77a404e90fe23473d2774.")

    add_heading(doc, "Image Characteristics and Quality Checks", 2)
    rows = [
        ("Image mode", "5,541 grayscale (L); 283 RGB"),
        ("Width", "Minimum 384 px; median 1,280 px; maximum 2,916 px"),
        ("Height", "Minimum 127 px; median 886 px; maximum 2,713 px"),
        ("Aspect ratio", "Minimum 0.83539; median 1.41608; maximum 3.37879"),
        ("Near-duplicate review", "16 perceptual-hash candidate pairs retained for manual review"),
    ]
    add_table(doc, ("Characteristic", "Observed distribution"), rows, (2500, 6860), font_size=8.7)
    add_keep_note(doc, "Grouping limitation: the dataset does not provide authoritative patient metadata. Conservative filename-derived group identifiers reduce leakage risk but may merge unrelated images when identifiers are reused.")


def add_preprocessing(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "3. Preprocessing and Data Augmentation", 1)
    add_heading(doc, "Baseline Image Pipeline", 2)
    steps = [
        "Open the image with Pillow and convert it to RGB so grayscale and RGB files have a consistent three-channel tensor.",
        "Center-fit the image to 224 x 224 pixels with bilinear interpolation.",
        "Convert pixel values to the [0, 1] range.",
        "Normalize with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].",
    ]
    for step in steps:
        add_bullet(doc, step)

    add_heading(doc, "Training-Only Augmentation", 2)
    rows = [
        ("Random horizontal flip", "Probability 0.50", "Adds left-right variation while preserving the requested conservative policy"),
        ("Random rotation", "Uniform from -5 to +5 degrees", "Bilinear interpolation with black fill"),
        ("Validation/test transforms", "Deterministic", "No random augmentation during evaluation"),
    ]
    add_table(doc, ("Operation", "Setting", "Use"), rows, (2600, 2400, 4360), font_size=8.6)
    add_body(doc, "No random color jitter, MixUp, CutMix, aggressive crop policy, or large geometric distortion was applied. This keeps the baseline easy to interpret and prevents augmentation changes from becoming a hidden model-selection variable.")

    add_heading(doc, "Implemented Preprocessing Ablations", 2)
    rows = [
        ("CLAHE", "Implemented; full ablation planned", "Grayscale CLAHE, clip limit 2.0, 8 x 8 tile grid, then RGB replication"),
        ("Histogram equalization", "Implemented; full ablation planned", "Global grayscale histogram equalization, then RGB replication"),
        ("Pseudo-coloring", "Implemented; full ablation planned", "OpenCV VIRIDIS mapping; changes representation but adds no clinical information"),
        ("PCA", "Unsupported / deferred", "Raw-image PCA was not considered scientifically justified; no train-only feature-space design exists"),
        ("FILM", "Unresolved", "No verified repository definition or conditioning signal was found; no definition was invented"),
    ]
    add_table(doc, ("Concept", "Status", "Verified implementation or reason"), rows, (1800, 2200, 5360), font_size=8.2)
    add_keep_note(doc, "Smoke runs confirm that the three implemented preprocessing paths execute end to end. Their one-epoch smoke metrics are diagnostics only and are not evidence that one preprocessing method is better.")


def add_training(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "4. Training Design", 1)
    add_heading(doc, "Common Configuration", 2)
    rows = [
        ("Task", "Three-class image classification"),
        ("Class order", "Normal, Viral Pneumonia, Bacterial Pneumonia"),
        ("Classifier output", "Three logits"),
        ("Initialization", "Official ImageNet weights for completed torchvision baselines"),
        ("Fine-tuning", "Full network trainable; freeze_backbone = false"),
        ("Maximum epochs", "12"),
        ("Optimizer", "AdamW"),
        ("Learning rate", "0.0001"),
        ("Weight decay", "0.0001"),
        ("Loss", "Class-weighted cross-entropy"),
        ("Class weights", "Normal 1.229563; Viral 1.307668; Bacterial 0.703244"),
        ("Scheduler", "ReduceLROnPlateau on validation macro F1; patience 2; factor 0.5"),
        ("Early stopping", "Patience 4; minimum delta 0.0001"),
        ("Mixed precision", "Enabled"),
        ("Seed / determinism", "Seed 42; deterministic settings where practical"),
        ("DataLoader workers", "0 for Windows reliability"),
        ("Selection metric", "Validation macro F1"),
        ("Checkpoints", "Best and last checkpoints retained; history saved every epoch"),
    ]
    add_table(doc, ("Configuration item", "Value"), rows, (3000, 6360), font_size=8.4)

    add_heading(doc, "Architecture-Specific Batch Sizes", 2)
    rows = [
        ("DenseNet121", "16"), ("DenseNet169", "12"), ("DenseNet201", "8"),
        ("ResNet50", "16"), ("ResNet101", "8"), ("ResNet152", "6"),
        ("YOLOv5 classification", "32"), ("YOLOv8 classification", "32"),
    ]
    add_table(doc, ("Architecture", "Batch size"), rows, (4680, 4680), font_size=8.6)

    add_heading(doc, "Hardware and Software Snapshot", 2)
    rows = [
        ("GPU", "NVIDIA GeForce RTX 2050; 4,096 MiB VRAM"),
        ("Python", "3.12.6"),
        ("PyTorch", "2.5.1+cu121"),
        ("torchvision", "0.20.1+cu121"),
        ("CUDA runtime", "12.1"),
        ("cuDNN", "90100"),
        ("Ultralytics", "8.3.168"),
        ("Optuna", "4.9.0"),
    ]
    add_table(doc, ("Component", "Recorded value"), rows, (3000, 6360), font_size=8.7)
    add_body(doc, "General tuning is planned as an Optuna TPE validation-only search. It is not called SAS Tuning. SAS Tuning and automated grading of chest X-rays remain unresolved because the repository contains no verified implementation or exact definition for either term.")


def add_results(doc: Document, completed: list[tuple[str, str, dict]]) -> None:
    doc.add_page_break()
    add_heading(doc, "5. Completed Baseline Results", 1)
    add_body(doc, "All results in this section are from the best checkpoint of a completed baseline run evaluated on the same 874-image validation manifest. Percentages are rounded for presentation; ranking calculations used the full-precision artifacts.")

    rows = []
    for _, name, m in completed:
        rows.append((
            name,
            m.get("best_epoch", "-"),
            pct(m.get("accuracy")),
            pct(m.get("balanced_accuracy")),
            pct(m.get("macro_f1")),
            pct(m.get("weighted_f1")),
        ))
    add_table(doc, ("Model", "Best epoch", "Accuracy", "Balanced acc.", "Macro F1", "Weighted F1"), rows, (2100, 1160, 1500, 1700, 1450, 1450), font_size=7.9)

    add_heading(doc, "Per-Class Precision, Recall, and F1", 2)
    rows = []
    for _, name, m in completed:
        for class_name in ("Normal", "Viral", "Bacterial"):
            values = m["per_class"][class_name]
            rows.append((name, class_name, pct(values["precision"]), pct(values["recall"]), pct(values["f1"]), values["support"]))
    add_table(doc, ("Model", "Class", "Precision", "Recall", "F1", "Support"), rows, (2100, 1500, 1500, 1500, 1400, 1360), font_size=7.8)

    add_heading(doc, "Discrimination, Calibration, and Agreement", 2)
    rows = []
    for _, name, m in completed:
        rows.append((
            name,
            dec(m.get("roc_auc_ovr_macro")),
            dec(m.get("pr_auc_macro")),
            dec(m.get("log_loss")),
            dec(m.get("expected_calibration_error")),
            dec(m.get("brier_score")),
            dec(m.get("matthews_correlation_coefficient")),
            dec(m.get("cohen_kappa")),
        ))
    add_table(doc, ("Model", "ROC AUC", "PR AUC", "Log loss", "ECE", "Brier", "MCC", "Kappa"), rows, (1800, 1080, 1080, 1080, 900, 1080, 1140, 1200), font_size=7.2)

    add_heading(doc, "Efficiency and Resource Use", 2)
    rows = []
    for _, name, m in completed:
        inference = m.get("inference", {})
        rows.append((
            name,
            f"{m.get('parameter_count', 0):,}",
            mb(m.get("model_file_size_bytes")),
            mb(m.get("peak_gpu_memory_bytes")),
            minutes(m.get("training_duration_seconds")),
            f"{inference.get('latency_ms_per_image', 0):.2f} ms",
            f"{inference.get('throughput_images_per_second', 0):.1f}/s",
        ))
    add_table(doc, ("Model", "Parameters", "Artifact", "Peak GPU", "Train time", "Latency", "Throughput"), rows, (1800, 1600, 1350, 1450, 1300, 950, 910), font_size=7.4)
    add_body(doc, "Inference was measured on CUDA with batch size 1, input size 224, 5 warm-up runs, and 20 measured runs. Image preprocessing and host-to-device transfer were excluded from the latency measurement.")


def add_confusions(doc: Document, completed: list[tuple[str, str, dict]]) -> None:
    doc.add_page_break()
    add_heading(doc, "6. Confusion Matrices", 1)
    add_body(doc, "Rows are actual classes and columns are predicted classes. Each matrix sums to 874 validation images. The pattern common to all completed models is strong Normal recognition and the largest error flow from Viral to Bacterial.")
    for _, name, metrics in completed:
        add_heading(doc, name, 2)
        cm = metrics["confusion_matrix"]
        rows = [
            ("Actual Normal", cm[0][0], cm[0][1], cm[0][2]),
            ("Actual Viral", cm[1][0], cm[1][1], cm[1][2]),
            ("Actual Bacterial", cm[2][0], cm[2][1], cm[2][2]),
        ]
        add_table(doc, ("Actual / predicted", "Normal", "Viral", "Bacterial"), rows, (3000, 2000, 2000, 2360), font_size=8.7)


def add_interpretation(doc: Document, completed: list[tuple[str, str, dict]],
                       ledger: dict[str, dict]) -> None:
    doc.add_page_break()
    add_heading(doc, "7. Interpretation", 1)
    if completed:
        macro_leader = max(completed, key=lambda x: x[2]["macro_f1"])
        balanced_leader = max(completed, key=lambda x: x[2]["balanced_accuracy"])
        viral_recall_leader = max(completed, key=lambda x: x[2]["per_class"]["Viral"]["recall"])
        speed_leader = max(completed, key=lambda x: x[2]["inference"]["throughput_images_per_second"])
        add_bullet(doc, f"{macro_leader[1]} currently leads the primary selection metric, validation macro F1, at {pct(macro_leader[2]['macro_f1'])}.")
        add_bullet(doc, f"{balanced_leader[1]} has the highest validation balanced accuracy at {pct(balanced_leader[2]['balanced_accuracy'])}.")
        add_bullet(doc, f"{viral_recall_leader[1]} has the strongest Viral recall at {pct(viral_recall_leader[2]['per_class']['Viral']['recall'])}, but its overall accuracy is {pct(viral_recall_leader[2]['accuracy'])}.")
        add_bullet(doc, f"{speed_leader[1]} is the strongest efficiency candidate: {speed_leader[2]['inference']['throughput_images_per_second']:.1f} images/second and {speed_leader[2]['inference']['latency_ms_per_image']:.2f} ms/image in the controlled timing setup.")
        add_bullet(doc, "DenseNet169 and DenseNet201 are separated by less than 0.2 percentage points in macro F1, so the ordering should not be over-interpreted before preprocessing ablations, tuning, and seed robustness runs finish.")
        total_hours = sum(item[2].get("training_duration_seconds", 0) for item in completed) / 3600
        add_bullet(doc, f"The eight baseline runs recorded {total_hours:.2f} cumulative GPU training hours. This is experiment time, not a clinical-performance measure.")
    add_body(doc, "Class-specific behavior matters more than accuracy alone. Bacterial cases are the largest class, and all completed models show high Bacterial recall. Viral cases have lower recall and are frequently predicted as Bacterial, which depresses the macro F1 and balanced accuracy despite strong Normal performance.")

    add_heading(doc, "What Has and Has Not Been Established", 2)
    rows = [
        ("Established", "The multiclass pipeline trains reproducibly on a locked, leakage-audited manifest using the available NVIDIA GPU."),
        ("Established", "Eight completed architecture baselines have directly comparable validation artifacts."),
        ("Not established", "A final selected architecture or preprocessing policy; the experiment matrix is still incomplete."),
        ("Not established", "Locked-test generalization; the test set remains reserved for the final frozen evaluation."),
        ("Not established", "Clinical usefulness, fairness, robustness across institutions, or regulatory readiness."),
    ]
    add_table(doc, ("Boundary", "Meaning"), rows, (2100, 7260), font_size=8.7)

    res = ledger.get("baseline_resnet152_s42", {})
    if res.get("status") == "running":
        add_heading(doc, "In-Progress ResNet152 Snapshot", 2)
        add_body(doc, f"ResNet152 is recorded as running. The latest ledger entry reports best epoch {res.get('best_epoch', '-')} and provisional validation macro F1 {pct(res.get('best_validation_macro_f1'))}. Because the run has no finished_at value or final metrics.json artifact, it is excluded from all completed-model tables and rankings.")


def add_limitations_next(doc: Document) -> None:
    doc.add_page_break()
    add_heading(doc, "8. Limitations and Remaining Work", 1)
    limitations = [
        "This is a single-dataset internal validation study. Distribution shift across hospitals, devices, acquisition protocols, demographics, and disease prevalence has not been measured.",
        "The dataset lacks authoritative patient metadata; conservative filename grouping reduces but cannot prove the absence of patient-level leakage.",
        "The classes come from folder and filename labels, not a new radiologist adjudication process. Label noise and ambiguous viral-versus-bacterial appearance are possible.",
        "The current table is validation-only. It does not estimate final locked-test performance.",
        "Confidence calibration is measured internally but has not been externally reviewed or calibrated for a deployment population.",
        "No clinical workflow, bias, safety, regulatory, or medical-device validation has been performed.",
    ]
    for item in limitations:
        add_bullet(doc, item)

    add_heading(doc, "Planned Experimental Sequence", 2)
    rows = [
        ("1", "Run preprocessing ablations", "CLAHE, histogram equalization, and exploratory VIRIDIS on DenseNet169, DenseNet201, and ResNet50; reuse the completed controls"),
        ("2", "Focused comparison", "Combine the top validation architecture candidates with the best supported preprocessing policy"),
        ("3", "General tuning", "Optuna TPE using validation metrics only; retain completed, pruned, and failed trials"),
        ("4", "Seed robustness", "Retrain the frozen selected configuration with seeds 42, 123, and 2026"),
        ("5", "Single locked-test evaluation", "Unseal the locked test only after architecture, preprocessing, and tuning choices are frozen"),
        ("6", "Final reporting", "Regenerate the final Markdown and Word reports from the completed source artifacts"),
    ]
    add_table(doc, ("Stage", "Work", "Selection boundary"), rows, (900, 2800, 5660), font_size=8.4)
    add_keep_note(doc, "Clinical-use statement: the classifier is a research system. External clinical validation, broader datasets, bias assessment, calibration review, and regulatory/medical review are required before any clinical use.")


def add_reproduction(doc: Document, snapshot: str) -> None:
    add_heading(doc, "9. Reproducibility and Artifact Map", 1)
    add_body(doc, "The following repository artifacts are the sources of truth for this report. Paths are relative to the repository root so the report remains portable.")
    rows = [
        ("Locked manifests", "model/data_manifests/train_manifest.csv; val_manifest.csv; test_manifest.csv"),
        ("Class mapping", "model/data_manifests/class_mapping.json"),
        ("Split summary", "model/data_manifests/split_summary.json"),
        ("Experiment ledger", "model/results/experiment_ledger.csv and experiment_ledger.json"),
        ("Run matrix", "model/results/run_matrix.csv"),
        ("Run metrics", "model/results/runs/<run_id>/metrics.json"),
        ("Epoch history", "model/results/runs/<run_id>/history.csv"),
        ("Best / last checkpoints", "model/results/runs/<run_id>/best_checkpoint.pt and last_checkpoint.pt"),
        ("Continuous documentation", "model/docs/CURRENT_STATUS.md; DEVELOPMENT_LOG.md; TRAINING_JOURNAL.md; ISSUES_AND_FIXES.md"),
    ]
    add_table(doc, ("Artifact", "Repository-relative location"), rows, (2600, 6760), font_size=8.2)
    add_body(doc, f"Report snapshot: {snapshot}. Values should be refreshed after additional runs complete. The source Markdown, CSV, and JSON artifacts remain authoritative if the live experiment changes after this document was generated.")


def force_visible_colors_black(path: Path) -> None:
    """Patch all Word text/rule colors to black while retaining grayscale fills."""
    with zipfile.ZipFile(path, "r") as source, NamedTemporaryFile(delete=False, suffix=".docx") as temp:
        temp_path = Path(temp.name)
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename.startswith("word/") and info.filename.endswith(".xml"):
                    try:
                        root = parse_xml(data)
                        for element in root.iter(qn("w:color")):
                            if element.get(qn("w:val")) not in (None, "auto"):
                                element.set(qn("w:val"), "000000")
                            for attr in ("themeColor", "themeTint", "themeShade"):
                                element.attrib.pop(qn(f"w:{attr}"), None)
                        for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
                            for element in root.iter(qn(f"w:{tag}")):
                                if element.get(qn("w:val")) not in (None, "nil", "none"):
                                    element.set(qn("w:color"), "000000")
                        for element in root.iter(qn("w:shd")):
                            fill = element.get(qn("w:fill"), "auto").upper()
                            if fill not in {"AUTO", "NONE", WHITE_HEX, LIGHT_GRAY_HEX, MID_GRAY_HEX, "000000"}:
                                element.set(qn("w:fill"), LIGHT_GRAY_HEX)
                                element.attrib.pop(qn("w:themeFill"), None)
                                element.attrib.pop(qn("w:themeFillTint"), None)
                                element.attrib.pop(qn("w:themeFillShade"), None)
                        data = etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
                    except Exception:
                        pass
                target.writestr(info, data)
    temp_path.replace(path)


def audit_docx(path: Path) -> dict[str, object]:
    doc = Document(path)
    findings: dict[str, object] = {
        "path": str(path),
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "sections": len(doc.sections),
        "headings": sum(1 for p in doc.paragraphs if p.style and p.style.name.startswith("Heading")),
    }
    geometry_errors: list[str] = []
    for idx, table in enumerate(doc.tables, start=1):
        tblw = table._tbl.tblPr.find(qn("w:tblW"))
        tblind = table._tbl.tblPr.find(qn("w:tblInd"))
        grid = [int(c.get(qn("w:w"), "0")) for c in table._tbl.tblGrid]
        if tblw is None or int(tblw.get(qn("w:w"), "0")) != TABLE_WIDTH_DXA:
            geometry_errors.append(f"table {idx}: tblW")
        if tblind is None or int(tblind.get(qn("w:w"), "0")) != TABLE_INDENT_DXA:
            geometry_errors.append(f"table {idx}: tblInd")
        if sum(grid) != TABLE_WIDTH_DXA:
            geometry_errors.append(f"table {idx}: grid sum {sum(grid)}")
        for ridx, row in enumerate(table.rows):
            widths = []
            for cell in row.cells:
                tcw = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
                widths.append(int(tcw.get(qn("w:w"), "0")) if tcw is not None else 0)
            if widths != grid:
                geometry_errors.append(f"table {idx} row {ridx + 1}: tcW {widths} vs grid {grid}")
                break
    findings["table_geometry_errors"] = geometry_errors

    disallowed_colors: list[str] = []
    allowed_fills = {"auto", WHITE_HEX, LIGHT_GRAY_HEX, MID_GRAY_HEX, "000000"}
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        findings["zip_test"] = "ok" if bad is None else bad
        for name in archive.namelist():
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            try:
                root = parse_xml(archive.read(name))
            except Exception:
                continue
            for element in root.iter(qn("w:color")):
                value = element.get(qn("w:val"), "auto").upper()
                if value not in {"AUTO", "000000"}:
                    disallowed_colors.append(f"{name}: text/rule color {value}")
            for element in root.iter(qn("w:shd")):
                fill = element.get(qn("w:fill"), "auto").upper()
                if fill not in {x.upper() for x in allowed_fills}:
                    disallowed_colors.append(f"{name}: fill {fill}")
    findings["disallowed_visible_colors"] = disallowed_colors

    section = doc.sections[0]
    findings["page_setup"] = {
        "width_in": round(section.page_width.inches, 3),
        "height_in": round(section.page_height.inches, 3),
        "margins_in": [
            round(section.top_margin.inches, 3),
            round(section.right_margin.inches, 3),
            round(section.bottom_margin.inches, 3),
            round(section.left_margin.inches, 3),
        ],
    }
    findings["pass"] = not geometry_errors and not disallowed_colors and findings["zip_test"] == "ok"
    return findings


def build_report(output: Path) -> dict[str, object]:
    ledger, matrix, metrics = model_artifacts()
    completed: list[tuple[str, str, dict]] = []
    for run_id, display_name in MODEL_ORDER:
        if status_for(run_id, ledger, matrix) == "completed" and run_id in metrics:
            completed.append((run_id, display_name, metrics[run_id]))

    snapshot = datetime.now().astimezone().strftime("%B %d, %Y at %I:%M %p %Z")
    doc = Document()
    configure_styles(doc)
    configure_sections(doc)
    doc.core_properties.title = "Pneumonia Model Training Details"
    doc.core_properties.subject = "Three-class chest X-ray data preparation, augmentation, training, and results"
    doc.core_properties.author = "Pneumonia Research Pipeline"
    doc.core_properties.keywords = "pneumonia, chest X-ray, DenseNet, ResNet, validation, multiclass"

    add_cover(doc, snapshot, len(completed))
    add_executive_summary(doc, completed, ledger, matrix)
    add_data_prep(doc)
    add_preprocessing(doc)
    add_training(doc)
    add_results(doc, completed)
    add_confusions(doc, completed)
    add_interpretation(doc, completed, ledger)
    add_limitations_next(doc)
    add_reproduction(doc, snapshot)

    saved_output = save_docx_with_fallback(doc, output)
    force_visible_colors_black(saved_output)
    audit = audit_docx(saved_output)
    audit["completed_full_baselines"] = len(completed)
    audit["completed_run_ids"] = [run_id for run_id, _, _ in completed]
    audit["requested_output"] = str(output)
    audit["saved_output"] = str(saved_output)
    audit_path = saved_output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["pass"]:
        raise RuntimeError(f"DOCX structural audit failed; see {audit_path}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_report(args.output.resolve())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
