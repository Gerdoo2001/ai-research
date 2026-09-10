"""Patch the Word-normalized discussion guide without reserializing it via python-docx."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

from lxml import etree


REPO = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO
    / "model"
    / "docs"
    / "Pneumonia_Classification_Discussion_Notes_and_Overview_20260812_repaired.docx"
)
OUTPUT = (
    REPO
    / "model"
    / "docs"
    / "Pneumonia_Classification_Discussion_Notes_and_Overview_20260812.docx"
)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def qn(local: str) -> str:
    return f"{{{W}}}{local}"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def paragraph_text(paragraph) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def replace_all_text(paragraph, text: str) -> None:
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not nodes:
        return
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


def replace_after_label(paragraph, label: str, body: str) -> None:
    nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if len(nodes) < 2:
        replace_all_text(paragraph, f"{label} {body}")
        return
    nodes[0].text = label
    nodes[1].text = " " + body
    for node in nodes[2:]:
        node.text = ""


def set_list_bullet(paragraph) -> None:
    ppr = paragraph.find(qn("pPr"))
    if ppr is None:
        ppr = etree.Element(qn("pPr"))
        paragraph.insert(0, ppr)
    style = ppr.find(qn("pStyle"))
    if style is None:
        style = etree.Element(qn("pStyle"))
        ppr.insert(0, style)
    style.set(qn("val"), "ListBullet")
    num_pr = ppr.find(qn("numPr"))
    if num_pr is not None:
        ppr.remove(num_pr)


def update_document(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    status = load_json(
        REPO
        / "model"
        / "results"
        / "runs"
        / "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s123"
        / "status.json"
    )
    live = load_json(
        REPO
        / "model"
        / "results"
        / "runs"
        / "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s123"
        / "live_progress.json"
    )
    current_status = (
        "FT-010 is running at seed 123, with "
        f"{status.get('current_epoch', 0)}/12 completed epochs and active epoch "
        f"{live.get('epoch', '-')}/12. Seed 2026 follows automatically."
    )

    overview = (
        "We created a leakage-controlled pediatric chest-X-ray classifier for Normal, Viral, and "
        "Bacterial cases. After auditing duplicates and locking group-aware splits, we compared "
        "eight pretrained architectures and every ordered VIRIDIS, GE, and CLAHE combination for "
        "the leading models. DenseNet-201 with VIRIDIS -> GE -> CLAHE became the selected validation "
        "pipeline. Fine-tuning then tested resolution, augmentation, ensembles, alternative losses, "
        "lung segmentation, and hierarchical classification. The label-smoothed DenseNet-201 is the "
        "best observed single seed-42 model, while FT-004 is the strongest multi-seed confirmed "
        "ensemble. FT-010 is checking whether the label-smoothing gain repeats at seeds 123 and 2026."
    )
    script = {
        "Opening:": (
            "Our project evaluates three-class pediatric pneumonia classification: Normal, Viral, "
            "and Bacterial. The emphasis is fair measurement and leakage prevention, not accuracy alone."
        ),
        "Data preparation:": (
            "We audited 17,591 files, retained 5,824 unique canonical images, and created group-aware "
            "training, validation, and test assignments. Eight pretrained classifiers were compared "
            "under shared controls."
        ),
        "Controlled comparison:": (
            "DenseNet-201 and ResNet-50 received all 16 ordered VIRIDIS, GE, and CLAHE conditions. "
            "DenseNet-201 with VIRIDIS -> GE -> CLAHE ranked first on three-seed mean validation macro F1."
        ),
        "Fine-tuning:": (
            "Higher resolution and added affine augmentation did not improve macro F1. Focal loss "
            "raised Viral recall but reduced overall balance. Lung segmentation failed class-dependent "
            "gates. The hierarchy showed that Viral-versus-Bacterial separation is the main bottleneck."
        ),
        "Findings:": (
            "Label smoothing produced the best single seed-42 model. FT-004 remains the strongest "
            "multi-seed confirmed average system, while FT-010 is testing whether the single-model gain repeats."
        ),
        "Conclusion:": (
            "We will decide only after seeds 123 and 2026 finish. Newer models do not inherit the earlier "
            "locked-test score, and external validation remains required."
        ),
    }
    shorter_steps = {
        "Compare FT-010 fairly with the confirmed FT-004 ensemble and original DenseNet-201 pipeline using validation evidence only.":
            "Compare FT-010 with FT-004 and the original DenseNet-201 pipeline using validation evidence only.",
        "Freeze the chosen development configuration and regenerate Grad-CAM, calibration, and complete end-to-end latency for that exact checkpoint.":
            "Freeze the chosen configuration; regenerate Grad-CAM, calibration, and end-to-end latency.",
        "Evaluate on a genuinely independent external pediatric dataset before making a generalization or clinical-readiness claim.":
            "Use an independent external pediatric dataset before any generalization or clinical-readiness claim.",
    }
    checklist_starts = {
        "Start with the research problem",
        "Explain the data audit",
        "Explain the shared training",
        "Show how the ordered preprocessing",
        "Walk through fine-tuning",
        "End with the selection boundary",
        "Finish FT-010 seed 123",
        "Calculate the three-seed mean",
        "Compare FT-010 fairly",
        "Freeze the chosen development",
        "Evaluate on a genuinely independent",
    }

    for paragraph in root.xpath(".//w:p", namespaces=NS):
        text = paragraph_text(paragraph)
        if text == "RESEARCH DISCUSSION NOTES":
            ppr = paragraph.find(qn("pPr"))
            spacing = ppr.find(qn("spacing")) if ppr is not None else None
            if spacing is not None:
                spacing.set(qn("before"), "840")
        elif text.startswith("We created a leakage-controlled three-class classifier"):
            replace_all_text(paragraph, overview)
        elif text.startswith("FT-010 is running at seed 123"):
            replace_all_text(paragraph, current_status)
        elif text.startswith("Snapshot created"):
            replace_all_text(
                paragraph,
                f"Snapshot created {datetime.now().astimezone().strftime('%d %b %Y, %H:%M %Z')}.",
            )
        elif any(text.startswith(prefix) for prefix in checklist_starts):
            set_list_bullet(paragraph)
            if text in shorter_steps:
                replace_all_text(paragraph, shorter_steps[text])
        else:
            for label, body in script.items():
                if text.startswith(label):
                    replace_after_label(paragraph, label, body)
                    break

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_header(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    texts = root.xpath(".//w:t", namespaces=NS)
    if texts:
        texts[0].text = "PNEUMONIA CLASSIFICATION | DISCUSSION GUIDE"
        for node in texts[1:]:
            node.text = ""
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_footer(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    paragraph = root.find(qn("p"))
    if paragraph is not None:
        runs = paragraph.findall(qn("r"))
        for run in runs[6:]:
            paragraph.remove(run)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def update_styles(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    title_styles = root.xpath(".//w:style[@w:styleId='Title']", namespaces=NS)
    for style in title_styles:
        ppr = style.find(qn("pPr"))
        if ppr is not None:
            border = ppr.find(qn("pBdr"))
            if border is not None:
                ppr.remove(border)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def main() -> None:
    replacements = {
        "word/document.xml": update_document,
        "word/header1.xml": update_header,
        "word/footer1.xml": update_footer,
        "word/styles.xml": update_styles,
    }
    temporary = OUTPUT.with_suffix(".patched.tmp.docx")
    with zipfile.ZipFile(SOURCE, "r") as source, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename in replacements:
                payload = replacements[item.filename](payload)
            target.writestr(item, payload)
    temporary.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
