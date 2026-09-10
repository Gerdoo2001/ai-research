"""Create the 2026-08-10 accessible full-journal update from the approved DOCX."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_BREAK, WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "model" / "docs" / "Pneumonia_Three_Class_Classification_Full_Journal_20260809.docx"
OUTPUT = ROOT / "model" / "docs" / "Pneumonia_Three_Class_Classification_Full_Journal_20260810.docx"
AUDIT = ROOT / "model" / "docs" / "Pneumonia_Three_Class_Classification_Full_Journal_20260810.audit.json"
EXPECTED_SOURCE_SHA256 = "1f9d3181e7eb330476a768774f0884e3da38ea7e5fb6fe85baec56fafb5f2403"
BLACK = RGBColor(0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def black_run(run, *, size: float | None = None, bold: bool | None = None, italic: bool | None = None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), "Arial")
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), "Arial")
    run.font.color.rgb = BLACK
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    return run


def clear_paragraph_content(paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def replace_exact(doc: Document, old: str, new: str) -> None:
    matches = [p for p in doc.paragraphs if p.text == old]
    if len(matches) != 1:
        raise RuntimeError(f"expected one paragraph match, found {len(matches)}: {old[:100]}")
    paragraph = matches[0]
    clear_paragraph_content(paragraph)
    black_run(paragraph.add_run(new))


def find_heading(doc: Document, text: str):
    matches = [p for p in doc.paragraphs if p.text == text]
    if len(matches) != 1:
        raise RuntimeError(f"expected one heading match, found {len(matches)}: {text}")
    return matches[0]


def insert_paragraph_before(doc: Document, anchor, text: str = "", style: str | None = None):
    paragraph = doc.add_paragraph(style=style)
    if text:
        black_run(paragraph.add_run(text))
    anchor._p.addprevious(paragraph._p)
    return paragraph


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def apply_table_geometry(table, widths: list[int]) -> None:
    if sum(widths) != 9360:
        raise ValueError(f"table widths must total 9360 DXA, got {sum(widths)}")
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), "9360")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    # Align the visible outer border with surrounding body text by matching the
    # table indent to the 100-DXA start-cell margin.
    tbl_ind.set(qn("w:w"), "100")
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), "000000")

    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            set_cell_width(cell, widths[column_index])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    black_run(run, size=8.0, bold=(row_index == 0))
    set_repeat_table_header(table.rows[0])


def insert_table_before(doc: Document, anchor, headers: list[str], rows: list[list[str]], widths: list[int]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = doc.tables[0].style
    for cell, value in zip(table.rows[0].cells, headers):
        clear_paragraph_content(cell.paragraphs[0])
        black_run(cell.paragraphs[0].add_run(value), size=8.0, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row_values):
            clear_paragraph_content(cell.paragraphs[0])
            black_run(cell.paragraphs[0].add_run(str(value)), size=8.0)
    apply_table_geometry(table, widths)
    anchor._p.addprevious(table._tbl)
    return table


def set_table_cell(table, row: int, col: int, text: str, *, bold: bool = False) -> None:
    paragraph = table.cell(row, col).paragraphs[0]
    clear_paragraph_content(paragraph)
    black_run(paragraph.add_run(text), bold=bold)


def append_table_row(table, values: list[str]) -> None:
    cells = table.add_row().cells
    for cell, value in zip(cells, values):
        paragraph = cell.paragraphs[0]
        clear_paragraph_content(paragraph)
        black_run(paragraph.add_run(value), size=8.0)


def add_definition(doc: Document, term: str, definition: str) -> None:
    paragraph = doc.add_paragraph(style="Normal")
    paragraph.paragraph_format.keep_together = True
    black_run(paragraph.add_run(f"{term}. "), bold=True)
    black_run(paragraph.add_run(definition))


def add_glossary(doc: Document) -> None:
    page_break = doc.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)
    doc.add_paragraph("Appendix B. Plain-Language Glossary of Technical Terms", style="Heading 1")
    doc.add_paragraph(
        "This glossary is written for readers who are new to medical imaging and machine learning. "
        "A term may have a broader meaning elsewhere; each definition below explains how the term is used in this project."
    )
    categories = [
        (
            "B.1 Medical and imaging terms",
            [
                ("Chest radiograph or chest X-ray", "A two-dimensional image produced with X-rays to show structures inside the chest, especially the lungs, ribs, and heart."),
                ("Pneumonia", "An infection or inflammatory condition that affects the lungs and can create visible opacity patterns on a chest radiograph."),
                ("Normal", "The project label for an image without the dataset's labeled pneumonia finding. It does not guarantee that the patient has no other medical condition."),
                ("Viral Pneumonia", "The project label for pneumonia associated with a viral cause in the source dataset."),
                ("Bacterial Pneumonia", "The project label for pneumonia associated with a bacterial cause in the source dataset."),
                ("Pediatric", "Relating to children. The principal public dataset used in this experiment contains pediatric chest radiographs."),
                ("Clinical information", "Patient information that has medical meaning, such as symptoms, age, laboratory findings, or radiographic evidence. VIRIDIS does not create new clinical information."),
                ("Diagnostic system", "A system intended to support or make a medical diagnosis. This experiment is a research classifier and has not been prospectively validated as a diagnostic medical device."),
            ],
        ),
        (
            "B.2 Dataset preparation and leakage-control terms",
            [
                ("Dataset", "The organized collection of images and labels used for training and evaluation."),
                ("Source file", "A file found in the original dataset folders before auditing, filtering, or duplicate removal."),
                ("Canonical image", "The one retained representative when identical copies of the same underlying image are discovered. Redundant copies point to the same canonical item and are not treated as separate evidence."),
                ("Duplicate", "A second file that contains the same image content as another file. Exact duplicates were identified using content hashes and audited before removal."),
                ("SHA-256 hash", "A long digital fingerprint calculated from exact file bytes. Identical SHA-256 values provide strong evidence that two files are byte-for-byte identical."),
                ("Perceptual hash or pHash", "A compact fingerprint designed to find images that look similar even when their file bytes differ. pHash produces candidates that still require verification."),
                ("Manifest", "A locked CSV record listing each image path, class label, split, group, and audit identity used by the experiment."),
                ("Locked manifest", "A manifest that is not changed after the experimental protocol begins. Locking prevents quiet movement of images between training, validation, and test sets."),
                ("Group-aware split", "A split procedure that keeps related images or duplicates in one partition so related evidence cannot appear on both sides of an evaluation."),
                ("Training split", "The images used to calculate gradients and update model parameters."),
                ("Validation split", "The separate images used to compare configurations, select the best epoch, and make model-selection decisions. Their labels do not directly update weights."),
                ("Test split", "A held-out set intended for final evaluation after all model-selection decisions have been frozen."),
                ("Locked test", "The test split protected from routine access so it cannot influence architecture, preprocessing, tuning, seed, or checkpoint decisions."),
                ("Data leakage", "Information from validation or test data improperly influencing training or model selection, causing performance to look better than it will be on truly unseen data."),
                ("Class imbalance", "A condition in which the classes contain different numbers of images. Accuracy may then be dominated by the largest class."),
                ("Support", "The number of real examples of a particular class included in an evaluation."),
            ],
        ),
        (
            "B.3 Image preprocessing and augmentation terms",
            [
                ("Preprocessing", "A deterministic image transformation applied before the model receives an image. The same selected preprocessing is used for training, validation, and later inference."),
                ("Augmentation", "A random training-only modification that creates varied views of existing images. It is not applied to validation images in this experiment."),
                ("Pipeline", "The ordered sequence of operations applied to an image or used to train and evaluate a model."),
                ("Operation order", "The exact sequence in which preprocessing operations run. Order matters because each operation consumes the previous operation's output."),
                ("RGB", "A three-channel image representation containing red, green, and blue channels. Grayscale information can be copied into all three channels."),
                ("Grayscale", "An image represented by intensity or brightness rather than natural color."),
                ("Global Histogram Equalization or GE", "A whole-image contrast operation that redistributes intensity values using the image's overall histogram."),
                ("CLAHE", "Contrast Limited Adaptive Histogram Equalization. It improves local contrast in small regions while limiting amplification through a clip limit."),
                ("VIRIDIS pseudo-coloring", "A mapping that converts grayscale intensity values into artificial colors from the VIRIDIS color map. It adds no new anatomy or clinical evidence."),
                ("Histogram", "A count of how many pixels occur at each intensity level."),
                ("Contrast", "The visual separation between darker and brighter image regions."),
                ("Resize", "Changing image dimensions so the model receives a standard input size."),
                ("Center crop", "Keeping a central region after aspect-ratio-preserving resizing so the final image is exactly 224 by 224 pixels."),
                ("Horizontal flip", "A left-to-right mirror transformation applied randomly to training images with probability 0.5."),
                ("Random rotation", "A small training-only turn between approximately -5 and +5 degrees, used to reduce sensitivity to minor positioning differences."),
                ("ImageNet normalization", "Subtracting the ImageNet channel means and dividing by the ImageNet channel standard deviations so inputs match the scale expected by pretrained networks."),
                ("Float32", "A 32-bit floating-point number format. Pixel values are converted from integers to decimal values before normalization."),
                ("Tensor", "A multidimensional numeric array used by PyTorch. Each input image becomes a tensor with shape 3 by 224 by 224."),
                ("ROI or region of interest", "A selected anatomical area, such as the lungs. This experiment did not require an ROI mask for the final classifier."),
                ("Segmentation", "Assigning pixels to anatomical or disease regions. It is different from assigning one class to the entire image."),
                ("PCA", "Principal Component Analysis, a dimensionality-reduction method. It was not part of the final image-classification pipeline."),
                ("FiLM", "Feature-wise Linear Modulation, a method that conditions features using another input. It was not part of the selected final configuration."),
            ],
        ),
        (
            "B.4 Model and architecture terms",
            [
                ("Artificial intelligence or AI", "A broad term for computer systems that perform tasks associated with human intelligence. Machine learning is one approach within AI."),
                ("Machine learning or ML", "A method in which a computer learns patterns from examples rather than following only manually written decision rules."),
                ("Deep learning", "Machine learning that uses neural networks with many computational layers."),
                ("Neural network", "A connected set of mathematical operations containing learned weights that transform an input into an output."),
                ("CNN", "Convolutional Neural Network. A neural-network family that learns spatial image features using convolution operations."),
                ("Architecture", "The predefined arrangement of model layers and connections, such as DenseNet-201, DenseNet-169, or ResNet-50."),
                ("DenseNet", "Densely Connected Convolutional Network. Each layer receives feature maps from earlier layers, encouraging feature reuse."),
                ("DenseNet-201", "The selected architecture depth used for the final candidate. The name identifies the DenseNet design variant; it does not mean 201 classes or 201 images."),
                ("DenseNet-169", "A shallower DenseNet variant with fewer parameters than DenseNet-201 and practically comparable validation performance in this study."),
                ("ResNet", "Residual Network. It uses shortcut or skip connections that help information and gradients pass through deep networks."),
                ("ResNet-50", "A 50-layer residual-network variant evaluated as one of the principal competitors."),
                ("YOLO classification", "The classification variants of YOLOv5 and YOLOv8 used to assign one class to an entire image. This is different from YOLO object detection boxes."),
                ("Parameter", "A learned numeric value, usually a weight or bias, that the optimizer changes during training."),
                ("Weight", "A learned multiplier controlling how strongly one feature influences later computations."),
                ("Bias", "A learned offset added by a layer."),
                ("Trainable parameter", "A parameter allowed to receive gradients and change during training."),
                ("Hyperparameter", "A setting chosen by the researcher, such as learning rate, batch size, weight decay, or maximum epochs."),
                ("Classifier head", "The final model component that converts learned image features into scores for Normal, Viral, and Bacterial classes."),
                ("Backbone", "The main feature-extraction portion of a pretrained neural network."),
                ("ImageNet weights", "Parameters learned previously from the large ImageNet dataset and used as the starting point for transfer learning."),
                ("Transfer learning", "Starting from knowledge learned on another dataset and adapting the model to the current task."),
                ("Fine-tuning", "Continuing training so pretrained parameters adapt to the pneumonia classes."),
                ("Freeze backbone", "Preventing backbone parameters from changing. It was set to false, so the selected network was fully fine-tuned."),
                ("SAS", "A spatial-attention selection component explored in separate DenseNet-201 experiments. It was not part of the selected standard DenseNet-201 winner."),
                ("MFF", "Multi-level Feature Fusion, which combines features from different network depths. The experimental SAS/MFF variants did not exceed the selected standard DenseNet-201 result."),
            ],
        ),
        (
            "B.5 Training terms",
            [
                ("Training", "The repeated process of making predictions, calculating error, computing gradients, and updating parameters."),
                ("Epoch", "One complete pass through the available training images. Twelve epochs was the maximum, not a guarantee that every run would use all twelve."),
                ("Batch", "A small group of images processed together before one optimizer update."),
                ("Batch size", "The number of images in a batch. The final DenseNet-201 configuration used eight images per batch because of GPU-memory constraints."),
                ("Iteration or step", "One batch-level forward-and-backward update within an epoch."),
                ("Forward pass", "The calculation that turns an input image into class scores and a prediction."),
                ("Loss", "A numeric measure of prediction error used to guide parameter updates. Lower loss is generally better when comparing the same definition."),
                ("Cross-entropy", "The classification loss used in the selected configuration. It penalizes assigning low probability to the correct class."),
                ("Class weighting", "Giving errors from less-represented classes more influence in the training loss."),
                ("Gradient", "A calculated direction showing how each trainable parameter should change to reduce the loss."),
                ("Backpropagation", "The procedure used to compute gradients from the loss backward through the neural network."),
                ("Optimizer", "The algorithm that uses gradients to update model parameters."),
                ("AdamW", "The selected optimizer. It adapts update sizes for individual parameters and applies decoupled weight decay."),
                ("Learning rate", "The initial scale of each optimizer update. The selected value was 0.0001."),
                ("Weight decay", "A regularization setting that discourages excessively large parameter values. The selected value was 0.0001."),
                ("Regularization", "A technique intended to reduce overfitting and improve performance on unseen images."),
                ("Scheduler", "A rule that changes the learning rate during training."),
                ("ReduceLROnPlateau", "The scheduler used here. It reduces the learning rate when validation macro F1 stops improving for the configured patience."),
                ("Early stopping", "Ending training when validation performance fails to improve sufficiently for a defined number of epochs."),
                ("Patience", "The number of non-improving epochs tolerated before early stopping or a scheduler action. Early-stopping patience was four."),
                ("Checkpoint", "A saved package containing model parameters and training state."),
                ("Best checkpoint", "The checkpoint from the epoch with the highest validation macro F1 under the configured minimum-improvement rule."),
                ("Mixed precision or AMP", "Training with a controlled mixture of numeric precisions to reduce GPU memory and improve speed while maintaining stable calculations."),
                ("Overfitting", "Learning patterns too specific to the training data, producing strong training performance but weaker unseen-data performance."),
                ("Seed", "An integer that initializes pseudo-random sequences. Seeds 42, 123, and 2026 represent three different controlled training conditions."),
                ("Deterministic", "Configured to repeat the same random sequence and GPU behavior as closely as supported by the same software and hardware."),
                ("Reproducibility", "The ability to repeat the documented procedure and obtain the same or closely comparable results."),
            ],
        ),
        (
            "B.6 Evaluation and statistical terms",
            [
                ("Prediction", "The class selected by the model for an input image."),
                ("Probability", "The model's numeric confidence allocation across the three possible classes."),
                ("Accuracy", "The proportion of all evaluated images classified correctly. It can be influenced by the largest class."),
                ("Precision", "Among images predicted as one class, the proportion that truly belong to that class."),
                ("Recall or sensitivity", "Among real examples of one class, the proportion correctly found by the model."),
                ("Specificity", "Among examples that are not a class, the proportion correctly rejected as not belonging to that class."),
                ("F1 score", "The harmonic mean of precision and recall for one class. It is high only when both precision and recall are reasonably high."),
                ("Macro F1", "The arithmetic mean of the Normal, Viral, and Bacterial F1 scores, giving each class equal weight regardless of class size."),
                ("Balanced accuracy", "The arithmetic mean of recall across the three classes."),
                ("Weighted F1", "An average of class F1 scores weighted by how many examples each class contains."),
                ("Confusion matrix", "A table counting each combination of true class and predicted class, showing the exact types of correct decisions and mistakes."),
                ("ROC AUC", "Area Under the Receiver Operating Characteristic Curve. It summarizes how well a class score ranks positive examples above negative examples across thresholds."),
                ("PR AUC", "Area Under the Precision-Recall Curve. It summarizes the precision-recall tradeoff across thresholds and is useful for imbalanced classes."),
                ("Calibration", "Agreement between predicted confidence and observed correctness."),
                ("Expected Calibration Error or ECE", "A binned summary of the difference between confidence and accuracy. Lower values indicate better calibration under that calculation."),
                ("Brier score", "The mean squared difference between predicted probabilities and the true one-hot labels. Lower is better."),
                ("Log loss", "A probability-sensitive loss that strongly penalizes confident incorrect predictions. Lower is better."),
                ("Mean", "The arithmetic average. Three-seed mean macro F1 is the sum of the three seed results divided by three."),
                ("Standard deviation or SD", "A measure of how spread out the seed results are around their mean. Lower SD means the observed seed results were more consistent."),
                ("Percentage point or pp", "The direct difference between two percentages. For example, 84.168% minus 84.142% is 0.026 percentage points."),
                ("Primary metric", "The measurement declared most important for model selection. This study used mean validation macro F1 across three seeds."),
                ("Tie-breaker", "A secondary criterion used when primary results are equal or practically indistinguishable. Balanced accuracy and accuracy were considered secondary evidence."),
                ("Statistical significance", "Evidence that an observed difference is unlikely under a defined chance model. No claim of statistically significant superiority is made for the very small DenseNet-201 versus DenseNet-169 mean difference."),
            ],
        ),
        (
            "B.7 Experiment, interpretation, and deployment terms",
            [
                ("Baseline", "The reference configuration without an added enhancement operation."),
                ("Ablation study", "A controlled comparison that changes or removes one component to measure its effect."),
                ("Controlled comparison", "An experiment in which relevant settings are held fixed while the factor under study is changed."),
                ("Configuration", "The complete set of model, preprocessing, training, and evaluation settings defining one run."),
                ("Run", "One execution of a configuration at one seed, producing checkpoints, histories, and metrics."),
                ("Smoke test", "A short, small-sample check used to verify that code and hardware work. Smoke-test scores are not scientific model results."),
                ("Optuna", "The automated tuning library used for the bounded hyperparameter studies."),
                ("TPE", "Tree-structured Parzen Estimator, the Optuna sampling method used to propose tuning trials based on earlier observations."),
                ("Grad-CAM", "Gradient-weighted Class Activation Mapping, an explanation method that highlights image regions influencing a model's class score."),
                ("Explainability", "Methods that help people inspect why a model produced an output. Explainability does not prove that the reasoning is medically correct."),
                ("Inference", "Using a trained model to produce a prediction for a new image without updating its parameters."),
                ("Latency", "The measured time required to process an input under a stated measurement boundary."),
                ("Throughput", "The number of inputs processed per unit of time."),
                ("Final validation-selected candidate", "The configuration chosen after the validation comparison. Here it is DenseNet-201 plus VIRIDIS -> GE -> CLAHE."),
                ("Frozen configuration", "A configuration that will not be changed after selection, especially after any test labels are viewed."),
                ("External validation", "Evaluation on a genuinely independent dataset from another source, institution, population, or acquisition setting."),
                ("Generalization", "The ability to perform well on unseen data rather than only on the images used during development."),
                ("Research limitation", "A boundary that restricts how broadly a result can be interpreted, such as one public pediatric dataset or lack of prospective clinical testing."),
            ],
        ),
    ]
    for heading, definitions in categories:
        doc.add_paragraph(heading, style="Heading 2")
        for term, definition in definitions:
            add_definition(doc, term, definition)


def main() -> int:
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"reference hash changed: {source_hash}")

    doc = Document(SOURCE)

    replacements = {
        "Updated 9 August 2026 | Reconstructed exclusively from audited repository artifacts":
            "Updated 10 August 2026 | Reconstructed exclusively from audited repository artifacts",
        "Reporting boundary: the staged ordered-preprocessing study used the locked validation set only. The test set was evaluated once for the earlier frozen DenseNet-201 + global-histogram-equalization model and was not reopened after the new validation winner emerged.":
            "Reporting boundary: model and preprocessing selection used the locked validation set. The final three-seed comparison is complete for DenseNet-201, DenseNet-169, and ResNet-50. The test set was evaluated once for the earlier frozen DenseNet-201 + global-histogram-equalization model and was not reopened for the newly selected DenseNet-201 + VIRIDIS -> GE -> CLAHE candidate.",
        "Eight ImageNet-initialized classifiers were compared under matched optimization controls. DenseNet-201 and ResNet-50 then received exhaustive evaluation of the baseline plus every ordered subset of CLAHE, global histogram equalization (GE), and VIRIDIS pseudo-coloring (16 conditions per model). The DenseNet-201 winner, VIRIDIS -> GE -> CLAHE, achieved 85.294% validation macro F1, 84.546% balanced accuracy, and 86.613% accuracy. Transfer of that pipeline across all supported architectures placed DenseNet-169 second overall at 84.742% macro F1. ResNet-50's own winner was CLAHE -> VIRIDIS -> GE at 84.663%.":
            "Eight ImageNet-initialized classifiers were compared under matched optimization controls. DenseNet-201 and ResNet-50 then received exhaustive evaluation of the baseline plus every ordered subset of CLAHE, global histogram equalization (GE), and VIRIDIS pseudo-coloring (16 conditions per model). The three strongest seed-42 configurations were retrained at seeds 123 and 2026, producing three independent controlled runs per candidate.",
        "The leading DenseNet-201 configuration exceeded the second-ranked DenseNet-169 by 0.551 percentage points in macro F1, while DenseNet-169 was smaller and faster. The result also demonstrated an architecture-by-preprocessing-order interaction: the DenseNet-201 winning order produced only 82.587% on ResNet-50, whereas ResNet-50 required a different order to reach its best score.":
            "Across seeds 42, 123, and 2026, DenseNet-201 + VIRIDIS -> GE -> CLAHE achieved the highest mean validation macro F1 at 84.168%, followed extremely closely by DenseNet-169 with the same pipeline at 84.142%; ResNet-50 + CLAHE -> VIRIDIS -> GE averaged 83.472%. DenseNet-201 also led mean balanced accuracy and mean accuracy, while DenseNet-169 had lower seed-to-seed variability and slightly higher mean Viral recall.",
        "The current validation winner has not yet received multi-seed confirmation or locked-test evaluation. Therefore, it is the best completed validation configuration, not a clinically validated diagnostic system. The only locked-test result remains the earlier DenseNet-201 + GE model, which achieved 78.877% macro F1.":
            "DenseNet-201 + VIRIDIS -> GE -> CLAHE is therefore the final validation-selected candidate under the predefined primary rule of highest three-seed mean macro F1. Its 0.026-percentage-point mean advantage over DenseNet-169 is practically very small and is not presented as statistically significant superiority. The newly selected candidate has not been evaluated on the locked test and is not a clinically validated diagnostic system. The only locked-test result remains the earlier DenseNet-201 + GE model, which achieved 78.877% macro F1.",
        "The experiment ledger contains 119 completed entries: 95 full/analysis entries and 24 smoke or pilot checks; one historical failure remains preserved for auditability.":
            "The experiment ledger and dedicated comparison summaries preserve every completed full, analysis, pilot, and robustness run, including the four new competitor seed runs.",
        "DenseNet-201 + VIRIDIS -> GE -> CLAHE is the current validation leader at 85.294% macro F1.":
            "DenseNet-201 + VIRIDIS -> GE -> CLAHE is the selected final validation candidate with 84.168% mean macro F1 across three seeds.",
        "DenseNet-169 with the same transferred pipeline is second overall at 84.742%; ResNet-50's architecture-specific winner is third at 84.663%.":
            "DenseNet-169 was practically tied at 84.142% three-seed mean macro F1 and was more stable; ResNet-50 averaged 83.472%.",
        "The sealed test split was not used to choose any staged preprocessing order and was not reopened for the new winner.":
            "The sealed test split was not used to choose preprocessing, architecture, seed, or checkpoint and was not reopened for the final validation-selected candidate.",
        "7. Multi-Seed Robustness and Single Locked-Test Result":
            "7. Earlier Multi-Seed Robustness and Single Locked-Test Result",
        "The three-seed confirmation mean of 83.895% applies only to DenseNet-201 + global HE. It must not be transferred to the newer VIRIDIS -> GE -> CLAHE winner, which currently has a single completed seed-42 result.":
            "The earlier three-seed confirmation mean of 83.895% applies only to DenseNet-201 + global HE. The newer VIRIDIS -> GE -> CLAHE candidate now has its own completed three-seed evidence: 85.294%, 83.570%, and 83.640% macro F1, with a mean of 84.168%.",
        "The bounded Optuna studies had already shown that the original AdamW, balanced cross-entropy, conservative augmentation, learning rate 0.0001, and weight decay 0.0001 controls outperformed the tested SGD/focal alternatives for each parent candidate. The SAS/MFF additions also did not exceed standard DenseNet-201. These findings justify retaining the controlled optimization recipe while varying preprocessing order. However, the three-seed robustness experiment applies only to the earlier DenseNet-201 + GE configuration; the new winner still requires independent seeds.":
            "The bounded Optuna studies had already shown that the original AdamW, balanced cross-entropy, conservative augmentation, learning rate 0.0001, and weight decay 0.0001 controls outperformed the tested SGD/focal alternatives for each parent candidate. The SAS/MFF additions also did not exceed standard DenseNet-201. These findings justified retaining the controlled optimization recipe while varying preprocessing order. Independent seeds 123 and 2026 have now been completed for the leading DenseNet-201, DenseNet-169, and ResNet-50 configurations.",
        "The 78.877% locked-test macro F1 belongs to DenseNet-201 + GE, not to DenseNet-201 + VIRIDIS -> GE -> CLAHE. Reporting the old test score as the new winner's performance would be incorrect. A future test of the new winner would constitute an additional test-set access and should be governed by a newly documented protocol or, preferably, an independent external dataset.":
            "The 78.877% locked-test macro F1 belongs to DenseNet-201 + GE, not to DenseNet-201 + VIRIDIS -> GE -> CLAHE. Reporting the old test score as the newly selected candidate's performance would be incorrect. The new candidate was selected using validation evidence only. Any additional access to the existing test set must be documented as a second access and must not trigger further tuning; independent external validation remains preferable.",
        "13.1 Why DenseNet-201 is the best current validation model":
            "13.1 Why DenseNet-201 is the final validation-selected candidate",
        "DenseNet-201 + VIRIDIS -> GE -> CLAHE is the best current validation configuration because it has the highest completed validation macro F1 after distinct-configuration deduplication. Macro F1 gives equal weight to Normal, Viral, and Bacterial F1 scores, making it more appropriate than accuracy alone for the imbalanced class distribution. Its 86.613% accuracy is also the highest among the two leading configurations, but the choice is based primarily on 85.294% macro F1 rather than raw accuracy.":
            "DenseNet-201 + VIRIDIS -> GE -> CLAHE is the final validation-selected candidate because it achieved the highest mean macro F1 across the three prespecified seeds: 84.168%. Macro F1 gives equal weight to Normal, Viral, and Bacterial F1 scores, making it more appropriate than accuracy alone for the imbalanced class distribution. DenseNet-201 also produced the highest mean balanced accuracy (83.932%) and mean accuracy (85.202%) among the three finalists.",
        "The model is not uniformly superior on every measure. DenseNet-169 is approximately 31% smaller by parameter count, has a smaller model file, and its measured batch-1 model latency is lower. DenseNet-169 also produced a slightly higher Viral F1, while DenseNet-201 produced higher Normal and Bacterial F1. Thus DenseNet-201 is preferred for balanced predictive performance; DenseNet-169 is the stronger efficiency alternative.":
            "The decision is close rather than absolute. DenseNet-201 exceeded DenseNet-169 by only 0.026 percentage points in three-seed mean macro F1. DenseNet-169 had lower seed-to-seed standard deviation (0.532 versus 0.975 percentage points), slightly higher mean Viral recall, fewer parameters, a smaller file, and lower measured model-only latency. DenseNet-201 is selected because it ranks first under the predefined primary metric and the two principal aggregate secondary metrics; DenseNet-169 remains a practically comparable and more efficient alternative.",
        "The new DenseNet-201 and DenseNet-169 leaders currently have only seed-42 evidence. Their small 0.551-point separation may change under additional seeds.":
            "The final three candidates now have completed seeds 42, 123, and 2026. Only three seeds were evaluated, so the means and standard deviations remain estimates based on a small replication count.",
        "The current winner was selected after the earlier one-time locked-test evaluation. It has validation evidence only and must not inherit the earlier test result.":
            "The final validation-selected candidate emerged after the earlier one-time locked-test evaluation of DenseNet-201 + GE. It must not inherit that earlier model's 78.877% test macro F1.",
        "Grad-CAM, calibration, and end-to-end efficiency measurements were produced for the earlier frozen model and have not yet been regenerated for the new validation winner.":
            "Grad-CAM, calibration, and complete end-to-end efficiency measurements were produced for the earlier frozen model and have not yet been regenerated for the newly selected DenseNet-201 + VIRIDIS -> GE -> CLAHE checkpoint.",
        "No statistical confidence interval or paired significance test was computed for the 0.551-point top-two difference.":
            "No statistical significance claim is made for the 0.026-percentage-point three-seed mean difference between DenseNet-201 and DenseNet-169. Three seeds are insufficient for a strong distributional claim.",
        "Retrain DenseNet-201 + VIRIDIS -> GE -> CLAHE and DenseNet-169 + the same pipeline at seeds 123 and 2026.":
            "Freeze DenseNet-201 + VIRIDIS -> GE -> CLAHE as the final validation-selected configuration and preserve the three completed seed checkpoints.",
        "Compare mean macro F1, standard deviation, per-class recall, calibration, and paired predictions across seeds.":
            "Use the seed-42 best checkpoint at epoch 6 as the documented single-checkpoint representative because seed 42 was the original screening seed; do not choose a checkpoint using test results.",
        "Freeze one final checkpoint-selection rule without consulting the sealed test labels.":
            "If the existing locked test is accessed again, evaluate the frozen representative once, label the event as a second test access, and prohibit further configuration changes.",
        "Prefer an external dataset for final confirmation; if the existing locked test is reopened, document that this is a second access and no longer the original one-shot test protocol.":
            "Prefer an independent external dataset for stronger generalization evidence across hospitals, devices, age groups, and prevalence patterns.",
        "Regenerate Grad-CAM, calibration, and true end-to-end latency for the final frozen winner.":
            "Regenerate Grad-CAM, calibration, and true end-to-end latency for the final frozen DenseNet-201 checkpoint.",
        "The repository now provides a complete, leakage-controlled comparison of eight supported classifiers, exhaustive ordered preprocessing for DenseNet-201 and ResNet-50, cross-model transfer of the DenseNet-201 winner, bounded hyperparameter trials, a prior three-seed robustness study, and a preserved one-time locked-test evaluation. The strongest current validation result is DenseNet-201 + VIRIDIS -> GE -> CLAHE at 85.294% macro F1. DenseNet-169 with the same pipeline is second overall at 84.742%, and ResNet-50's best architecture-specific order reaches 84.663%.":
            "The repository now provides a leakage-controlled comparison of eight supported classifiers, exhaustive ordered preprocessing for DenseNet-201 and ResNet-50, cross-model transfer of the leading pipeline, bounded hyperparameter trials, and completed three-seed confirmation of the top DenseNet-201, DenseNet-169, and ResNet-50 configurations. DenseNet-201 + VIRIDIS -> GE -> CLAHE averaged 84.168% macro F1, 83.932% balanced accuracy, and 85.202% accuracy across seeds 42, 123, and 2026.",
        "DenseNet-201 is therefore the best overall validation model, while DenseNet-169 offers the more efficient top-two alternative. This conclusion is deliberately limited to the audited validation evidence. Multi-seed confirmation and independent test or external validation are still required before the new winner can be presented as a robust final model.":
            "DenseNet-201 + VIRIDIS -> GE -> CLAHE is therefore selected as the final validation candidate under the predefined primary metric of three-seed mean macro F1. DenseNet-169 was practically tied at 84.142%, more stable across seeds, and more efficient, so the report does not claim substantial or statistically significant superiority. The selected candidate still requires a properly governed locked-test evaluation or, preferably, independent external validation before any clinical interpretation.",
    }
    for old, new in replacements.items():
        replace_exact(doc, old, new)

    # Executive summary table.
    summary_table = doc.tables[0]
    summary_updates = {
        1: ("Status", "Top-three multi-seed validation complete; final validation candidate selected"),
        2: ("Final validation-selected candidate", "DenseNet-201 + VIRIDIS -> GE -> CLAHE"),
        3: ("Seeds", "42, 123, and 2026"),
        4: ("Three-seed mean validation macro F1", "0.841680 (84.168%)"),
        5: ("Three-seed mean balanced accuracy", "0.839324 (83.932%)"),
        6: ("Three-seed mean accuracy", "0.852021 (85.202%)"),
        7: ("Locked-test boundary", "Selected candidate not tested; earlier DenseNet-201 + GE test macro F1 = 0.788772"),
        8: ("Generated", "August 10, 2026 (Asia/Manila)"),
    }
    for row, (label, value) in summary_updates.items():
        set_table_cell(summary_table, row, 0, label, bold=True)
        set_table_cell(summary_table, row, 1, value)

    status_table = doc.tables[1]
    set_table_cell(status_table, 5, 2, "Complete for earlier GE study and final top-three comparison; 9/9 finalist seed-runs")
    set_table_cell(status_table, 5, 3, "model/results/top_competitor_multiseed_confirmation/summary.json")
    set_table_cell(status_table, 7, 0, "Earlier frozen/tested selection")
    set_table_cell(status_table, 7, 2, "Preserved: DenseNet-201 + GE / final_selected_s42")
    set_table_cell(status_table, 12, 0, "Final validation-selected candidate")
    set_table_cell(status_table, 12, 2, "DenseNet-201 + VIRIDIS -> GE -> CLAHE; three-seed mean macro F1 84.168%")
    set_table_cell(status_table, 12, 3, "model/results/top_competitor_multiseed_confirmation/summary.json")
    append_table_row(
        status_table,
        [
            "Final competitor confirmation",
            "Seed 42 only",
            "Complete: DenseNet-201, DenseNet-169, and ResNet-50 at seeds 42/123/2026",
            "model/results/top_competitor_multiseed_confirmation/summary.csv",
        ],
    )

    # Plain-language reader guide before numbered Section 1.
    section_one = find_heading(doc, "1. Repository, Source, and Locked Dataset Audit")
    insert_paragraph_before(doc, section_one, "Reader Guide: How to Understand This Report", "Heading 1")
    insert_paragraph_before(
        doc,
        section_one,
        "In one sentence: the project trained several computer-vision models to classify pediatric chest X-rays as Normal, Viral Pneumonia, or Bacterial Pneumonia, then selected DenseNet-201 + VIRIDIS -> GE -> CLAHE because it had the highest average validation macro F1 across three controlled random seeds.",
        "Normal",
    )
    for text in [
        "A model is the learned mathematical system; an architecture is its predefined layer design; a configuration is the complete collection of model, preprocessing, and training settings.",
        "Seed 42, seed 123, and seed 2026 are three repeatable random training conditions. They do not mean 42, 123, or 2026 images, epochs, or percent performance.",
        "Macro F1 is the primary selection metric because it gives Normal, Viral, and Bacterial classes equal importance. Accuracy is reported as supporting evidence but can be influenced by the largest class.",
        "Validation results were used for selection. The locked test was not reopened for the newly selected candidate, so the earlier DenseNet-201 + GE test score must not be assigned to it.",
        "Technical terms are explained when first used and collected in Appendix B for quick lookup.",
    ]:
        insert_paragraph_before(doc, section_one, text, "List Bullet")

    # Update leading-model configuration table from seed-42-only to aggregate evidence.
    leading = doc.tables[46]
    set_table_cell(leading, 15, 1, "42, 123, 2026")
    set_table_cell(leading, 15, 2, "42, 123, 2026")
    set_table_cell(leading, 15, 3, "42, 123, 2026")
    set_table_cell(leading, 16, 0, "Mean validation macro F1")
    set_table_cell(leading, 16, 1, "84.168%")
    set_table_cell(leading, 16, 2, "84.142%")
    set_table_cell(leading, 16, 3, "83.472%")
    append_table_row(leading, ["Macro F1 sample SD", "0.975 pp", "0.532 pp", "1.057 pp"])
    append_table_row(leading, ["Mean balanced accuracy", "83.932%", "83.804%", "83.207%"])
    append_table_row(leading, ["Mean accuracy", "85.202%", "85.088%", "84.477%"])
    append_table_row(leading, ["Mean Viral recall", "64.275%", "64.873%", "63.528%"])

    # Final three-seed section before Discussion.
    discussion = find_heading(doc, "13. Discussion")
    insert_paragraph_before(doc, discussion, "12.10 Completed Three-Seed Confirmation", "Heading 2")
    insert_paragraph_before(
        doc,
        discussion,
        "The same selected configuration for each finalist was trained under seeds 42, 123, and 2026. A seed initializes controlled pseudo-random choices such as the new layer weights, training-batch order, horizontal-flip decisions, and small rotation angles. The dataset manifests, preprocessing order, architecture, optimizer, learning rate, weight decay, loss, epoch limit, and early-stopping rules remained fixed within each candidate. Multiple seeds test whether a result is repeatable rather than the product of one favorable random sequence.",
        "Normal",
    )
    insert_paragraph_before(
        doc,
        discussion,
        "Table 12.12. Per-seed validation results for the three final candidates. Viral recall is shown because Viral Pneumonia was the most difficult class.",
        "Caption",
    )
    per_seed_rows = [
        ["DenseNet-201", "VIRIDIS -> GE -> CLAHE", "42", "85.294%", "84.546%", "86.613%", "62.332%", "6"],
        ["DenseNet-201", "VIRIDIS -> GE -> CLAHE", "123", "83.570%", "83.746%", "84.325%", "66.816%", "3"],
        ["DenseNet-201", "VIRIDIS -> GE -> CLAHE", "2026", "83.640%", "83.505%", "84.668%", "63.677%", "9"],
        ["DenseNet-169", "VIRIDIS -> GE -> CLAHE", "42", "84.742%", "84.523%", "85.698%", "66.368%", "6"],
        ["DenseNet-169", "VIRIDIS -> GE -> CLAHE", "123", "83.956%", "83.457%", "85.469%", "60.090%", "10"],
        ["DenseNet-169", "VIRIDIS -> GE -> CLAHE", "2026", "83.728%", "83.431%", "84.096%", "68.161%", "7"],
        ["ResNet-50", "CLAHE -> VIRIDIS -> GE", "42", "84.663%", "84.416%", "85.584%", "65.022%", "11"],
        ["ResNet-50", "CLAHE -> VIRIDIS -> GE", "123", "82.648%", "82.354%", "83.410%", "65.471%", "4"],
        ["ResNet-50", "CLAHE -> VIRIDIS -> GE", "2026", "83.105%", "82.852%", "84.439%", "60.090%", "4"],
    ]
    insert_table_before(
        doc,
        discussion,
        ["Model", "Preprocessing", "Seed", "Macro F1", "Balanced", "Accuracy", "Viral recall", "Best epoch"],
        per_seed_rows,
        [1100, 2050, 560, 1000, 1000, 1000, 1250, 1400],
    )
    insert_paragraph_before(
        doc,
        discussion,
        "Table 12.13. Aggregate validation comparison. SD means sample standard deviation across the three seeds; pp means percentage points.",
        "Caption",
    )
    aggregate_rows = [
        ["1", "DenseNet-201", "VIRIDIS -> GE -> CLAHE", "84.168%", "0.975 pp", "83.932%", "85.202%", "64.275%"],
        ["2", "DenseNet-169", "VIRIDIS -> GE -> CLAHE", "84.142%", "0.532 pp", "83.804%", "85.088%", "64.873%"],
        ["3", "ResNet-50", "CLAHE -> VIRIDIS -> GE", "83.472%", "1.057 pp", "83.207%", "84.477%", "63.528%"],
    ]
    insert_table_before(
        doc,
        discussion,
        ["Rank", "Model", "Preprocessing", "Mean F1", "F1 SD", "Mean balanced", "Mean accuracy", "Mean Viral recall"],
        aggregate_rows,
        [550, 1100, 2050, 1050, 900, 1200, 1200, 1310],
    )
    insert_paragraph_before(
        doc,
        discussion,
        "DenseNet-201 and DenseNet-169 were practically tied: their mean macro F1 values differed by only 0.026 percentage points. DenseNet-169 was more stable and had slightly higher mean Viral recall. DenseNet-201 nevertheless ranked first under the declared primary metric and also ranked first in mean balanced accuracy and mean accuracy.",
        "Normal",
    )
    insert_paragraph_before(doc, discussion, "12.11 Final Model Decision and Why It Was Chosen", "Heading 2")
    insert_paragraph_before(
        doc,
        discussion,
        "Final selection: DenseNet-201 + VIRIDIS -> GE -> CLAHE. The decision follows the predefined rule rather than a subjective preference: first compare mean validation macro F1 across seeds; use balanced accuracy and accuracy as supporting evidence; retain per-class recall and variability as limitations. DenseNet-201 finished first on all three aggregate performance measures. The small margin means the wording must be 'selected final validation candidate,' not 'proven substantially superior model.'",
        "Normal",
    )
    for text in [
        "Primary evidence: highest three-seed mean macro F1, 84.168%.",
        "Secondary evidence: highest mean balanced accuracy, 83.932%, and highest mean accuracy, 85.202%.",
        "Close-comparison limitation: DenseNet-169 averaged 84.142% macro F1, only 0.026 percentage points lower, and had lower variability.",
        "Checkpoint rule: the seed-42 best checkpoint at epoch 6 is the documented single-checkpoint representative because seed 42 was the original screening condition, not because of any test result.",
        "Test boundary: the new candidate has not been evaluated on the locked test; the previous DenseNet-201 + GE test result cannot be reassigned to it.",
    ]:
        insert_paragraph_before(doc, discussion, text, "List Bullet")

    # Appendix A artifact index additions.
    artifact_table = doc.tables[49]
    append_table_row(
        artifact_table,
        ["DenseNet-201 winner multi-seed summary", "model/results/winner_multiseed_confirmation/summary.json"],
    )
    append_table_row(
        artifact_table,
        ["Top-three multi-seed comparison", "model/results/top_competitor_multiseed_confirmation/summary.json"],
    )
    append_table_row(
        artifact_table,
        ["Top-three per-seed comparison CSV", "model/results/top_competitor_multiseed_confirmation/summary.csv"],
    )

    add_glossary(doc)

    # Ensure newly created heading/caption runs use black Arial and all inherited
    # hyperlinks or direct colors cannot introduce blue into the final copy.
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.color.type is not None and run.font.color.rgb is not None:
                run.font.color.rgb = BLACK
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.color.type is not None and run.font.color.rgb is not None:
                            run.font.color.rgb = BLACK

    doc.core_properties.modified = datetime.now(timezone.utc)
    doc.core_properties.comments = "Updated with completed three-seed finalist comparison and plain-language glossary."
    doc.save(OUTPUT)

    # Structural verification.
    check = Document(OUTPUT)
    source_images = {}
    final_images = {}
    with ZipFile(SOURCE) as archive:
        for name in archive.namelist():
            if name.startswith("word/media/"):
                source_images[name] = hashlib.sha256(archive.read(name)).hexdigest()
    with ZipFile(OUTPUT) as archive:
        for name in archive.namelist():
            if name.startswith("word/media/"):
                final_images[name] = hashlib.sha256(archive.read(name)).hexdigest()
        document_xml = archive.read("word/document.xml")
        footer_xml = archive.read("word/footer1.xml")

    all_text = "\n".join(p.text for p in check.paragraphs)
    stale_phrases = [
        "has not yet received multi-seed confirmation",
        "currently has a single completed seed-42 result",
        "new winner still requires independent seeds",
        "currently have only seed-42 evidence",
        "Multi-seed confirmation and independent test",
    ]
    missing_terms = [phrase for phrase in stale_phrases if phrase in all_text]
    direct_colors = set()
    for paragraph in check.paragraphs:
        for run in paragraph.runs:
            if run.font.color.rgb is not None:
                direct_colors.add(str(run.font.color.rgb))
    for table in check.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.color.rgb is not None:
                            direct_colors.add(str(run.font.color.rgb))

    expected_headings = {
        "Reader Guide: How to Understand This Report",
        "12.10 Completed Three-Seed Confirmation",
        "12.11 Final Model Decision and Why It Was Chosen",
        "Appendix B. Plain-Language Glossary of Technical Terms",
    }
    actual_headings = {p.text for p in check.paragraphs if p.style.name.startswith("Heading")}
    audit = {
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "source_unchanged": sha256(SOURCE) == EXPECTED_SOURCE_SHA256,
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "sections": len(check.sections),
        "portrait_letter": all(
            section.orientation == WD_ORIENT.PORTRAIT
            and round(section.page_width.inches, 2) == 8.50
            and round(section.page_height.inches, 2) == 11.00
            for section in check.sections
        ),
        "one_inch_margins": all(
            all(round(value.inches, 2) == 1.00 for value in (section.left_margin, section.right_margin, section.top_margin, section.bottom_margin))
            for section in check.sections
        ),
        "paragraph_count": len(check.paragraphs),
        "table_count": len(check.tables),
        "inline_image_count": len(check.inline_shapes),
        "media_hashes_preserved": source_images == final_images,
        "page_field_preserved": b"PAGE" in footer_xml,
        "required_headings_present": sorted(expected_headings - actual_headings) == [],
        "missing_required_headings": sorted(expected_headings - actual_headings),
        "stale_phrases_present": missing_terms,
        "direct_font_colors": sorted(direct_colors),
        "blue_direct_font_detected": any(color.upper() not in {"000000", "AUTO"} for color in direct_colors),
        "dense201_mean_present": b"84.168%" in document_xml,
        "dense169_mean_present": b"84.142%" in document_xml,
        "locked_test_boundary_present": "has not been evaluated on the locked test" in all_text,
        "render_qa": "Unavailable: LibreOffice absent and Microsoft Word PDF export stalled; structural fallback used.",
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    failures = [
        not audit["source_unchanged"],
        audit["sections"] != 1,
        not audit["portrait_letter"],
        not audit["one_inch_margins"],
        not audit["media_hashes_preserved"],
        not audit["page_field_preserved"],
        not audit["required_headings_present"],
        bool(audit["stale_phrases_present"]),
        audit["blue_direct_font_detected"],
        not audit["dense201_mean_present"],
        not audit["dense169_mean_present"],
        not audit["locked_test_boundary_present"],
    ]
    if any(failures):
        raise RuntimeError(f"structural audit failed; see {AUDIT}")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
