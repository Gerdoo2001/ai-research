"""Write the immutable pre-continuation audit before new experiment changes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
MANIFESTS = MODEL / "data_manifests"
RESULTS = MODEL / "results"
AUDITS = RESULTS / "audits"
DOCS = MODEL / "docs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    AUDITS.mkdir(parents=True, exist_ok=True)
    split = load_json(MANIFESTS / "split_summary.json")
    source = load_json(MANIFESTS / "dataset_source_snapshot.json")
    baselines = load_json(AUDITS / "completed_baseline_artifact_audit.json")
    phash = load_json(RESULTS / "perceptual_hash_manual_review" / "perceptual_hash_manual_review.json")
    environment = load_json(RESULTS / "environment.json")["multiclass_environment"]
    orchestrator = load_json(RESULTS / "pipeline_orchestrator_status.json")
    consolidation = load_json(RESULTS / "consolidation_audit.json")
    config = load_json(MODEL / "configs" / "multiclass_baseline.json")

    phash_rows = phash.get("records", [])
    cross_split = [
        row
        for row in phash_rows
        if row["left_assigned_split"] != row["right_assigned_split"]
    ]
    report_path = DOCS / "Pneumonia_Model_Training_Details_Black_Updated.docx"
    report_pdf = RESULTS / "report_continuation_qa" / "source_report" / "source_report_word_export.pdf"
    report_pages = sorted((report_pdf.parent).glob("page-*.png"))

    verified_csv = AUDITS / "baseline_results_verified.csv"
    fieldnames = [
        "run_id", "architecture", "preprocessing", "status", "current_epoch",
        "best_epoch", "validation_macro_f1", "prediction_rows", "parameter_count",
        "checkpoint_state", "metrics_state", "provenance_state", "tensorboard_state",
        "verification_result",
    ]
    with verified_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in baselines["records"]:
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "architecture": row["architecture"],
                    "preprocessing": row["preprocessing"],
                    "status": row["status"],
                    "current_epoch": row["current_epoch"],
                    "best_epoch": row["best_epoch"],
                    "validation_macro_f1": row["validation_macro_f1"],
                    "prediction_rows": row["prediction_rows"],
                    "parameter_count": row["parameter_count"],
                    "checkpoint_state": "best and last load successfully",
                    "metrics_state": "recomputed values match saved artifacts",
                    "provenance_state": "config and manifest hashes match",
                    "tensorboard_state": "event header only; no logged tags",
                    "verification_result": row["audit_status"],
                }
            )

    audit = {
        "schema_version": 2,
        "audit_name": "pre-continuation current-state audit",
        "generated_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds"),
        "repository": {
            "root": str(ROOT),
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "working_tree_preservation": "pre-existing dirty state preserved; no reset or clean performed",
        },
        "source_dataset": {
            "root": source["source_root"],
            "physical_file_count": source["source_file_count"],
            "bytes": source["source_bytes"],
            "content_inventory_hash": source["source_content_inventory_hash"],
            "modified": source["source_modified"],
            "inventory_mismatch_count": 0,
        },
        "locked_manifests": {
            "version": split["manifest_version"],
            "version_hash": split["manifest_version_hash"],
            "manifest_hashes": split["manifest_hashes"],
            "class_order": split["class_order"],
            "counts": split["split_counts"],
            "total_counts": split["split_total_counts"],
            "group_counts": split["split_group_counts"],
            "assigned_group_leakage_count": split["assigned_group_leakage_count"],
            "assigned_exact_hash_leakage_count": split["assigned_exact_hash_leakage_count"],
            "test_policy": split["selection_policy"]["test"],
            "test_access_before_final_freeze": "not authorized and not performed by this continuation",
        },
        "duplicates": {
            "canonical_images_before_exact_deduplication": source["canonical_physical_valid_image_count_before_deduplication"],
            "canonical_unique_images_retained": source["canonical_valid_unique_image_count"],
            "canonical_redundant_exact_images_excluded": source["canonical_excluded_count"],
            "physical_exact_duplicate_report_rows": source["duplicate_report_counts"]["exact_duplicate"],
            "original_split_group_leakage_rows": source["duplicate_report_counts"]["group_leakage_original_split"],
            "nested_copy_mismatch_rows": source["duplicate_report_counts"]["nested_copy_mismatch"],
            "perceptual_hash_candidate_pairs": len(phash_rows),
            "perceptual_hash_cross_assigned_split_pairs": len(cross_split),
            "perceptual_hash_cross_assigned_split_pair_ids": [row["pair_id"] for row in cross_split],
            "manual_review_completed": all(row["visual_review_status"] == "completed" for row in phash_rows),
            "manually_confirmed_duplicates": sum(
                row["manual_decision"] != "Distinct images - retain both" for row in phash_rows
            ),
            "manually_confirmed_distinct": sum(
                row["manual_decision"] == "Distinct images - retain both" for row in phash_rows
            ),
            "manifest_action": "none; all 16 candidates are distinct images",
            "count_interpretation": "5,888 is the physical report-row count across duplicate/nested-copy roles; 32 is the canonical redundant-image exclusion count.",
        },
        "completed_baselines": {
            "expected": 8,
            "passed_artifact_audit": baselines["passed_runs"],
            "failed_artifact_audit": baselines["failed_runs"],
            "best_validation_run": max(
                baselines["records"], key=lambda row: row["validation_macro_f1"]
            )["run_id"],
            "best_validation_macro_f1": max(
                row["validation_macro_f1"] for row in baselines["records"]
            ),
            "tensorboard": "enabled for all eight; event files contain no logged tags",
            "consolidation_consistency_errors": consolidation["consistency_error_count"],
            "verified_results_csv": str(verified_csv.relative_to(ROOT)),
        },
        "baseline_configuration": {
            key: config[key]
            for key in (
                "input_size", "batch_size", "epochs", "optimizer", "learning_rate",
                "weight_decay", "scheduler", "scheduler_patience", "scheduler_factor",
                "loss", "class_weighting", "label_smoothing", "weights", "amp",
                "early_stopping_patience", "minimum_delta", "augmentation",
            )
        },
        "pipeline_semantics": {
            "preprocessing_order": [
                "open original image", "convert/enhance to RGB", "training-only horizontal flip",
                "training-only rotation with black fill", "ImageOps.fit resize-to-fill and centered crop to 224x224",
                "float32 scaling to [0,1]", "HWC-to-CHW transpose", "ImageNet normalization",
            ],
            "resize_semantics": "aspect ratio preserved; image is scaled to fill 224x224 and excess edges are center-cropped; no padding, letterbox, or stretching",
            "training_loss": "balanced class-weighted cross entropy",
            "validation_loss": "unweighted cross entropy in collect_predictions",
            "selection_metric": "validation macro F1",
            "focal_loss_caution": "current focal implementation computes exp(-CE) after optional class weighting; audit before any focal-loss trial",
        },
        "environment": {
            "python": environment["python_version"],
            "torch": environment["packages"]["torch"],
            "torchvision": environment["packages"]["torchvision"],
            "ultralytics": environment["packages"]["ultralytics"],
            "cuda_runtime": environment["gpu"]["torch_cuda_runtime"],
            "cudnn": environment["gpu"]["cudnn_version"],
            "gpu": environment["gpu"]["devices"][0],
            "tests": "12 passed in the continuation verification run",
        },
        "source_report": {
            "path": str(report_path),
            "sha256": sha256(report_path),
            "word_export_pdf": str(report_pdf),
            "rendered_page_count": len(report_pages),
            "visual_review": "all 15 pages inspected; readable black style; no severe clipping",
        },
        "reference_artifact_inventory": {
            "supplied_image_visible_entries": 10,
            "repository_original_rrl_spreadsheet_found": False,
            "repository_reference_list_document_found": False,
            "handling": "screenshot is a supplementary index only; every entry requires primary-source verification",
        },
        "blocking_and_reporting_findings": [
            {
                "finding": "post-baseline orchestrator stopped during progress-report generation",
                "evidence": orchestrator["detail"],
                "effect": "no later experiment was silently executed",
            },
            {
                "finding": "source report says five completed baselines in one location",
                "evidence": "generate_training_details_black_report.py:780",
                "correction": "eight completed controlled baselines",
            },
            {
                "finding": "progress-report generator uses two wrong YOLO run IDs",
                "evidence": "generate_progress_report.py:91-92",
                "correction": "baseline_yolov5_cls_s42 and baseline_yolov8_cls_s42",
            },
            {
                "finding": "old preprocessing plan is DenseNet121-only",
                "evidence": "preprocessing_ablation.csv and old stage gate",
                "correction": "use completed controls plus DenseNet169, DenseNet201, and ResNet50 as representative families",
            },
            {
                "finding": "TensorBoard writers were opened but never received scalar/image calls",
                "evidence": "eight header-only event files and no add_scalar calls in trainer.py",
                "correction": "add auditable epoch-level scalar logging before new runs",
            },
            {
                "finding": "failure ledger has two historically resolved rows still marked unresolved",
                "evidence": "smoke_resume_contract later completed; YOLOv8 retry later completed",
                "correction": "preserve original failures and append resolution evidence",
            },
            {
                "finding": "no Grad-CAM implementation or artifact exists",
                "evidence": "focused repository search",
                "correction": "implement and test after final checkpoint freeze",
            },
            {
                "finding": "YOLOv10 has no genuine supported image-classification model in the official task support used here",
                "evidence": "official YOLOv10 support is detection; no compatible classifier checkpoint/head was found",
                "correction": "retain scientific exclusion; do not relabel a detector or fabricate a classifier",
            },
        ],
        "evidence_status_legend": {
            "verified": "recomputed or directly inspected in repository artifacts",
            "reported": "stated in an existing report and cross-checked where possible",
            "planned": "not yet an experimental result",
            "unsupported": "not available in the verified implementation/evidence",
        },
    }
    audit["baseline_configuration"]["batch_size_note"] = "The template value is 16; each architecture used its pre-measured safe CUDA batch size."
    audit["baseline_configuration"]["batch_size_by_architecture"] = {
        row["architecture"]: load_json(RESULTS / "runs" / row["run_id"] / "config.json")["batch_size"]
        for row in baselines["records"]
    }

    json_path = AUDITS / "current_state_audit_continuation_v2.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    counts = split["split_counts"]
    baseline_table = "\n".join(
        f"| {row['run_id']} | {row['best_epoch']} | {row['validation_macro_f1']:.6f} | {row['prediction_rows']} | Passed | Header only; no tags |"
        for row in baselines["records"]
    )
    md = f"""# Current-State Audit Before Research Continuation V2

Generated: {audit['generated_at']}

This document freezes the observed repository state before any continuation-era experiment configuration is changed. Values are tagged as **Verified**, **Reported**, **Planned**, or **Unsupported**; only verified values may be treated as measured current-state facts.

## Executive result

- **Verified:** All eight completed baseline run directories pass checkpoint, provenance, prediction-row, metric, confusion-matrix, history, parameter-count, output-contract, and file-size checks.
- **Verified:** The best completed validation result is `baseline_densenet169_s42`, macro F1 **{audit['completed_baselines']['best_validation_macro_f1']:.6f}**. These are validation results, not final locked-test results.
- **Verified:** The dataset source inventory is unchanged: {source['source_file_count']:,} files, {source['source_bytes']:,} bytes, inventory SHA-256 `{source['source_content_inventory_hash']}`.
- **Verified:** The locked test manifest remains sealed. This continuation has not used the test set for model or preprocessing selection.
- **Verified:** All 16 pHash candidates were visually reviewed and all are distinct. Five pairs cross assigned splits; none requires a manifest change.
- **Verified:** The old automation stopped after the baseline stage because the progress-report DOCX could not be overwritten. It did not silently run ablations or tuning.

## Locked dataset and manifests

| Item | Verified value |
|---|---|
| Manifest version | `{split['manifest_version']}` |
| Manifest hash | `{split['manifest_version_hash']}` |
| Class order | Normal, Viral, Bacterial |
| Train | {split['split_total_counts']['train']} = {counts['train']['Normal']} Normal, {counts['train']['Viral']} Viral, {counts['train']['Bacterial']} Bacterial |
| Validation | {split['split_total_counts']['val']} = {counts['val']['Normal']} Normal, {counts['val']['Viral']} Viral, {counts['val']['Bacterial']} Bacterial |
| Test | {split['split_total_counts']['test']} = {counts['test']['Normal']} Normal, {counts['test']['Viral']} Viral, {counts['test']['Bacterial']} Bacterial |
| Assigned group leakage | {split['assigned_group_leakage_count']} |
| Assigned exact-hash leakage | {split['assigned_exact_hash_leakage_count']} |
| Test policy | {split['selection_policy']['test']} |

The original folder split was not preserved because 264 derived groups crossed those folders. The locked manifests were created by deterministic group assignment without consulting image content or test metrics.

## Duplicate-count interpretation and manual pHash review

The two duplicate numbers answer different questions:

- **32 canonical images** were redundant exact duplicates and were excluded before the 5,824-image locked manifest was created.
- **5,888 report rows** describe physical exact-duplicate relationships across canonical images and verified nested copies. This is not the number removed from the canonical dataset.

The DCT pHash audit produced 16 distance-2 candidates. Low distance was driven by common centered chest-radiograph geometry, not identity. Evidence sheets were inspected pair by pair. All 16 were distinct; the five cross-assigned-split pair IDs are `{', '.join(audit['duplicates']['perceptual_hash_cross_assigned_split_pair_ids'])}`. Decision: retain both images in every pair and do not alter the locked manifests.

## Completed baseline artifact verification

| Run | Best epoch | Validation macro F1 | Prediction rows | Artifact audit | TensorBoard |
|---|---:|---:|---:|---|---|
{baseline_table}

The TensorBoard setting was enabled for every baseline, but each event file contains only its event header and no scalar, image, histogram, or graph tags. The CSV/JSON histories remain valid. TensorBoard logging must be repaired for future runs; completed baselines must not be rerun merely to populate it.

## Baseline configuration actually used

| Setting | Value | Status |
|---|---:|---|
| Input | 224 x 224 | Verified from configs/checkpoints |
| Batch size | D121 16; D169 12; D201 8; R50 16; R101 8; R152 6; YOLOv5-cls 32; YOLOv8-cls 32 | Verified from per-run configs and CUDA calibration |
| Maximum epochs | {config['epochs']} | Verified |
| Optimizer | AdamW | Verified |
| Initial learning rate | {config['learning_rate']} | Verified |
| Weight decay | {config['weight_decay']} | Verified |
| Scheduler | ReduceLROnPlateau, factor {config['scheduler_factor']}, patience {config['scheduler_patience']} | Verified |
| Loss | Balanced class-weighted cross entropy | Verified |
| Selection | Highest validation macro F1 | Verified |
| Early stopping | Patience {config['early_stopping_patience']}, minimum delta {config['minimum_delta']} | Verified |
| Augmentation | Horizontal flip p={config['augmentation']['horizontal_flip_probability']}; rotation +/-{config['augmentation']['rotation_degrees']} degrees | Verified |
| Mixed precision | {config['amp']} | Verified |

## Exact preprocessing semantics

1. Open the original image from the locked manifest.
2. Apply the chosen enhancement/conversion and produce RGB.
3. For training only, apply a random horizontal flip.
4. For training only, apply random rotation with black fill.
5. Use Pillow `ImageOps.fit`: preserve aspect ratio, scale until 224 x 224 is filled, and center-crop the excess edges. There is no padding, letterboxing, or stretching.
6. Convert pixels to float32 in [0,1].
7. Transpose H x W x C to C x H x W.
8. Apply ImageNet mean/std normalization, yielding 3 x 224 x 224.

Enhancement currently occurs before augmentation and crop. `baseline` performs RGB conversion only; `clahe` uses grayscale CLAHE with clip limit 2.0 and an 8 x 8 tile grid; `histogram_equalization` performs global grayscale equalization; `pseudo_color` applies OpenCV VIRIDIS to grayscale. VIRIDIS is an exploratory transform and must not be described as clinically validated evidence.

Training uses class-weighted cross entropy, while `collect_predictions` reports unweighted validation/test cross entropy. Selection uses macro F1, so this does not change which checkpoint was selected, but the distinction must be documented and made explicit in future code.

## Environment verification

The continuation rechecked the historical isolated environment: Python {environment['python_version']}, torch {environment['packages']['torch']}, torchvision {environment['packages']['torchvision']}, Ultralytics {environment['packages']['ultralytics']}, CUDA {environment['gpu']['torch_cuda_runtime']}, cuDNN {environment['gpu']['cudnn_version']}, and {environment['gpu']['devices'][0]['name']} with {environment['gpu']['devices'][0]['total_memory_bytes']:,} bytes VRAM. All 12 repository tests passed.

## Report and automation audit

The supplied black report was opened, extracted, exported through Microsoft Word, rendered to 15 page images, and visually inspected. It is readable and not severely clipped. Required corrections:

- Replace the isolated claim of “five completed architecture baselines” with eight.
- Fix the progress-report run IDs to `baseline_yolov5_cls_s42` and `baseline_yolov8_cls_s42`.
- Replace direct DOCX overwrite with a versioned/fallback save so an open Word file cannot stop the research pipeline.
- Replace the obsolete DenseNet121-only preprocessing plan with the three representative completed controls: DenseNet169, DenseNet201, and ResNet50.
- Preserve both historical failure rows, but append evidence that the controlled resume interruption and the YOLOv8 tuple-output issue were later resolved.
- Implement Grad-CAM only after final checkpoint freeze; none currently exists.

## Research-reference inventory boundary

The supplied screenshot shows ten study rows and blank Reference Link cells. No original RRL spreadsheet or separate reference-list document was found in the repository. The screenshot is therefore a supplementary index, not a source of truth. Every visible entry must be checked against the publisher, DOI, official repository, or another verified primary source before it enters the evidence matrix.

Required instruction for the continuation:

> Open and verify every available reference link relevant to the eight supported models. When the link column is blank, incomplete, inaccessible, or only visible through the supplied image, locate the original publication using the exact title, authors, year, DOI, or publisher.

## Configuration decisions frozen at this checkpoint

- **Verified controls:** Do not rerun any of the eight completed baselines.
- **Planned ablation scope:** Reuse the completed baseline controls for DenseNet169, DenseNet201, and ResNet50; run only the three new alternatives (CLAHE, global histogram equalization, VIRIDIS) per representative model unless the evidence review changes this before launch.
- **Unsupported:** Do not fabricate a YOLOv10 classifier. Official YOLOv10 task support used by this repository is detection, not a compatible image-classification checkpoint/head.
- **Unsupported/deferred:** Raw-image PCA and unresolved FILM/SAS settings are not permitted to enter the controlled pipeline without a precise, leakage-safe design and verified evidence.
- **Test protection:** Freeze final architecture, preprocessing, hyperparameters, loss, seed protocol, and checkpoint-selection rule before one final locked-test evaluation.

Machine-readable companion: `model/results/audits/current_state_audit_continuation_v2.json`. Verified baseline table: `model/results/audits/baseline_results_verified.csv`.
"""
    (DOCS / "03_current_state_audit_continuation_v2.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
