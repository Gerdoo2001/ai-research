# Current-State Audit Before Research Continuation V2

Generated: 2026-08-02T12:41:34+08:00

This document freezes the observed repository state before any continuation-era experiment configuration is changed. Values are tagged as **Verified**, **Reported**, **Planned**, or **Unsupported**; only verified values may be treated as measured current-state facts.

## Executive result

- **Verified:** All eight completed baseline run directories pass checkpoint, provenance, prediction-row, metric, confusion-matrix, history, parameter-count, output-contract, and file-size checks.
- **Verified:** The best completed validation result is `baseline_densenet169_s42`, macro F1 **0.840293**. These are validation results, not final locked-test results.
- **Verified:** The dataset source inventory is unchanged: 17,591 files, 2,483,915,296 bytes, inventory SHA-256 `4ee6dd784207a910043f62cff542a5d805c312fcddafa90ce1a058d03402f734`.
- **Verified:** The locked test manifest remains sealed. This continuation has not used the test set for model or preprocessing selection.
- **Verified:** All 16 pHash candidates were visually reviewed and all are distinct. Five pairs cross assigned splits; none requires a manifest change.
- **Verified:** The old automation stopped after the baseline stage because the progress-report DOCX could not be overwritten. It did not silently run ablations or tuning.

## Locked dataset and manifests

| Item | Verified value |
|---|---|
| Manifest version | `three_class_seed42_3b815cf769c8` |
| Manifest hash | `3b815cf769c85658a04748da5eb2e15b981b89d9c9b77a404e90fe23473d2774` |
| Class order | Normal, Viral, Bacterial |
| Train | 4076 = 1105 Normal, 1039 Viral, 1932 Bacterial |
| Validation | 874 = 237 Normal, 223 Viral, 414 Bacterial |
| Test | 874 = 237 Normal, 223 Viral, 414 Bacterial |
| Assigned group leakage | 0 |
| Assigned exact-hash leakage | 0 |
| Test policy | locked until final selected configuration |

The original folder split was not preserved because 264 derived groups crossed those folders. The locked manifests were created by deterministic group assignment without consulting image content or test metrics.

## Duplicate-count interpretation and manual pHash review

The two duplicate numbers answer different questions:

- **32 canonical images** were redundant exact duplicates and were excluded before the 5,824-image locked manifest was created.
- **5,888 report rows** describe physical exact-duplicate relationships across canonical images and verified nested copies. This is not the number removed from the canonical dataset.

The DCT pHash audit produced 16 distance-2 candidates. Low distance was driven by common centered chest-radiograph geometry, not identity. Evidence sheets were inspected pair by pair. All 16 were distinct; the five cross-assigned-split pair IDs are `near_00f755ace402461b, near_9baff834314599d5, near_108295e9cc425d75, near_46a898efcc56e29e, near_90cf97b9b16e2b4e`. Decision: retain both images in every pair and do not alter the locked manifests.

## Completed baseline artifact verification

| Run | Best epoch | Validation macro F1 | Prediction rows | Artifact audit | TensorBoard |
|---|---:|---:|---:|---|---|
| baseline_densenet121_s42 | 10 | 0.837745 | 874 | Passed | Header only; no tags |
| baseline_densenet169_s42 | 6 | 0.840293 | 874 | Passed | Header only; no tags |
| baseline_densenet201_s42 | 10 | 0.838713 | 874 | Passed | Header only; no tags |
| baseline_resnet50_s42 | 5 | 0.836919 | 874 | Passed | Header only; no tags |
| baseline_resnet101_s42 | 9 | 0.837300 | 874 | Passed | Header only; no tags |
| baseline_resnet152_s42 | 6 | 0.833275 | 874 | Passed | Header only; no tags |
| baseline_yolov5_cls_s42 | 11 | 0.794705 | 874 | Passed | Header only; no tags |
| baseline_yolov8_cls_s42 | 11 | 0.773396 | 874 | Passed | Header only; no tags |

The TensorBoard setting was enabled for every baseline, but each event file contains only its event header and no scalar, image, histogram, or graph tags. The CSV/JSON histories remain valid. TensorBoard logging must be repaired for future runs; completed baselines must not be rerun merely to populate it.

## Baseline configuration actually used

| Setting | Value | Status |
|---|---:|---|
| Input | 224 x 224 | Verified from configs/checkpoints |
| Batch size | D121 16; D169 12; D201 8; R50 16; R101 8; R152 6; YOLOv5-cls 32; YOLOv8-cls 32 | Verified from per-run configs and CUDA calibration |
| Maximum epochs | 12 | Verified |
| Optimizer | AdamW | Verified |
| Initial learning rate | 0.0001 | Verified |
| Weight decay | 0.0001 | Verified |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 | Verified |
| Loss | Balanced class-weighted cross entropy | Verified |
| Selection | Highest validation macro F1 | Verified |
| Early stopping | Patience 4, minimum delta 0.0001 | Verified |
| Augmentation | Horizontal flip p=0.5; rotation +/-5.0 degrees | Verified |
| Mixed precision | True | Verified |

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

The continuation rechecked the historical isolated environment: Python 3.12.6, torch 2.5.1+cu121, torchvision 0.20.1+cu121, Ultralytics 8.3.168, CUDA 12.1, cuDNN 90100, and NVIDIA GeForce RTX 2050 with 4,294,443,008 bytes VRAM. All 12 repository tests passed.

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
