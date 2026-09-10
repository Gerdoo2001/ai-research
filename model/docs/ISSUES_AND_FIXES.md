# Issues and Fixes

## ISSUE-001 - Existing Python environment is incomplete and internally inconsistent

- **Timestamp:** 2026-07-16T16:59:25+08:00
- **Symptoms:** Several packages declared by `model/requirements.txt` are absent from the repository virtual environment. `pip check` reports TensorFlow/tensorboard protobuf constraints are violated.
- **Affected command or run ID:** Phase 0 environment snapshot; no training run affected yet.
- **Relevant error message:** `tensorflow 2.10.1 has requirement protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3` and the analogous tensorboard error.
- **Root cause:** The virtual environment does not match the declared requirements and the untracked working freeze contains versions outside TensorFlow 2.10.1 constraints.
- **Files changed:** Documentation only; no environment packages changed yet.
- **Fix applied:** Pending a minimal dependency plan after the binary audit.
- **How the fix was verified:** Not yet fixed.
- **Whether older results were invalidated:** Existing result files are not deleted, but reproducibility is not established.

## ISSUE-002 - Dataset root contains duplicated trees and macOS metadata

- **Timestamp:** 2026-07-16T16:59:25+08:00
- **Symptoms:** The source root contains top-level `train/val/test`, a nested `chest_xray/train/val/test`, and `__MACOSX` resource-fork records.
- **Affected command or run ID:** Phase 0 inventory; no new training run.
- **Relevant error message:** None; this is a data-integrity risk detected by audit.
- **Root cause:** The archive appears to have been extracted with both wrapper and direct contents retained, including macOS metadata.
- **Files changed:** Added audit/manifests outside the immutable source under `model/data_manifests`; source files were not changed.
- **Fix applied:** Verified 5,856 nested image files as SHA-256-identical copies, excluded macOS/other metadata, and excluded 32 redundant exact canonical copies. Built locked group-aware manifests from 5,824 retained unique images.
- **How the fix was verified:** Rehashed all 17,591 physical source files after generation with zero source mismatches; confirmed zero assigned exact-hash leakage and zero assigned group-ID leakage; locked manifest version `three_class_seed42_3b815cf769c8`.
- **Whether older results were invalidated:** Existing binary run provenance is insufficient to determine whether duplicates influenced training; those results will not be treated as three-class evidence.

## ISSUE-003 - Progress DOCX cannot be visually rendered in the available environment

- **Timestamp:** 2026-07-16T17:31:07+08:00
- **Symptoms:** The DOCX generator succeeds and structural audits pass, but the required page-rendering step cannot produce PNG/PDF pages.
- **Affected command or run ID:** Phase 1/2 `Pneumonia_Research_Progress_Report.docx` verification.
- **Relevant error message:** Packaged `render_docx.py` raised `FileNotFoundError` because LibreOffice is unavailable. A Microsoft Word COM PDF export did not complete within 60 seconds and was terminated.
- **Root cause:** No supported headless office renderer is installed. Desktop Word is installed but did not provide a reliable unattended export in this session.
- **Files changed:** No source or report content was changed by the failed render attempts.
- **Fix applied:** Used the document skill's structural fallback: ZIP integrity, relationship/media, section, image, style, and heading-hierarchy audits. Do not claim visual page QA.
- **How the fix was verified:** The current DOCX opens as a valid Office package, contains the expected embedded media, and has no heading-level jumps. Visual layout verification remains pending an available renderer.
- **Whether older results were invalidated:** No research metrics are affected. Report layout should be treated as structurally verified, not visually verified.

## ISSUE-004 - Audit plotting is unavailable in the repository TensorFlow environment

- **Timestamp:** 2026-07-16T17:31:07+08:00
- **Symptoms:** Dataset audit tables generated successfully, but the repository virtual environment and bundled document runtime do not provide Matplotlib.
- **Affected command or run ID:** Phase 2 dataset audit figure generation.
- **Relevant error message:** `ModuleNotFoundError: No module named 'matplotlib'` in the attempted runtimes.
- **Root cause:** The working environments do not match a single fully installed dependency lock.
- **Files changed:** Generated figures only under `model/results/dataset_audit`.
- **Fix applied:** Ran the deterministic plotting script with the installed system Python 3.12 runtime, which has Matplotlib. No model training or metrics were computed in that alternate runtime.
- **How the fix was verified:** `class_counts.png`, `image_size_distributions.png`, and their figure manifest were created and visually inspected.
- **Whether older results were invalidated:** No. Phase 3 subsequently created a separate pinned PyTorch CUDA environment; the legacy environment remains unchanged.

## ISSUE-005 - Official YOLOv5 classifier import required an undeclared package

- **Timestamp:** 2026-07-16T18:00:25+08:00
- **Symptoms:** The first isolated YOLOv5 CUDA model smoke failed during official source import.
- **Affected command or run ID:** `gpu_yolo_smokes.json`, first `yolov5_cls` attempt.
- **Relevant error message:** `ModuleNotFoundError: No module named 'seaborn'`.
- **Root cause:** The initial isolated dependency file covered Ultralytics but omitted Seaborn, which the pinned official YOLOv5 utilities import.
- **Files changed:** `model/requirements.multiclass-cu121.txt`.
- **Fix applied:** Pinned and installed `seaborn==0.13.2`.
- **How the fix was verified:** `pip check` passes; YOLOv5 CUDA forward/backward smoke and corrected end-to-end training/inference smoke pass.
- **Whether older results were invalidated:** The failed isolated model smoke has no metrics. It remains recorded in the overwritten machine-readable smoke audit history only through documentation; no benchmark result was affected.

## ISSUE-006 - YOLOv8 evaluation output violated the common logits contract

- **Timestamp:** 2026-07-16T18:02:12+08:00
- **Symptoms:** YOLOv8 passed training-mode forward/backward but failed the end-to-end output assertion when the model was switched to evaluation mode.
- **Affected command or run ID:** `smoke_yolov8_cls_e2e`.
- **Relevant error message:** `AttributeError: 'tuple' object has no attribute 'shape'`.
- **Root cause:** Ultralytics classification returns logits during training and `(probabilities, logits)` during evaluation. The common interface expected logits in both modes.
- **Files changed:** `model/src/multiclass/models.py`; tests and documentation.
- **Fix applied:** Added a classification-only logits adapter that selects the raw-logit element in evaluation tuples and rejects detection heads.
- **How the fix was verified:** YOLOv8 CPU output test, CUDA forward/backward smoke, retry-2 end-to-end training, artifact generation, checkpoint reload, and inference probability/class-order checks pass.
- **Whether older results were invalidated:** The failed run contains no benchmark result. The pre-final-adapter/hash retry artifacts are superseded diagnostics; `smoke_yolov8_cls_e2e_retry2` is the verified smoke.

## ISSUE-007 - Adding a test-only interruption field changed diagnostic configuration hashes

- **Timestamp:** 2026-07-16T18:04:37+08:00
- **Symptoms:** A corrected YOLOv8 smoke checkpoint could not pass the later strict embedded-configuration hash check after the controlled-resume diagnostic field was added.
- **Affected command or run ID:** `smoke_yolov8_cls_e2e_retry1` inference check.
- **Relevant error message:** `checkpoint configuration hash does not match its embedded configuration`.
- **Root cause:** The test-only `controlled_stop_after_epoch` field was initially included in the scientific configuration hash, so even its default `null` altered hashes created before the field existed.
- **Files changed:** `model/src/multiclass/config.py`; `model/src/multiclass/inference.py`.
- **Fix applied:** Excluded the run-control diagnostic field from scientific hashes and required inference to validate embedded checkpoint hashes. Reran YOLOv5 and YOLOv8 under the final contract.
- **How the fix was verified:** DenseNet121, `smoke_yolov5_cls_e2e_retry1`, and `smoke_yolov8_cls_e2e_retry2` all pass strict checkpoint load and normalized inference checks.
- **Whether older results were invalidated:** Retry 1 and the pre-adapter YOLOv5 smoke are superseded diagnostics, not benchmark evidence.
