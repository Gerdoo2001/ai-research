# Testing and Validation

## Current Gate

Phase 3 implementation and smoke validation are complete, and the controlled Stage B benchmark is active. DenseNet121 has completed; DenseNet169 is running. The locked test manifest has not been loaded. Smoke metrics use tiny balanced subsets and random initialization and are not model-selection evidence.

## Environment and Legacy Artifact Checks

- Environment and repository inventories parse as valid JSON.
- TensorFlow 2.10.1 detects the NVIDIA RTX 2050 GPU in the repository virtual environment.
- All six complete legacy HDF5 models load with `compile=False`; their input/output shapes and parameter counts match stored configurations.
- Legacy report text, confusion matrices, and model family identities were cross-checked.
- `pip check` currently fails because protobuf 3.20.3 is outside TensorFlow/tensorboard 2.10.1's declared `<3.20` range. This must be resolved in the Phase 3 dependency plan.

## Dataset and Manifest Checks

- 17,591 physical source files were inventoried by path, size, and SHA-256.
- 5,856 canonical images decoded and had valid label evidence.
- 32 redundant exact canonical copies were excluded, leaving 5,824 retained images.
- Manifest coverage is exact: train 4,076 + validation 874 + test 874 = 5,824.
- All three classes occur in every split.
- Assigned split-group intersection count is zero.
- Assigned SHA-256 intersection count is zero.
- Locked manifests and class mapping have recorded SHA-256 hashes in `model/data_manifests/split_summary.json`.
- A post-generation immutability pass rehashed all source files and found zero missing, size-mismatched, or content-mismatched files.

## Figure and DOCX Checks

- Dataset class-count and image-size figures were generated from the locked manifest and visually inspected.
- The progress DOCX passes package/ZIP integrity, media relationship, section, style, and heading-hierarchy checks.
- Visual DOCX page rendering could not be completed because LibreOffice is absent and Word COM export did not complete. The report is structurally verified only; ISSUE-003 records this limitation.

## Phase 3 Required Tests

- Manifest/class-order/hash/count assertions pass for train 4,076, validation 874, and test 874.
- Locked-test access without explicit unlock raises `PermissionError`.
- Baseline, CLAHE, global histogram equalization, and pseudo-color transforms return finite 3 by 64 by 64 tensors in unit tests and completed real-data GPU smokes.
- All eight supported model heads emit exactly three logits in CPU tests. YOLOv10 classification raises `UnsupportedArchitectureError`.
- DenseNet121/169/201 and ResNet50/101/152 passed isolated 224 by 224 CUDA forward/backward smokes. Peak allocated memory at batch 1 ranged from 129,611,264 to 533,166,592 bytes.
- YOLOv5s-cls and YOLOv8n-cls passed isolated batch-2 CUDA forward/backward smokes with three-logit outputs.
- End-to-end GPU smokes completed for DenseNet, ResNet, YOLOv5, YOLOv8, and all implemented preprocessing families.
- A controlled interruption after epoch 1 was recorded and resumed at epoch 2; checkpoint/history/journal continuity passed.
- Best/last checkpoints, atomic status updates, ledger updates, metric artifacts, plots, and journal updates were verified.
- DenseNet, YOLOv5, and corrected YOLOv8 checkpoints load for inference and return probabilities summing to one in Normal/Viral/Bacterial order.

The current unittest suite contains ten passing tests. It covers the eight pipeline contracts plus Optuna completed/failed trial export and predecessor-stage gating. `compileall` and isolated-environment `pip check` pass.

## Stage B Evidence to Date

- `baseline_densenet121_s42` completed all 12 epochs, selected epoch 10, and produced every required validation artifact plus best/last checkpoints.
- Checkpoint-validated macro F1 is 0.837745, balanced accuracy is 0.827333, and accuracy is 0.854691.
- Per-class recall is 0.970464 Normal, 0.569507 Viral, and 0.942029 Bacterial. Viral-to-Bacterial confusion is the dominant error.
- Configuration, manifest, metrics, ledger, journal, and matrix identifiers agree; consolidated consistency errors are zero.
- The Stage B launcher moved to DenseNet169 without intervention. Its first epoch completed on CUDA and persisted validation macro F1 0.807401.

The Optuna dependency and later-stage orchestration scripts pass import/compile/help checks. Placeholder resolution is gated on completed predecessor evidence, and locked-test evaluation requires a frozen selection artifact plus explicit unlock. No tuning or test result is claimed yet.

## Preserved Failure and Superseded Diagnostics

`smoke_yolov8_cls_e2e` failed because Ultralytics evaluation mode returned a tuple. The run directory, error log, failure ledger row, and journal event remain. A logits adapter fixed the root cause and `smoke_yolov8_cls_e2e_retry2` completed and passed inference.

The first YOLOv5 completed smoke predates the adapter and is superseded by `smoke_yolov5_cls_e2e_retry1`. The first YOLOv8 retry predates the final diagnostic-hash exclusion and is superseded by retry 2. These runs remain historical pipeline diagnostics and are not benchmark evidence.

## Remaining Validation During the Benchmark

- Complete the remaining seven supported baseline runs and reconsolidate all per-run metrics.
- Run the three supported preprocessing ablations and validation-gated focused comparisons.
- Execute and persist the planned Optuna studies, three robustness seeds, and the one final locked-test evaluation after selection freezes.
- Re-run the full unit/consistency/inference/source-immutability gates before final reporting.
