# Three-Class Implementation Notes

## Implementation Outcome

A parallel PyTorch pipeline now supports manifest-driven binary or three-class chest X-ray classification without changing the legacy TensorFlow entry points or immutable source dataset. Multiclass output order is locked to Normal, Viral, Bacterial. The implementation supports DenseNet121/169/201, ResNet50/101/152, official YOLOv5s classification, and YOLOv8n classification. YOLOv10 classification is explicitly unsupported because the installed implementation supplies YOLOv10 detection heads only.

The code is under `model/src/multiclass`, with Windows-safe entry points under `model/scripts` and configuration under `model/configs`. Long runs use stable run directories under `model/results/runs`.

## Environment Isolation

The original repository `venv` remains unchanged and continues to provide TensorFlow 2.10.1 GPU support for the legacy binary models. The new pipeline uses a git-ignored `.venv` created from Python 3.12.6 and pinned by `model/requirements.multiclass-cu121.txt`.

Verified core versions are PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, Ultralytics 8.3.168, CUDA runtime 12.1, and cuDNN 9.1. `pip check` passes. PyTorch detects the NVIDIA GeForce RTX 2050 with 4,294,443,008 bytes reported device memory and compute capability 8.6. The environment occupies 5,270,688,006 bytes and is recorded alongside the legacy profile in `model/results/environment.json`.

## Configuration Contract

`ExperimentConfig` validates and hashes scientific/run settings. The default JSON exposes:

- task mode and exact class order;
- architecture, initialization source, and optional backbone freezing;
- preprocessing, input size, batch size, epochs, and augmentation;
- optimizer, learning rate, weight decay, scheduler, loss, class weighting, and label smoothing;
- seed, Windows-safe worker count, device, explicit CPU fallback, and AMP;
- early stopping, checkpoint/report intervals, manifest path, results path, and smoke limits.

Run IDs may be supplied or derived from architecture, preprocessing, seed, and configuration hash. Run IDs and the controlled-interruption diagnostic switch are excluded from the scientific configuration hash. Paths are resolved with `pathlib` from the repository root.

Binary mode uses two logits with Normal/Pneumonia mapping and collapses Viral/Bacterial manifest labels into Pneumonia. The original one-sigmoid TensorFlow binary pipeline remains unchanged; this new binary mode is an additional consistent interface, not a silent replacement.

## Locked Data Interface

Every loader validates the manifest version, every stored manifest SHA-256, and the Normal/Viral/Bacterial class order before reading samples. Training loads only `train_manifest.csv`; validation loads only `val_manifest.csv`. The test loader raises `PermissionError` unless `allow_locked_test=True` is explicitly provided through the evaluation entry point.

The dataset opens immutable source paths directly with Pillow, converts the requested transform to a normalized tensor, and returns the label plus a de-identified relative path. Smoke subsets are deterministic and class-balanced. `num_workers` defaults to zero for Windows reliability and is configurable.

## Preprocessing Interface

The current independent transforms are:

- Baseline: safe Pillow RGB conversion, centered fit to the configured square input, float tensor conversion, and ImageNet mean/standard-deviation normalization.
- CLAHE: grayscale luminance, OpenCV CLAHE with clip limit 2.0 and 8 by 8 tiles, RGB replication, resize, and ImageNet normalization.
- Global histogram equalization: Pillow `ImageOps.equalize` on grayscale, RGB replication, resize, and normalization.
- Pseudo-color: OpenCV VIRIDIS color map applied to grayscale, RGB conversion, resize, and normalization. No additional clinical information is claimed.

Training-only augmentation is a configurable horizontal flip plus a conservative rotation. Validation/test transforms are deterministic. PCA is deferred to a justified training-only feature-space experiment; raw-image PCA is not implemented. FILM remains unresolved and is not implemented.

## Model Factory and Initialization

Torchvision supplies the six DenseNet/ResNet models. Each classifier head is replaced with exactly three outputs for multiclass or two for binary. ImageNet default weights are recorded by enum name; `weights=none` provides random initialization for offline smoke tests.

The audited binary DenseNet checkpoints are Keras HDF5 models with one-sigmoid heads and cannot be directly loaded into torchvision PyTorch modules. No legacy-binary initialization result is claimed. Such an experiment would require a separate, exhaustively verified layer-by-layer conversion; the standard baselines use official ImageNet weights instead.

YOLOv5 uses the official `yolov5s-cls` implementation at pinned source commit `ed3bdee6bbe696ce9b35f32efb261e83f72ce03c` and the v7.0 classification checkpoint. The checkpoint SHA-256 is `53ebfe1c19d60bea82c821b93b52bf58d81d5a06b693668e516fd8d44769881e`.

YOLOv8 uses Ultralytics 8.3.168's genuine `yolov8n-cls.yaml` and the v8.3.0 `yolov8n-cls` asset. The checkpoint SHA-256 is `11fa19f2aea79bc960d680a13f82f22105982b325eb9e17a4a5e1a9f8245980a`.

Pinned external source and weights are prepared under git-ignored `.research_cache` by `prepare_yolo_assets.py`; their source, versions, sizes, and hashes are duplicated in `model/results/yolo_support_audit.json`.

Ultralytics classification returns logits during training but `(probabilities, logits)` during evaluation. A narrow adapter normalizes this to raw logits for the common loss/evaluation interface. It rejects non-`Classify` heads, preventing a YOLO detection configuration from being mislabeled as classification.

The installed package's YOLOv10 YAMLs terminate in `v10Detect`, and no YOLOv10 classification YAML/checkpoint is present. Forcing `task='classify'` still produced a detection head during the support audit. YOLOv10 is therefore unsupported rather than trained as an invalid object detector.

## Training and Resume Behavior

The trainer:

- requires CUDA when configured unless CPU fallback is explicitly enabled;
- sets Python, NumPy, PyTorch, and CUDA seeds and deterministic cuDNN behavior;
- uses current `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` APIs;
- supports balanced class weights, cross-entropy or focal loss, Adam/AdamW/SGD, cosine or validation-macro-F1 plateau scheduling;
- selects the best checkpoint on validation macro F1 and applies early stopping;
- atomically writes status/config/metrics JSON and checkpoints;
- flushes `history.csv` and appends the Markdown training journal every epoch;
- keeps best and last checkpoints for each run;
- automatically skips an already completed identical run;
- records failures and interruptions without deleting their run directories.

Checkpoints include model, optimizer, scheduler, scaler, configuration/hash, history, best epoch/metric, and Python/NumPy/PyTorch/CUDA RNG states. A controlled interruption test stopped after epoch 1, marked the run interrupted, and resumed at epoch 2 from `last_checkpoint.pt` with both rows retained in `history.csv`.

## Evaluation and Inference

The evaluation layer calculates accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/specificity/F1/support, confusion matrix, one-vs-rest ROC-AUC, macro/per-class PR-AUC, MCC, Cohen kappa, log loss, expected calibration error, and multiclass Brier score when valid.

Completed run artifacts include configuration/status/model metadata, history, metrics, class report, predictions, error samples, confusion matrix CSV/PNG, ROC/PR/calibration plots, training curves, best/last checkpoints, and optional TensorBoard logs. Inference timing records hardware, batch/input size, warmup/measured iterations, and whether preprocessing/transfer are included.

`MulticlassPredictor` validates the checkpoint's embedded configuration hash and returns predicted index/class, per-class probabilities, confidence, class mapping, task mode, and configuration hash. DenseNet121, corrected YOLOv5, and corrected YOLOv8 smoke checkpoints were loaded on CUDA and returned probability sums of one in the locked class order.

## Entry Points

- `train_multiclass.py`: train, smoke, or resume one configuration.
- `evaluate_multiclass.py`: validation evaluation or explicit locked-test evaluation.
- `run_benchmark.py`: sequential matrix execution that continues after independent failures.
- `resolve_run_matrix.py`: validation-only stage gates that resolve focused, tuning, robustness, and locked-test placeholders from persisted evidence.
- `run_tuning.py`: resumable Optuna 4.9.0 TPE studies with one normal checkpointed run per trial and SQLite/CSV/JSON trial persistence.
- `generate_final_report.py`: completion-gated Markdown/DOCX generation from consolidated and original per-run artifacts with a structural package audit.
- `run_research_pipeline.py`: a stop-on-failure, resumable stage orchestrator that waits for an active baseline sequence and then invokes only the guarded preprocessing, selection, tuning, robustness, test, verification, and reporting entry points.
- `gpu_smoke_models.py`: isolated CUDA forward/backward/output-contract tests.
- `prepare_yolo_assets.py`: pinned YOLO source/weight preparation and hash audit.
- `snapshot_multiclass_environment.py`: append the isolated environment profile.

All multiprocessing-capable scripts protect entry points with `if __name__ == "__main__"`.

The final locked-test path has a second gate beyond the manifest loader. `resolve_run_matrix.py --stage test` first requires completed robustness runs at seeds 42, 123, and 2026 and writes `model/results/final_selection.json` from validation evidence only. Test evaluation then requires the matching frozen selection file, the exact representative checkpoint, and explicit `--allow-locked-test`; it records a single `final_locked_test` run and changes the selection lock to `locked_test_evaluated`.

## Compatibility and Scope Boundaries

No file under `model/datasets/chest_xray`, no legacy source under `model/src` outside the new package, no legacy HDF5 file, and no FastAPI inference contract was modified. API integration is intentionally deferred until a final three-class checkpoint and mapping are selected.

Smoke metrics are pipeline diagnostics based on 12 training and 12 validation images with random initialization. They are not architecture comparisons and must not be used for model selection. Pre-adapter YOLO checkpoints and the preserved failed YOLOv8 run are superseded diagnostic artifacts, not benchmark evidence.
