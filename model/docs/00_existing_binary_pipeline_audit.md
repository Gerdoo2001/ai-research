# Existing Binary Pipeline Audit

**Audit completed:** 2026-07-16T17:05:59+08:00
**Repository commit:** `3dbd85ecbe8c408c858c84aa2b8e333c422f6192`
**Scope:** Existing files under `model`, related FastAPI inference code, dependency files, legacy checkpoints, text reports, and all 17 saved result figures.

## Executive finding

The only substantive chest-X-ray classifier is a TensorFlow/Keras binary transfer-learning script for DenseNet121, DenseNet169, and DenseNet201. It trains on a physically generated `model/augmented_data/train` tree containing CLAHE images and horizontal flips, uses a derived 80/20 training/validation split, freezes the complete ImageNet backbone, and emits one sigmoid output for `NORMAL` versus `PNEUMONIA`. It does not use the official test directory, does not save locked manifests, and does not implement patient-aware splitting, resumable checkpoints, early stopping, or configuration tracking.

The current generated training tree has 2,146 Normal images and only 243 Pneumonia images. That is inconsistent with the original dataset distribution and strongly suggests the augmentation job did not finish. The saved validation reports therefore describe an imbalanced, generated-data split with 429 Normal and 48 Pneumonia samples; they are retained as legacy artifacts but cannot serve as evidence for the new three-class benchmark.

No legacy implementation or definition was found for FILM, SAS Tuning, automated grading of chest X-ray images, PCA, histogram equalization, pseudo-coloring, YOLO, or ResNet. CLAHE is the only requested preprocessing concept implemented in the legacy code.

## 1. Existing project structure

- `model/src/augmentation.py`: offline binary CLAHE and horizontal-flip generator.
- `model/src/train_densenet.py`: binary DenseNet121/169/201 training and validation reporting.
- `model/src/model.py` and `model/src/train.py`: unrelated PyTorch `FakeData` demonstration.
- `model/src/unet/`: separate and incomplete lung-segmentation experiments.
- `model/src/app.py`: Streamlit comparison of two segmentation models.
- `model/augmented_data/train`: generated binary classifier input.
- `model/outputs`: five DenseNet model files/copies, two segmentation HDF5 artifacts, 17 PNG figures, and three classification reports.
- `web/backend/inference.py`: FastAPI-side binary DenseNet201 loading and inference.
- `model/datasets/chest_xray`: original source dataset, treated as read-only by the new work.

At the Phase 0 snapshot, `model` contained 20,023 files and approximately 4.63 GB. The full code and legacy-artifact inventory, including SHA-256 hashes and HDF5 metadata, is in `model/docs/00_code_inventory.json`.

## 2. Current binary workflow

1. `augmentation.py` discovers `.jpg`, `.jpeg`, and `.png` files in top-level source `train/NORMAL` and `train/PNEUMONIA` folders.
2. Each source image is loaded with OpenCV as BGR.
3. It is converted to LAB; CLAHE with clip limit 2.0 and an 8 x 8 tile grid is applied only to the L channel.
4. The CLAHE image and a horizontal flip of it are written as JPEG files to `model/augmented_data/train/<class>`.
5. `train_densenet.py` uses Keras `ImageDataGenerator(rescale=1/255, validation_split=0.2)` on that generated tree.
6. Directory names are mapped alphabetically by Keras: `NORMAL = 0`, `PNEUMONIA = 1`.
7. A DenseNet backbone with ImageNet weights and no top is instantiated, then global average pooling, dropout 0.3, and a one-unit sigmoid Dense layer are added.
8. Every backbone layer is frozen; only the final Dense head is trained.
9. The script trains each DenseNet variant for ten epochs and saves the final complete `.h5` model.
10. Predictions on the same derived validation subset are thresholded at 0.5 and used for reports and plots.
11. The FastAPI backend loads the saved DenseNet201 file, applies the same LAB-luminance CLAHE, resizes to 224 x 224, scales to `[0,1]`, and returns a binary label and sigmoid probability.

## 3. Existing entry points and exact commands

From the repository root:

```bat
venv\Scripts\python.exe model\src\augmentation.py
venv\Scripts\python.exe model\src\train_densenet.py
```

The training script has no `if __name__ == "__main__"` guard and begins data loading/training when imported. That behavior is unsafe for reusable modules and Windows multiprocessing.

The unrelated PyTorch demonstration can be launched with:

```bat
cd model\src
..\..\venv\Scripts\python.exe train.py
```

It is not a chest-X-ray training path and is currently nonfunctional in the repository environment because PyTorch/torchvision are absent. Independently, its classifier expects `64*6*6` features even though a 28 x 28 input produces `64*5*5`, so the forward pass would fail at the first fully connected layer.

## 4. Architectures and checkpoint inspection

The legacy classifier supports only:

- DenseNet121: 430 layers, 7,038,529 parameters, 1,025 trainable parameters.
- DenseNet169: 598 layers, 12,644,545 parameters, 1,665 trainable parameters.
- DenseNet201: 710 layers, 18,323,905 parameters, 1,921 trainable parameters.

All six complete saved classifier files loaded successfully with TensorFlow 2.10.1 using `compile=False`. Each expects `[batch, 224, 224, 3]` and returns `[batch, 1]`. The DenseNet121 files under `outputs/1`, `outputs/2`, and `outputs/densenet121` have distinct SHA-256 hashes and are not byte-identical copies, but no saved configuration links the two numbered runs to a command or dataset snapshot.

The two segmentation artifacts are separate: `trained_model.hdf5` loads with input/output `[batch, 256, 256, 1]`, while `cxr_reg_weights.best.hdf5` is weights-only.

## 5. Classifier head and output shape

The head is `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1, sigmoid)`. It is binary and cannot represent the requested three-class order. A valid multiclass head must return exactly three logits or probabilities in the stable order `Normal`, `Viral`, `Bacterial`.

## 6. Label mapping and derivation

The binary mapping is inferred exclusively from directory names. Keras alphabetical class discovery yields `NORMAL = 0` and `PNEUMONIA = 1`. Viral versus bacterial evidence in Pneumonia filenames is ignored.

## 7. Dataset split behavior

The script never loads the official `test` or `val` directories. It derives both fitting and selection data from `model/augmented_data/train` using `validation_split=0.2`. No split seed is declared and no manifest is saved. The split is not patient/group aware.

Because both a CLAHE image and its horizontal flip share the same patient/source, augmented relatives can cross the fitting/validation boundary or introduce ordering bias. In addition, multiple original files for the same `person<N>` patient can cross boundaries. The new pipeline must split canonical source images by patient/group before online augmentation.

## 8. Preprocessing and augmentation

- Offline CLAHE: LAB color space, L channel only, clip limit 2.0, 8 x 8 tile grid.
- Offline augmentation: unconditional horizontal flip of every processed image.
- Model input: resize to 224 x 224 and scale by `1/255`.
- No model-specific DenseNet `preprocess_input` is used.
- No online geometric, intensity, or regularization augmentation is used during training.
- The backend again applies CLAHE to uploaded source images, which matches the generated training-domain intent but depends on the legacy file having been trained on the current augmentation implementation.

## 9. Loss, optimizer, scheduler, and stopping

- Loss: `binary_crossentropy` in source. HDF5 metadata agrees for DenseNet121 and DenseNet169. The saved DenseNet201 metadata records the string `loss`, which is inconsistent with the inspected source and shows that artifact provenance is incomplete.
- Optimizer: Adam with learning rate `1e-4`.
- Scheduler: none.
- Early stopping: none.
- Warmup/freeze schedule: none; the complete backbone remains frozen for all ten epochs.

## 10. Checkpoint and resume behavior

Only the final complete model is saved. There is no best checkpoint, last checkpoint, per-epoch checkpoint, optimizer-resume metadata, status file, configuration hash, interruption handler, or resume command. Re-running the script can overwrite output paths.

## 11. Metrics and outputs

During training, Keras tracks accuracy, precision, recall, and ROC AUC. After training, the script saves:

- final `.h5` model;
- text classification report;
- confusion matrix PNG;
- classification-report heatmap;
- per-class precision/recall/F1 bar chart;
- ROC curve PNG;
- combined accuracy/loss curve PNG.

It does not save raw predictions, history CSV/JSON, model configuration, class mapping, per-class specificity, balanced accuracy, macro model-selection metrics, calibration, PR-AUC, MCC, Cohen's kappa, latency, throughput, GPU memory, or status metadata.

## 12. Legacy result interpretation

All 17 saved figures were visually inspected. The text and plotted confusion matrices agree:

- DenseNet121: 429/429 Normal recalled, 3/48 Pneumonia recalled; minority recall 0.0625; macro F1 approximately 0.53; plotted ROC AUC 0.981.
- DenseNet169: 429/429 Normal recalled, 19/48 Pneumonia recalled; minority recall approximately 0.40; macro F1 approximately 0.77; plotted ROC AUC 0.980.
- DenseNet201: 429/429 Normal recalled, 20/48 Pneumonia recalled; minority recall approximately 0.42; macro F1 approximately 0.78; plotted ROC AUC 0.998.

The high accuracy and AUC coexist with poor Pneumonia recall at the fixed threshold because the generated validation set is about 90% Normal. These are legacy diagnostics, not locked-test or three-class benchmark results. The plot-only histories under `outputs/1` and `outputs/2` lack raw values, run configuration, class report, or dataset hash.

## 13. GPU and mixed precision

TensorFlow automatically uses an available GPU. The DenseNet script does not explicitly require a GPU, check VRAM, stop on CPU fallback, enable memory growth, or enable mixed precision. By contrast, `train_unet.py` attempts TensorFlow memory growth. The current repository virtual environment sees the RTX 2050, but the DenseNet entry point does not record device/runtime facts.

## 14. Configuration system

There is no configuration file or CLI. Image size, batch size, epochs, learning rate, model list, paths, dropout, class mode, and split fraction are module-level constants. No configuration hash exists.

## 15. Inference compatibility

`web/backend/inference.py` is compatible with the legacy DenseNet201 one-sigmoid model and returns `label`, `probability`, and `threshold`. It is not compatible with three probabilities or the requested class mapping. Its `load_model` docstring incorrectly says DenseNet121 while the configured path points to DenseNet201. `schemas.py` also hard-codes the binary response meaning.

The new research inference adapter should be implemented and validated separately. The production-facing FastAPI response should not change until a selected checkpoint and stable three-class contract exist.

## 16. Hard-coded paths and working-directory sensitivity

The main DenseNet and augmentation scripts compute the repository root with `Path(__file__)`, which is reusable. Other legacy paths are fragile:

- `app.py` loads `../outputs/...` relative to the process working directory.
- `train.py` saves `../outputs/simple_cnn.pth` relative to the current directory.
- several U-Net modules use `parents[2]` from `model/src/unet`, which resolves to `model`, then append another `model` segment.
- `train_unet.py` imports `model.unet.data_augment`, but the repository file is named `date_augment.py`.
- `date_augment.py`, `five_metric.py`, and `best_threshold.py` reference names they do not import or define.
- `unet/evaluate.py` iterates files directly under `test` even though images are stored in class subdirectories.

The segmentation defects are outside the new classification scope and are documented rather than silently refactored.

## 17. Reproducibility, imbalance, and leakage risks

- no Python/NumPy/TensorFlow seed;
- no deterministic-operation configuration;
- no data manifest or manifest hash;
- no patient/group-aware split;
- no exact- or perceptual-duplicate audit;
- physically augmented relatives can contaminate validation;
- incomplete generated class tree produces a 2,146:243 imbalance;
- no class weighting or focal loss;
- no dataset/environment/config snapshot per run;
- no batch/epoch history persisted in machine-readable form;
- no locked test evaluation;
- no failure/status/resume tracking;
- nested physical dataset copies can be double-counted by broad recursive discovery.

## 18. Reusable parts

- `pathlib`-based repository-root discovery in the main classifier scripts;
- the verified LAB-luminance CLAHE parameters as one candidate transform;
- Keras ImageNet DenseNet construction and global pooling concept;
- TensorFlow 2.10.1 GPU runtime already working on this Windows machine;
- legacy HDF5 loading for an optional, explicitly separate binary-initialization experiment;
- FastAPI image validation and cached-model-loading pattern, after adapting the model contract.

## 19. Required three-class extensions

- canonical source-image manifests with patient-aware locked splits;
- filename-based Normal/Viral/Bacterial parsing with exclusions for ambiguous cases;
- shared class order and three-output head;
- configuration files and CLI entry points guarded by `__main__`;
- consistent model/preprocessing interfaces;
- training-only online augmentation;
- best/last checkpoints and resume support;
- GPU requirement checks and mixed precision where safe;
- macro F1-centered metrics and complete per-run artifacts;
- DenseNet and ResNet families in the verified TensorFlow environment;
- separately verified Ultralytics classification support if a CUDA PyTorch environment can be established without breaking TensorFlow;
- controlled preprocessing ablations and validation-only tuning;
- report generation from actual ledgers, histories, metrics, and figures.

## 20. Binary compatibility strategy

Keep all legacy files and artifacts in place. Add the new pipeline under `model/scripts`, `model/src/pneumonia_multiclass`, `model/configs`, `model/data_manifests`, and `model/results/runs`. The new configuration will support `task_mode = binary` or `multiclass`, but the legacy entry points will not be overwritten. Backend integration remains deferred until a final selected three-class checkpoint is available.

## 21. Requested concepts found or unresolved

- **CLAHE:** implemented exactly as LAB luminance-only CLAHE, clip limit 2.0, tile grid 8 x 8.
- **FILM:** no occurrence or definition in legacy code, documentation, dependencies, comments, outputs, or backend files. Unresolved; no implementation will be invented.
- **SAS Tuning:** no occurrence or SAS dependency/tooling. Unresolved; any future Optuna work will be labeled general hyperparameter tuning.
- **Automated grading of chest X-ray images tuning:** no occurrence, definition, label, or implementation. Unresolved.
- **PCA:** no occurrence or implementation.
- **Histogram equalization:** no implementation distinct from CLAHE.
- **Pseudo-coloring:** no implementation in the classifier. The segmentation UI uses a JET overlay for visualization, which is not classifier pseudo-color preprocessing.
- **YOLO classification:** no implementation or dependency.
- **ResNet:** no implementation.

## 22. Recommended implementation approach

Use a parallel manifest-driven pipeline and retain TensorFlow/Keras for DenseNet and ResNet because the existing Python 3.10 environment already exposes the NVIDIA GPU. First repair only the minimal audited dependency gaps. Build labels and locked group-aware splits from canonical source images before any augmentation. Apply augmentation online after the split. Keep YOLO classification behind an explicit optional dependency/support check so installing CUDA PyTorch cannot silently disturb the working TensorFlow environment. Every run must consume the same manifest hashes, persist best and last checkpoints, and be selected by validation macro F1 without consulting the locked test set.

The existing binary results remain preserved for historical comparison only; they do not influence three-class model or preprocessing selection.
