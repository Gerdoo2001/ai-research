# Decision Log

## DEC-001 - Preserve the original dataset as immutable input

- **Date:** 2026-07-16
- **Context:** The source directory contains top-level splits, a nested duplicate dataset tree, and macOS metadata/resource-fork files.
- **Decision:** Do not modify any file under `model/datasets/chest_xray`. Audit all physical copies, identify a canonical image set using path and content evidence, and place all locked manifests or generated layouts outside the source directory.
- **Alternatives considered:** Reorganize source folders in place; delete duplicated trees; train directly from every discovered image.
- **Consequences:** Audit and manifest generation take longer, but source integrity is preserved and duplication/leakage can be controlled.

## DEC-002 - Preserve the existing binary and segmentation implementations

- **Date:** 2026-07-16
- **Context:** The repository contains TensorFlow DenseNet binary classification, a small PyTorch FakeData demonstration, and separate U-Net segmentation code.
- **Decision:** Add a parallel, configuration-driven three-class classification pipeline. Do not rewrite or repurpose the existing entry points until compatibility is explicitly validated.
- **Alternatives considered:** Convert `train_densenet.py` in place; replace the backend model contract immediately.
- **Consequences:** Some code is duplicated at the boundary, but existing demos remain usable and research artifacts are isolated.

## DEC-003 - Use validation macro F1 as the primary selection metric

- **Date:** 2026-07-16
- **Context:** Viral and bacterial classes are expected to be imbalanced and clinically meaningful; existing binary artifacts already show accuracy masking low minority recall.
- **Decision:** Select architectures, preprocessing, and tuning on validation macro F1. Use balanced accuracy, minimum per-class recall, validation loss/calibration, and efficiency as ordered tie-breakers.
- **Alternatives considered:** Accuracy; binary/one-vs-rest AUC alone; validation loss alone.
- **Consequences:** Model selection will not reward majority-class performance at the expense of viral or bacterial recall.

## DEC-004 - Treat legacy classifier results as historical diagnostics only

- **Date:** 2026-07-16
- **Context:** Existing DenseNet reports use an incomplete generated tree, a non-group-aware derived validation split, no manifest hash, no locked test set, and a roughly 90:10 validation imbalance.
- **Decision:** Preserve and audit the artifacts, but do not use their metrics to select a three-class architecture, preprocessing method, threshold, or hyperparameter.
- **Alternatives considered:** Use DenseNet201 as the new baseline because its legacy ROC AUC is 0.998; discard the artifacts entirely.
- **Consequences:** The new benchmark starts from controlled manifests while retaining useful historical evidence and optional initialization checkpoints.

## DEC-005 - Leave unverified diagram terms unresolved

- **Date:** 2026-07-16
- **Context:** Full legacy source, dependency, backend, and result searches produced no implementation or definition of FILM, SAS Tuning, or automated grading of chest X-ray images.
- **Decision:** Mark all three terms unresolved. Do not implement Feature-wise Linear Modulation or rename Optuna search as SAS Tuning without repository evidence or an explicit definition.
- **Alternatives considered:** Infer common meanings from the literature; create substitutes under the requested names.
- **Consequences:** Supported experiments can proceed honestly, while reports retain the unresolved status and exact search evidence.

## DEC-006 - Lock the three-class label order and evidence rules

- **Date:** 2026-07-16
- **Context:** The source has only `NORMAL` and `PNEUMONIA` folders, while pneumonia filenames consistently encode `_virus_` or `_bacteria_`.
- **Decision:** Use class order Normal = 0, Viral = 1, Bacterial = 2. Derive Normal from the canonical `NORMAL` folder and derive Viral/Bacterial only from the respective filename token inside the canonical `PNEUMONIA` folder.
- **Alternatives considered:** Keep a binary label; infer etiology from pixels; use alphabetical index order.
- **Consequences:** The task has an explicit stable output contract. Every retained image has inspectable filename evidence and no pixel-derived label.

## DEC-007 - Rebuild all splits with conservative grouping

- **Date:** 2026-07-16
- **Context:** The original validation folder contains only 16 images and no Viral example. In addition, 264 conservative filename-derived group identifiers cross the supplied folder splits, and no authoritative patient metadata is available.
- **Decision:** Do not preserve the supplied test folder as the locked test set. Reassign all retained canonical images at seed 42 to 70/15/15 train/validation/test splits using deterministic multi-label group balancing. Keep filename-derived groups and exact-hash groups intact, and lock the resulting test manifest before model development.
- **Alternatives considered:** Preserve the supplied test folder; random image-level splitting; treat every filename as unrelated.
- **Consequences:** Assigned split group and exact-hash leakage are both zero. The grouping is deliberately conservative and may combine unrelated images when source identifiers are reused, so cross-folder identifier matches are not described as proven patient leakage.

## DEC-008 - Exclude exact copies but retain review-only pHash candidates

- **Date:** 2026-07-16
- **Context:** SHA-256 identified a complete nested copy of the canonical tree and 32 redundant exact files within the canonical tree. Perceptual hashing also flagged 16 pairs with different cryptographic hashes.
- **Decision:** Exclude all nested physical copies and deterministically retain one canonical path from each exact canonical duplicate set. Do not automatically exclude or union perceptual candidates; flag them for review because close pHashes can arise from common chest anatomy.
- **Alternatives considered:** Train on all physical files; exclude every perceptual candidate; ignore duplicates entirely.
- **Consequences:** The modeling set contains 5,824 unique cryptographic images. Potential near duplicates remain visible in the audit without converting heuristic similarity into unsupported identity claims.

## DEC-009 - Isolate the PyTorch CUDA pipeline from legacy TensorFlow

- **Date:** 2026-07-16
- **Context:** The existing Python 3.10 environment provides working TensorFlow 2.10 GPU support but has dependency gaps and a protobuf conflict. The requested architectures and modern AMP/YOLO tooling require PyTorch/torchvision.
- **Decision:** Keep the legacy environment unchanged and create a git-ignored Python 3.12 `.venv` pinned by `model/requirements.multiclass-cu121.txt` with PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121.
- **Alternatives considered:** Modify the legacy environment in place; implement all new models in TensorFlow; use the system CPU-only PyTorch install.
- **Consequences:** Binary compatibility is protected and CUDA PyTorch is reproducible, at the cost of a separate 5.27 GB environment.

## DEC-010 - Use a parallel manifest-driven PyTorch training interface

- **Date:** 2026-07-16
- **Context:** Fair comparison requires identical manifests, logging, evaluation, checkpoints, and test-set controls across heterogeneous models.
- **Decision:** Implement a new `model/src/multiclass` package and shared trainer/evaluator. Leave all legacy TensorFlow source and FastAPI behavior unchanged until final model selection.
- **Alternatives considered:** Rewrite `train_densenet.py`; use separate unrelated training scripts for every architecture.
- **Consequences:** The eight supported model heads share one output/metrics/checkpoint contract, and legacy workflows remain available.

## DEC-011 - Support YOLOv5 and YOLOv8 classification; reject YOLOv10 detection substitution

- **Date:** 2026-07-16
- **Context:** Official pinned YOLOv5 source contains `yolov5s-cls`, and Ultralytics 8.3.168 contains `yolov8n-cls.yaml`. Its YOLOv10 files terminate in `v10Detect` and no YOLOv10 classification YAML/checkpoint is present.
- **Decision:** Support official YOLOv5s-cls and YOLOv8n-cls with pinned source/weight hashes. Mark YOLOv10 classification unsupported; never train its detector on an image-classification dataset without boxes.
- **Alternatives considered:** Force a detection YAML through the classification task flag; relabel a newer YOLO classifier as YOLOv10; omit all YOLO models.
- **Consequences:** Two requested YOLO classifiers are comparable through the common trainer. YOLOv10 appears in the matrix with an exact unsupported reason.

## DEC-012 - Keep best and last checkpoints for every run

- **Date:** 2026-07-16
- **Context:** Long work must survive crashes, and the best validation epoch may differ from the latest epoch.
- **Decision:** Atomically retain `best_checkpoint.pt` and `last_checkpoint.pt` for every run. Checkpoints include optimizer, scheduler, scaler, history, best metric, configuration hash, and RNG states. Do not retain redundant per-epoch checkpoints by default.
- **Alternatives considered:** Save only model weights; save every epoch; keep only the best checkpoint.
- **Consequences:** Runs resume and selected models remain recoverable while storage remains bounded to two checkpoints per run.

## DEC-013 - Use 224-pixel ImageNet-normalized baseline inputs

- **Date:** 2026-07-16
- **Context:** All supported pretrained classifiers are compatible with 224-pixel inputs, and the available RTX 2050 has 4 GB VRAM.
- **Decision:** Use centered 224 by 224 RGB inputs and ImageNet normalization for the controlled baseline. Use conservative horizontal flip and up to 5-degree rotation only in training. Model-specific deviations require a recorded experiment decision.
- **Alternatives considered:** Preserve arbitrary aspect ratios; use 512-pixel inputs; use aggressive generic augmentation.
- **Consequences:** Architectures receive a common, memory-safe baseline and compatible pretrained normalization, with acknowledged crop/resizing limitations.

## DEC-014 - Use balanced cross-entropy and measured architecture-specific batches for baselines

- **Date:** 2026-07-16
- **Context:** Bacterial images are about 47% of the retained set, validation macro F1 is primary, and the RTX 2050 has 4 GB VRAM. Candidate batch sizes were measured with real CUDA optimizer steps.
- **Decision:** Use balanced class-weighted cross-entropy, AdamW at 0.0001 with weight decay 0.0001, and architecture-specific batches of 16/12/8 for DenseNet121/169/201, 16/8/6 for ResNet50/101/152, and 32 for YOLOv5/YOLOv8. Keep all other baseline policy controls equal.
- **Alternatives considered:** Unweighted loss; focal loss for every baseline; a single batch size for all models; maximum-memory batches.
- **Consequences:** Initial comparisons address imbalance without adding focal-loss parameters and retain VRAM headroom. Focal loss and alternate weights remain tuning dimensions.

## DEC-015 - Do not initialize PyTorch baselines from legacy Keras HDF5 checkpoints

- **Date:** 2026-07-16
- **Context:** The legacy DenseNet checkpoints are TensorFlow/Keras HDF5 models with a one-unit sigmoid head. The new fair-comparison pipeline uses torchvision PyTorch modules and three-logit heads.
- **Decision:** Use the documented official ImageNet initialization for standard baselines. Do not claim the Keras HDF5 weights are directly compatible with PyTorch. A binary-checkpoint experiment would require a separately validated layer-by-layer conversion and is not part of the controlled baseline.
- **Alternatives considered:** Load the HDF5 file directly; hand-map weights without exhaustive equivalence tests; omit transfer learning.
- **Consequences:** Baseline provenance is consistent and technically valid. The requested optional binary initialization is deferred as incompatible with the current framework boundary rather than misrepresented.

## DEC-016 - Use bounded Optuna TPE studies for general tuning

- **Date:** 2026-07-16
- **Context:** The RTX 2050 baseline estimate is several hours, so an unconstrained search across three configurations would be impractical. SAS Tuning is undefined in the repository, while Optuna provides a local resumable study database and explicit trial states.
- **Decision:** Tune the top three validation-selected configurations in separate Optuna 4.9.0 TPE studies, four trials per configuration and six equal-budget epochs per trial. Search optimizer-specific learning rate, weight decay, optimizer, scheduler, cross-entropy versus focal loss, class weighting, label smoothing, and conservative versus disabled augmentation. Use validation macro F1 only and retain every trial state and normal run artifact.
- **Alternatives considered:** Call the search SAS Tuning; exhaustive grid search; an unbounded Bayesian study; tune only one configuration; use locked-test feedback.
- **Consequences:** The search is reproducible and computationally bounded but intentionally small, so it cannot establish a globally optimal hyperparameter set. Final robustness runs independently retrain the selected configuration for the full baseline epoch budget.

## DEC-017 - Independently retrain the selected configuration at three seeds

- **Date:** 2026-07-16
- **Context:** Six-epoch tuning trials are not directly interchangeable with the 12-epoch baseline policy, and reusing one as seed-42 robustness evidence would create unequal training budgets.
- **Decision:** After tuning selection, write a frozen 12-epoch configuration and independently train it at seeds 42, 123, and 2026. Report individual validation values plus mean and sample standard deviation.
- **Alternatives considered:** Reuse the best six-epoch tuning trial as seed 42; run only seeds 123 and 2026; select a different epoch budget per seed.
- **Consequences:** Robustness values share the same training policy at the cost of one additional full run. The run matrix contains an explicit `final_selected_s42` row instead of silently reusing a shorter trial.

## DEC-018 - Freeze a representative seed before the single locked-test evaluation

- **Date:** 2026-07-16
- **Context:** Choosing whichever robustness seed looks best and then testing it could turn random-seed variation into another model-selection dimension. Test access must remain a single, auditable terminal step.
- **Decision:** After all three robustness runs, record their validation distribution but use the predeclared seed-42 best checkpoint as the representative final checkpoint. Write `final_selection.json` before test access and require it, its exact configuration hash, checkpoint path, and an explicit unlock flag for one locked-test evaluation.
- **Alternatives considered:** Pick the best validation seed; ensemble three seeds; evaluate every seed on test; allow direct test evaluation with only a command-line flag.
- **Consequences:** The test result has a clear pre-test selection policy and cannot be used to choose among seeds. An ensemble remains future work and is not inferred after observing test outcomes.

## DEC-019 - Keep tuning eligibility within the three preregistered focused representatives

- **Date:** 2026-08-02
- **Context:** The focused preprocessing design deliberately selected DenseNet169 for baseline macro F1, DenseNet201 for Viral-class performance, and ResNet50 for an efficient independent residual family. Ranking every completed baseline before tuning could replace ResNet50 with a marginally higher one-seed DenseNet result and would abandon the preregistered performance-efficiency question.
- **Decision:** Resolve exactly one validation-best preprocessing parent for DenseNet169, DenseNet201, and ResNet50, then give each one the same four-trial, six-epoch Optuna budget. Rank the three resulting studies by the documented validation-only criteria.
- **Alternatives considered:** Tune the three highest one-seed validation scores regardless of research role; tune all eight architectures; tune only DenseNet169.
- **Consequences:** Tuning remains fair and bounded while retaining the macro-F1, Viral-class, and performance-efficiency representatives. It does not claim that un-tuned architectures are intrinsically inferior.

## DEC-020 - Retain FT-004 after the fixed FT-011 screening fails

- **Date:** 2026-10-13; terminal status recorded 15:04:15 Asia/Manila.
- **Context:** The final controlled experiment changed the initial DenseNet-201 backbone rate from 0.0001 to 0.00001 while keeping the classifier at 0.0001 and matching FT-010's fresh seed initialization, full-image VIRIDIS -> GE -> CLAHE pipeline, dataset assignments, loss, optimizer and runtime settings. The predeclared screening required all five gates to pass before seeds 123 and 2026 could run.
- **Evidence:** Seed 42 completed at epoch 9 with best epoch 5. Observed macro F1 0.8202782987500647 failed >= 0.8613107953930739; balanced accuracy 0.8173146418282663 failed >= 0.8507499938072187; Viral recall 0.5829596412556054 failed >= 0.6681614349775785; Normal recall 0.9873417721518988 passed >= 0.9831223628691983; Bacterial recall 0.8816425120772947 failed >= 0.9009661835748792. Exact thresholds and artifacts are recorded in `model/results/ft011/final_decision.json`.
- **Decision:** Retain the existing FT-004 frozen package and close the tuning phase. Do not run FT-011 seeds 123 or 2026, evaluate replacement gates, or retrospectively select a confirmation seed after the seed-42 failure.
- **Selected package:** `model/results/final_ft004_ensemble/frozen_selection.json`, SHA-256 `bbb65ee1202b2c34490cdb1047339795e5751f2c71edf6626d1b1f235acea57b`. Exactly three seed-42 DenseNet-201 members use VIRIDIS -> GE -> CLAHE, VIRIDIS -> GE, and GE. Average their probability vectors equally, then choose Normal/Viral/Bacterial argmax. Historical seeds 42, 123 and 2026 confirm this ensemble design; they do not create a nine-network deployment.
- **Protection rationale:** Preserve and disclose the initial RAM-paused smoke attempt and its unknown termination cause. Accept the later completed protected smoke as the execution check. Disclose the full-run required telemetry-file write failure at 14:16:39, complete state recovery, and resume at 14:18:45 after 122.547 healthy seconds. The interruption did not authorize changing batch size, precision, loss or learning-rate semantics. Actual overheating and thermal throttling were zero; peak GPU temperature was 51 C.
- **Consequences:** FT-011 supplies a completed negative validation result and only one candidate seed, so no three-seed mean or sample standard deviation is available. The 10,000 paired proxy-cluster resamples compare matched observed seed-42 systems; derived groups do not establish patient independence. The historical locked-test result of another configuration does not transfer to FT-004. The live website remains unchanged, and the next step is independent evaluation of the frozen FT-004 system.
