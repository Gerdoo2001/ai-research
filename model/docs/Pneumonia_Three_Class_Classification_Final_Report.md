# Pneumonia Three-Class Classification Final Report

**Document version:** 1.0
**Generated:** 2026-08-03T13:54:55+08:00
**Research-use status:** Final report from the completed locked-manifest experiment workflow.

## Executive Summary

This work preserved the legacy binary classifier and built a parallel reproducible classifier for Normal, Viral Pneumonia, and Bacterial Pneumonia. The final validation-selected configuration is `final_selected_s42` using densenet201 with histogram_equalization preprocessing. Across robustness seeds 42, 123, and 2026, validation macro F1 was 0.8389 +/- 0.0050 (sample standard deviation). The single frozen locked-test evaluation produced macro F1 0.7888, balanced accuracy 0.7858, and accuracy 0.7986.

The system remains a research classifier. It has not undergone external clinical validation, prospective evaluation, bias assessment across institutions or populations, calibration review for clinical decisions, regulatory review, or medical-device quality assurance.

## Research Background and Objective

The repository originally implemented binary Normal-versus-Pneumonia transfer learning. The objective was to audit that behavior first, derive defensible three-class labels from the actual source, prevent duplicate/group leakage, compare requested supported architectures fairly, evaluate conservative preprocessing choices, tune selected configurations without test feedback, measure seed variability, and evaluate one frozen model once on the locked test split.

## Class Definition and Diagram Interpretation

The locked output order is Normal = 0, Viral = 1, Bacterial = 2. Normal is derived from canonical NORMAL folders; Viral and Bacterial are derived only from explicit `_virus_` and `_bacteria_` filename evidence inside canonical PNEUMONIA folders.

The supplied research concepts were interpreted against repository evidence. CLAHE, global histogram equalization, and pseudo-coloring were implementable as independent transforms. Raw high-resolution PCA was deferred because no justified train-only feature-space design was supplied. FILM, SAS Tuning, and automated chest-X-ray grading remained unresolved and were not invented. Optuna TPE is reported only as general tuning.

## Existing Binary-System Audit

The legacy TensorFlow/Keras workflow uses DenseNet121/169/201 with frozen ImageNet backbones and a one-unit sigmoid head. It trains from a generated binary folder tree, lacks locked manifests and resumable checkpoints, and is vulnerable to split/duplicate issues. Its source, HDF5 models, output reports, and FastAPI binary contract were preserved. The complete evidence is in `model/docs/00_existing_binary_pipeline_audit.md`.

## Repository and Implementation Overview

The new `model/src/multiclass` package supplies configuration validation, locked-manifest loading, transforms, model adapters, training, metrics, artifact generation, tracking, evaluation, and inference. Every long run writes atomic status/config data, immediate epoch history, best/last checkpoints, a readable log, validation predictions, plots, and a journal entry. Windows data workers are configurable and default to zero. CUDA is required unless CPU fallback is explicitly enabled.

## Development Process Timeline

The append-only development log records the Phase 0 environment snapshot, legacy audit, immutable dataset audit, implementation, smoke/recovery validation, preprocessing evidence, resource estimates, controlled matrix, each benchmark stage, tuning, robustness, final test, consolidation, and reporting. Source: `model/docs/DEVELOPMENT_LOG.md`. Epoch-level events and checkpoint changes are in `model/docs/TRAINING_JOURNAL.md`.

## Major Engineering and Research Decisions

Stable decisions in `model/docs/DECISION_LOG.md` cover label evidence, conservative grouping, exact duplicate exclusion, environment isolation, model support, class order, primary metric, balanced loss, measured batches, checkpoint retention, preprocessing scope, bounded Optuna search, three independent robustness seeds, and predeclared seed-42 test representation.

## Problems Encountered and Verified Fixes

The issue log preserves dependency, integration, rendering, and third-party model-adapter problems. The failure ledger contains 2 rows, of which 0 remain marked unresolved. Superseded smoke failures were retained instead of deleted. Source: `model/docs/ISSUES_AND_FIXES.md` and `model/results/failures.csv`.

## Dataset Source, Counts, and Label Derivation

The immutable source inventory contains 17,591 physical files. The audit identified 5,856 canonical image candidates, excluded 32 redundant exact canonical copies, and retained 5,824 unique cryptographic images. The locked split version is `three_class_seed42_3b815cf769c8` with hash `3b815cf769c85658a04748da5eb2e15b981b89d9c9b77a404e90fe23473d2774`.

| Split | Normal | Viral | Bacterial | Total |
| --- | --- | --- | --- | --- |
| Train | 1105 | 1039 | 1932 | 4076 |
| Validation | 237 | 223 | 414 | 874 |
| Test | 237 | 223 | 414 | 874 |

## Duplicate, Leakage, and Split Analysis

The physical source contains a complete nested copy tree and 32 redundant exact files inside the canonical tree. Exact hashes and conservative filename-derived groups were kept within one assigned split. Assigned exact-hash leakage and assigned group leakage are both zero. Sixteen pHash candidate pairs were retained for review rather than automatically treated as identity. No authoritative patient metadata was available, so the grouping is conservative and is not claimed as proven patient identity.

## Architecture Coverage

DenseNet121, DenseNet169, and DenseNet201 use dense feature reuse; ResNet50, ResNet101, and ResNet152 use residual blocks. Torchvision ImageNet weights initialize all six. Official pinned YOLOv5s-cls and Ultralytics YOLOv8n-cls use genuine classification heads adapted to the common three-logit interface. YOLOv10 classification is unsupported because Ultralytics 8.3.168 provides YOLOv10 detection heads but no genuine YOLOv10 classification YAML/checkpoint; no detector was mislabeled as a classifier.

## Preprocessing Methods

Baseline uses safe RGB conversion, centered square fit, tensor conversion, and ImageNet normalization. CLAHE operates on grayscale luminance with clip limit 2.0 and 8x8 tiles. Global histogram equalization is distinct from CLAHE and uses grayscale equalization. Pseudo-coloring applies OpenCV VIRIDIS to grayscale and does not add clinical information. Training-only augmentation is conservative horizontal flip and rotation. No learned preprocessing is fitted on validation or test data.

## Training Configuration

The final configuration uses `densenet201`, `histogram_equalization` preprocessing, 224x224 input, batch 8, `adamw`, learning rate 0.0001, weight decay 0.0001, `reduce_on_plateau` scheduling, `cross_entropy` loss, class weighting `balanced`, AMP `True`, and at most 12 epochs with early-stopping patience 4.

## Hyperparameter-Tuning Methodology

Three validation-selected parent configurations received separate Optuna 4.9.0 TPE studies. Four equal-budget six-epoch trials per study searched optimizer-specific learning rate, weight decay, optimizer, scheduler, cross-entropy versus focal loss, class weighting, label smoothing, and conservative versus disabled augmentation. 12 trials completed; failed/pruned states remain in the table. The small search budget is a hardware-driven limitation.

## Hardware and Software Environment

Training used the audited NVIDIA GeForce RTX 2050 with 4,096 MiB VRAM. The isolated environment uses Python 3.12.6, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, cuDNN 90100, Ultralytics 8.3.168, and Optuna 4.9.0. Exact machine-readable details are in `model/results/environment.json` and the pinned requirements file.

## Complete Experiment Matrix

| Run | Stage | Architecture | Preprocessing | Seed | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| pilot_densenet121_full_epoch | pilot | densenet121 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_continuation_d169_clahe | smoke | densenet169 | clahe | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_continuation_d201_histogram | smoke | densenet201 | histogram_equalization | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_continuation_r50_viridis | smoke | resnet50 | pseudo_color | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_densenet121_e2e | smoke | densenet121 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_preprocess_clahe | smoke | densenet121 | clahe | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_preprocess_histogram | smoke | densenet121 | histogram_equalization | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_preprocess_pseudocolor | smoke | densenet121 | pseudo_color | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_pretrained_densenet121 | smoke | densenet121 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_pretrained_resnet50 | smoke | resnet50 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_pretrained_yolov5 | smoke | yolov5_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_pretrained_yolov8 | smoke | yolov8_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_resnet50_e2e | smoke | resnet50 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_resume_contract | smoke | densenet121 | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_yolov5_cls_e2e | smoke | yolov5_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_yolov5_cls_e2e_retry1 | smoke | yolov5_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_yolov8_cls_e2e | smoke | yolov8_cls | baseline | 42 | failed | pipeline diagnostic only; not model-selection evidence |
| smoke_yolov8_cls_e2e_retry1 | smoke | yolov8_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| smoke_yolov8_cls_e2e_retry2 | smoke | yolov8_cls | baseline | 42 | completed | pipeline diagnostic only; not model-selection evidence |
| baseline_densenet121_s42 | baseline | densenet121 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_densenet169_s42 | baseline | densenet169 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_densenet201_s42 | baseline | densenet201 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_resnet50_s42 | baseline | resnet50 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_resnet101_s42 | baseline | resnet101 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_resnet152_s42 | baseline | resnet152 | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_yolov5_cls_s42 | baseline | yolov5_cls | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_yolov8_cls_s42 | baseline | yolov8_cls | baseline | 42 | completed | controlled architecture benchmark; validation selection only |
| baseline_yolov10_cls_s42 | baseline | yolov10_cls | baseline | 42 | unsupported | Ultralytics 8.3.168 provides v10Detect YAMLs but no YOLOv10 classification YAML/checkpoint |
| ablation_densenet169_clahe_s42 | preprocessing_ablation | densenet169 | clahe | 42 | completed | independent transform comparison against baseline_densenet169_s42 |
| ablation_densenet169_histogram_equalization_s42 | preprocessing_ablation | densenet169 | histogram_equalization | 42 | completed | independent transform comparison against baseline_densenet169_s42 |
| ablation_densenet169_pseudo_color_s42 | preprocessing_ablation | densenet169 | pseudo_color | 42 | completed | independent transform comparison against baseline_densenet169_s42 |
| ablation_densenet201_clahe_s42 | preprocessing_ablation | densenet201 | clahe | 42 | completed | independent transform comparison against baseline_densenet201_s42 |
| ablation_densenet201_histogram_equalization_s42 | preprocessing_ablation | densenet201 | histogram_equalization | 42 | completed | independent transform comparison against baseline_densenet201_s42 |
| ablation_densenet201_pseudo_color_s42 | preprocessing_ablation | densenet201 | pseudo_color | 42 | completed | independent transform comparison against baseline_densenet201_s42 |
| ablation_resnet50_clahe_s42 | preprocessing_ablation | resnet50 | clahe | 42 | completed | independent transform comparison against baseline_resnet50_s42 |
| ablation_resnet50_histogram_equalization_s42 | preprocessing_ablation | resnet50 | histogram_equalization | 42 | completed | independent transform comparison against baseline_resnet50_s42 |
| ablation_resnet50_pseudo_color_s42 | preprocessing_ablation | resnet50 | pseudo_color | 42 | completed | independent transform comparison against baseline_resnet50_s42 |
| ablation_pca_unsupported | preprocessing_ablation | not_applicable | pca | 42 | unsupported | raw-image PCA is not scientifically justified; feature-space experiment deferred |
| ablation_film_unsupported | preprocessing_ablation | not_applicable | film | 42 | unsupported | FILM remains unresolved in repository evidence |
| focused_top1_bestprep_s42 | focused_comparison | densenet201 | histogram_equalization | 42 | skipped | reuse completed validation evidence ablation_densenet201_histogram_equalization_s42; redundant retraining prohibited |
| tuning_top1 | tuning | densenet201 | histogram_equalization | 42 | completed | parent_run_id=ablation_densenet201_histogram_equalization_s42; validation rank 1 after best-per-architecture filtering |
| focused_top2_bestprep_s42 | focused_comparison | densenet169 | baseline | 42 | skipped | reuse completed validation evidence baseline_densenet169_s42; redundant retraining prohibited |
| tuning_top2 | tuning | densenet169 | baseline | 42 | completed | parent_run_id=baseline_densenet169_s42; validation rank 2 after best-per-architecture filtering |
| focused_top3_bestprep_s42 | focused_comparison | resnet50 | pseudo_color | 42 | skipped | reuse completed validation evidence ablation_resnet50_pseudo_color_s42; redundant retraining prohibited |
| tuning_top3 | tuning | resnet50 | pseudo_color | 42 | completed | parent_run_id=ablation_resnet50_pseudo_color_s42; validation rank 3 after best-per-architecture filtering |
| final_selected_s42 | robustness | densenet201 | histogram_equalization | 42 | completed | independent robustness retrain; no test access |
| final_selected_s123 | robustness | densenet201 | histogram_equalization | 123 | completed | independent robustness retrain; no test access |
| final_selected_s2026 | robustness | densenet201 | histogram_equalization | 2026 | completed | independent robustness retrain; no test access |
| final_locked_test | locked_test | densenet201 | histogram_equalization | 42 | completed | selection frozen from validation evidence; representative seed policy fixed before test |

## Training-Progress Summary

The final matrix contains 49 rows spanning diagnostics, eight supported baselines, one unsupported YOLOv10 row, preprocessing ablation, focused comparison, three tuning studies, three independent robustness seeds, and one locked-test evaluation. All epoch histories and checkpoint changes remain available under `model/results/runs` and in the training journal.

## Main Validation Results and Ranked Model Comparison

| Rank | Run | Architecture | Macro F1 | Balanced acc. | Normal recall | Viral recall | Bacterial recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline_densenet169_s42 | densenet169 | 0.8402925204293371 | 0.837324318106221 | 0.9873417721518988 | 0.6188340807174888 | 0.9057971014492754 |
| 2 | baseline_densenet201_s42 | densenet201 | 0.8387127845905328 | 0.8402793945278848 | 0.9831223628691983 | 0.6995515695067265 | 0.8381642512077294 |
| 3 | baseline_densenet121_s42 | densenet121 | 0.8377454223641405 | 0.8273332823285808 | 0.9704641350210971 | 0.5695067264573991 | 0.9420289855072463 |
| 4 | baseline_resnet101_s42 | resnet101 | 0.8373000267945933 | 0.8358531783410421 | 0.9915611814345991 | 0.6367713004484304 | 0.8792270531400966 |
| 5 | baseline_resnet50_s42 | resnet50 | 0.8369186220186831 | 0.8344467085801419 | 0.9873417721518988 | 0.6367713004484304 | 0.8792270531400966 |
| 6 | baseline_resnet152_s42 | resnet152 | 0.8332751176034384 | 0.8321195481929555 | 0.9831223628691983 | 0.6412556053811659 | 0.8719806763285024 |
| 7 | baseline_yolov5_cls_s42 | yolov5_cls | 0.7947052576865478 | 0.7934594083064804 | 0.9789029535864979 | 0.5246636771300448 | 0.8768115942028986 |
| 8 | baseline_yolov8_cls_s42 | yolov8_cls | 0.7733963518313524 | 0.774431482107615 | 0.9704641350210971 | 0.4977578475336323 | 0.855072463768116 |

Macro F1 is primary. Balanced accuracy, minimum per-class recall, validation loss/calibration, and then efficiency break close ties; accuracy alone was never the selection rule.

## Preprocessing Ablation

| Run | Architecture | Macro F1 | Balanced acc. | Normal recall | Viral recall | Bacterial recall |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_densenet169_s42 | densenet169 | 0.8402925204293371 | 0.837324318106221 | 0.9873417721518988 | 0.6188340807174888 | 0.9057971014492754 |
| baseline_densenet201_s42 | densenet201 | 0.8387127845905328 | 0.8402793945278848 | 0.9831223628691983 | 0.6995515695067265 | 0.8381642512077294 |
| baseline_resnet50_s42 | resnet50 | 0.8369186220186831 | 0.8344467085801419 | 0.9873417721518988 | 0.6367713004484304 | 0.8792270531400966 |
| ablation_densenet169_clahe_s42 | densenet169 | 0.8338281396717196 | 0.8338417812468322 | 0.9831223628691983 | 0.6681614349775785 | 0.8502415458937198 |
| ablation_densenet169_histogram_equalization_s42 | densenet169 | 0.8294581225869765 | 0.8288444580822744 | 0.9746835443037974 | 0.6591928251121076 | 0.8526570048309179 |
| ablation_densenet169_pseudo_color_s42 | densenet169 | 0.8372928110485005 | 0.8346233056801653 | 0.9789029535864979 | 0.6457399103139013 | 0.8792270531400966 |
| ablation_densenet201_clahe_s42 | densenet201 | 0.8417333136508015 | 0.843873858483018 | 0.9873417721518988 | 0.6771300448430493 | 0.8671497584541062 |
| ablation_densenet201_histogram_equalization_s42 | densenet201 | 0.844714031735541 | 0.8428140016637786 | 0.9831223628691983 | 0.6636771300448431 | 0.8816425120772947 |
| ablation_densenet201_pseudo_color_s42 | densenet201 | 0.8370620673857595 | 0.8291372326740832 | 0.9831223628691983 | 0.5695067264573991 | 0.9347826086956522 |
| ablation_resnet50_clahe_s42 | resnet50 | 0.8236199768902815 | 0.8210276141375301 | 0.9746835443037974 | 0.6188340807174888 | 0.8695652173913043 |
| ablation_resnet50_histogram_equalization_s42 | resnet50 | 0.8195533224688325 | 0.8202320131237605 | 0.9409282700421941 | 0.7130044843049327 | 0.8067632850241546 |
| ablation_resnet50_pseudo_color_s42 | resnet50 | 0.8385466291362813 | 0.8291099935768748 | 0.9789029535864979 | 0.57847533632287 | 0.9299516908212561 |
| ablation_pca_unsupported | not_applicable | - | - | - | - | - |
| ablation_film_unsupported | not_applicable | - | - | - | - | - |

## Tuning Results

| Study | Trial | Run | Status | Macro F1 | Balanced acc. | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| tuning_top1 | 0 | tuning_top1_trial_000 | completed | 0.844714031735541 | 0.8428140016637786 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0001} |
| tuning_top1 | 1 | tuning_top1_trial_001 | completed | 0.812990025423075 | 0.8071742480478115 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.0, "learning_rate_sgd": 0.001205712628744377, "loss": "focal", "optimizer": "sgd", "scheduler": "cosine", "weight_decay": 2.9375384576328313e-06} |
| tuning_top1 | 2 | tuning_top1_trial_002 | completed | 0.799889695991391 | 0.7997572155500144 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.05, "learning_rate_sgd": 0.0002814509271606064, "loss": "focal", "optimizer": "sgd", "scheduler": "reduce_on_plateau", "weight_decay": 7.4763120622522945e-06} |
| tuning_top1 | 3 | tuning_top1_trial_003 | completed | 0.8316711403341853 | 0.8318884728985152 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001086355964087569, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0006264278041381329} |
| tuning_top2 | 0 | tuning_top2_trial_000 | completed | 0.8402925204293371 | 0.837324318106221 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0001} |
| tuning_top2 | 1 | tuning_top2_trial_001 | completed | 0.8089495099273433 | 0.8019392683304271 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.0, "learning_rate_sgd": 0.001205712628744377, "loss": "focal", "optimizer": "sgd", "scheduler": "cosine", "weight_decay": 2.9375384576328313e-06} |
| tuning_top2 | 2 | tuning_top2_trial_002 | completed | 0.8032148685899752 | 0.7978317861434713 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.05, "learning_rate_sgd": 0.0002814509271606064, "loss": "focal", "optimizer": "sgd", "scheduler": "reduce_on_plateau", "weight_decay": 7.4763120622522945e-06} |
| tuning_top2 | 3 | tuning_top2_trial_003 | completed | 0.8300251765872434 | 0.8272039423200234 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001086355964087569, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0006264278041381329} |
| tuning_top3 | 0 | tuning_top3_trial_000 | completed | 0.8385466291362813 | 0.8291099935768748 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0001} |
| tuning_top3 | 1 | tuning_top3_trial_001 | completed | 0.7860463965151318 | 0.7800747740640285 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.0, "learning_rate_sgd": 0.001205712628744377, "loss": "focal", "optimizer": "sgd", "scheduler": "cosine", "weight_decay": 2.9375384576328313e-06} |
| tuning_top3 | 2 | tuning_top3_trial_002 | completed | 0.7809039091522334 | 0.7723528099276821 | {"augmentation": "none", "class_weighting": "none", "label_smoothing": 0.05, "learning_rate_sgd": 0.0002814509271606064, "loss": "focal", "optimizer": "sgd", "scheduler": "reduce_on_plateau", "weight_decay": 7.4763120622522945e-06} |
| tuning_top3 | 3 | tuning_top3_trial_003 | completed | 0.8199287825636011 | 0.81829278135588 | {"augmentation": "conservative", "class_weighting": "balanced", "label_smoothing": 0.0, "learning_rate_adamw": 0.0001086355964087569, "loss": "cross_entropy", "optimizer": "adamw", "scheduler": "reduce_on_plateau", "weight_decay": 0.0006264278041381329} |

## Multi-Seed Robustness

| Run | Architecture | Macro F1 | Balanced acc. | Normal recall | Viral recall | Bacterial recall |
| --- | --- | --- | --- | --- | --- | --- |
| final_selected_s42 | densenet201 | 0.844714031735541 | 0.8428140016637786 | 0.9831223628691983 | 0.6636771300448431 | 0.8816425120772947 |
| final_selected_s123 | densenet201 | 0.8366182095700467 | 0.8325272205874193 | 0.9746835443037974 | 0.6412556053811659 | 0.8816425120772947 |
| final_selected_s2026 | densenet201 | 0.8355114189876733 | 0.8325272205874193 | 0.9746835443037974 | 0.6412556053811659 | 0.8816425120772947 |

Validation macro F1 mean is 0.8389; sample standard deviation is 0.0050.

## Locked Test Results

The single frozen test evaluation used `final_selected_s42` by the predeclared seed-42 policy. Macro F1 is 0.7888, balanced accuracy is 0.7858, accuracy is 0.7986, macro ROC-AUC is 0.8891, macro PR-AUC is 0.7987, log loss is 0.6307, ECE is 0.1004, and multiclass Brier score is 0.3131.

| Class | Precision | Recall/sensitivity | Specificity | F1 | Support | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Normal | 0.9746 | 0.9705 | 0.9906 | 0.9725 | 237 | 0.9981 | 0.9959 |
| Viral | 0.6200 | 0.5561 | 0.8833 | 0.5863 | 223 | 0.7892 | 0.5878 |
| Bacterial | 0.7854 | 0.8309 | 0.7957 | 0.8075 | 414 | 0.8799 | 0.8124 |

## Efficiency and Model-Size Trade-offs

| Run | Params | File bytes | Peak GPU bytes | Latency ms/image | Throughput img/s |
| --- | --- | --- | --- | --- | --- |
| baseline_densenet169_s42 | 12489475 | 151361720 | 1174747136 | 26.54583499988803 | 37.67069297327502 |
| baseline_densenet201_s42 | 18098691 | 219114744 | 1116384256 | 33.043815000019094 | 30.262849492391304 |
| baseline_densenet121_s42 | 6956931 | 84434520 | 1205961728 | 30.449769999904674 | 32.84097055587384 |
| baseline_resnet101_s42 | 42506307 | 511004682 | 1350572032 | 11.72324999988632 | 85.3005779122425 |
| baseline_resnet50_s42 | 23514179 | 282649771 | 1097041408 | 6.860245000098075 | 145.76738877193216 |
| baseline_resnet152_s42 | 58149955 | 699152172 | 1811506688 | 26.818460000140476 | 37.28774881163057 |
| baseline_yolov5_cls_s42 | 4176323 | 16873976 | 148133376 | 5.828599999949802 | 171.56778643389706 |
| baseline_yolov8_cls_s42 | 1442131 | 5904948 | 87646208 | 4.897005000020727 | 204.20644863457716 |

Latency values use batch one on CUDA after warmup and exclude preprocessing and host-to-device transfer. They are controlled diagnostics, not end-to-end clinical deployment claims.

## Error Analysis

The final test set contains 874 images and 176 incorrect predictions. Confusion directions are:

| Direction | Count |
| --- | --- |
| Viral -> Bacterial | 94 |
| Bacterial -> Viral | 69 |
| Normal -> Viral | 7 |
| Viral -> Normal | 5 |
| Bacterial -> Normal | 1 |

The highest-confidence incorrect predictions and lowest-confidence predictions remain in `model/results/runs/final_locked_test/sample_errors.csv` and `predictions.csv` with de-identified sample IDs and relative source paths. Viral-versus-Bacterial errors are interpreted cautiously because filename-derived etiology and single-source label quality may be imperfect.

## Final Selected Model and Justification

The final configuration is densenet201 with histogram_equalization preprocessing. It was selected before test access from validation macro F1 with declared tie-breakers, then independently retrained across three seeds. The seed-42 checkpoint was predeclared as representative rather than selected after viewing test outcomes. Final checkpoint: `C:\Users\JOHN MARK\Code\ai-research-main\model\results\runs\final_selected_s42\best_checkpoint.pt`.

## Limitations

The dataset is a single public source with filename-derived Viral/Bacterial labels and no authoritative patient metadata. Conservative grouping can merge unrelated reused identifiers, while unrecognized patient relations may remain. Class balance is imperfect, source/device diversity is limited, label noise is plausible, the tuning budget is small, and no external dataset is evaluated. Confidence calibration requires domain-specific review. The versioned research-complete Word deliverable is subject to a separate rendered-page visual QA record.

## Clinical-Safety Disclaimer

This classifier is for research only and is not validated, approved, or suitable for clinical diagnosis, triage, treatment, or deployment. Clinical use would require external and prospective validation, broader representative datasets, subgroup bias analysis, calibration and threshold review, human-factors testing, medical oversight, security/privacy review, and applicable regulatory and quality-system processes.

## Recommended Future Work

Obtain authoritative patient identifiers and radiologist-reviewed etiology labels; evaluate external institutions and acquisition devices; study label noise; expand tuning under a preregistered compute budget; examine feature-space PCA only with train-only fitting; define FILM or grading concepts before implementation; evaluate ensembling prospectively; conduct subgroup and calibration analyses; and integrate the API only after its three-class contract and clinical-use restrictions are reviewed.

## Reproduction Commands

Exact Windows commands for environment setup, audits, manifests, tests, training, resume, matrix execution, tuning, frozen test evaluation, consolidation, and report generation are in `model/docs/06_reproduction_guide.md`. The selected seed-42 training command is:

```text
.venv\Scripts\python.exe model\scripts\train_multiclass.py --config model\configs\final_selected.json --seed 42 --run-id final_selected_s42
```

## Output-File Inventory

Primary outputs are the locked manifests, environment/support audits, run matrix and ledgers, per-run directories, consolidated CSV/JSON files, continuous Markdown documentation, progressive Word report, this final Markdown report, and the final Word report. Paths are rooted under `model/data_manifests`, `model/results`, and `model/docs`.

## Appendix A - Final Configuration

```json
{
  "allow_cpu_fallback": false,
  "amp": true,
  "architecture": "densenet201",
  "augmentation": {
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": 5.0
  },
  "batch_size": 8,
  "class_names": [
    "Normal",
    "Viral",
    "Bacterial"
  ],
  "class_weighting": "balanced",
  "controlled_stop_after_epoch": null,
  "device": "cuda",
  "documentation_interval": 1,
  "early_stopping_patience": 4,
  "epochs": 12,
  "freeze_backbone": false,
  "input_size": 224,
  "label_smoothing": 0.0,
  "learning_rate": 0.0001,
  "loss": "cross_entropy",
  "manifest_dir": "model/data_manifests",
  "max_train_samples": null,
  "max_val_samples": null,
  "minimum_delta": 0.0001,
  "num_workers": 0,
  "optimizer": "adamw",
  "preprocessing": "histogram_equalization",
  "results_dir": "model/results/runs",
  "run_id": "final_selected_s42",
  "scheduler": "reduce_on_plateau",
  "scheduler_factor": 0.5,
  "scheduler_patience": 2,
  "schema_version": 1,
  "seed": 42,
  "smoke_test": false,
  "task_mode": "multiclass",
  "tensorboard": true,
  "weight_decay": 0.0001,
  "weights": "imagenet"
}
```

## Appendix B - Metric Definitions

Macro F1 is the unweighted mean of class F1 values. Balanced accuracy is mean class recall. Specificity is one-vs-rest true-negative rate. ROC-AUC and PR-AUC are one-vs-rest. ECE uses confidence bins; the multiclass Brier score sums squared probability error across classes. Reported standard deviation across seeds is the sample standard deviation.

## Appendix C - Failed, Unsupported, Skipped, and Unresolved Work

| Run | Stage | Architecture | Preprocessing | Seed | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_yolov10_cls_s42 | baseline | yolov10_cls | baseline | 42 | unsupported | Ultralytics 8.3.168 provides v10Detect YAMLs but no YOLOv10 classification YAML/checkpoint |
| ablation_pca_unsupported | preprocessing_ablation | not_applicable | pca | 42 | unsupported | raw-image PCA is not scientifically justified; feature-space experiment deferred |
| ablation_film_unsupported | preprocessing_ablation | not_applicable | film | 42 | unsupported | FILM remains unresolved in repository evidence |
| focused_top1_bestprep_s42 | focused_comparison | densenet201 | histogram_equalization | 42 | skipped | reuse completed validation evidence ablation_densenet201_histogram_equalization_s42; redundant retraining prohibited |
| focused_top2_bestprep_s42 | focused_comparison | densenet169 | baseline | 42 | skipped | reuse completed validation evidence baseline_densenet169_s42; redundant retraining prohibited |
| focused_top3_bestprep_s42 | focused_comparison | resnet50 | pseudo_color | 42 | skipped | reuse completed validation evidence ablation_resnet50_pseudo_color_s42; redundant retraining prohibited |

The failure ledger contains 2 preserved entries. FILM, SAS Tuning, and automated grading remain unresolved. Raw-image PCA and YOLOv10 classification remain unsupported for the documented technical reasons; no result is fabricated for them.

## Appendix D - Source Artifacts

All numeric claims derive from `model/results/summary.csv`, `model/results/model_ranking.csv`, `model/results/preprocessing_ablation.csv`, `model/results/tuning_trials.csv`, `model/results/seed_robustness.csv`, `model/results/test_results.csv`, the final per-run metrics/predictions, and the locked manifest summary. Consolidation verified those values against original per-run metrics with zero consistency errors.
