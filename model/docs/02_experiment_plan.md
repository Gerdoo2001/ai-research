# Controlled Experiment Plan

## Selection Boundary

All experiments use locked manifest `three_class_seed42_3b815cf769c8` and class order Normal, Viral, Bacterial. Training fits model parameters and any learned preprocessing. Validation controls architecture, preprocessing, thresholds, tuning, and early stopping. `test_manifest.csv` remains inaccessible unless the final evaluation command explicitly provides `--allow-locked-test` after all choices are frozen.

Primary selection metric is validation macro F1. Tie-breakers, in order, are balanced accuracy, minimum per-class recall with emphasis on Viral and Bacterial, validation loss/calibration, then smaller/faster model when performance is effectively tied. Accuracy alone cannot select a model.

Smoke and pilot metrics are pipeline/resource diagnostics, not selection evidence. In particular, the one-epoch DenseNet121 pilot used the full training and validation manifests only to measure end-to-end duration and prove artifact generation.

## Controlled Baseline Policy

The eight supported architecture baselines share:

- 224 by 224 centered RGB inputs and ImageNet normalization;
- official ImageNet/classification initialization recorded by exact source;
- maximum 12 epochs at seed 42;
- AdamW, learning rate 0.0001, weight decay 0.0001;
- balanced cross-entropy class weights;
- ReduceLROnPlateau on validation macro F1, factor 0.5, patience 2;
- early stopping patience 4 with minimum improvement 0.0001;
- horizontal flip probability 0.5 and rotation up to 5 degrees in training only;
- CUDA AMP, deterministic seeds, `num_workers=0`, best/last checkpoints;
- identical train and validation manifests and metric implementation.

Only batch size varies to stay comfortably within 4 GB VRAM. No locked-test image or metric is used.

## Measured Resource Envelope

All candidate batches passed eight measured CUDA training steps after three warmups. Peak values below cover the measured optimizer/AMP steps and are not total system VRAM. Synthetic throughput excludes decoding, preprocessing, disk I/O, metrics, and checkpointing.

| Architecture | Batch | Peak allocated bytes | Synthetic train images/s | Estimated 12-epoch end-to-end duration | Estimated run storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| DenseNet121 | 16 | 1,205,400,064 | 101.76 | 0:51:08 | 174,749,116 bytes |
| DenseNet169 | 12 | 1,172,914,688 | 71.12 | 0:54:57 | 309,743,190 bytes |
| DenseNet201 | 8 | 1,112,614,912 | 47.53 | 1:01:15 | 446,608,060 bytes |
| ResNet50 | 16 | 1,094,583,296 | 95.20 | 0:51:45 | 578,745,966 bytes |
| ResNet101 | 8 | 1,181,030,912 | 47.81 | 1:01:09 | 1,042,153,890 bytes |
| ResNet152 | 6 | 1,424,621,568 | 31.96 | 1:10:30 | 1,423,858,902 bytes |
| YOLOv5s-cls | 32 | 131,449,856 | 1,233.84 | 0:43:00 | 106,902,280 bytes |
| YOLOv8n-cls | 32 | 296,165,888 | 795.47 | 0:43:24 | 40,187,996 bytes |

The full-manifest DenseNet121 pilot took 253.996 seconds for its training/validation epoch versus 44.350 seconds of synthetic compute estimate, an observed non-compute/integration overhead of 209.646 seconds per epoch. Applying that measured overhead to all eight baselines yields an estimated 26,227 seconds, or about 7.29 hours, before early stopping. Estimated baseline artifact storage is 4,122,949,400 bytes. Approximately 97.2 GB remained free after the pilot, so the baseline matrix has substantial disk headroom.

These are scheduling estimates, not promised durations. Per-model actual time, peak memory, and file size will be recorded from completed runs.

## Stage A - Smoke and Recovery Tests

Completed evidence includes:

- CPU configuration, manifest lock, transform, metric, eight-head output, and YOLOv10 rejection tests;
- CUDA forward/backward steps for all eight supported model heads;
- real-data end-to-end DenseNet, ResNet, YOLOv5, YOLOv8, CLAHE, histogram equalization, and pseudo-color smokes;
- pretrained initialization smokes for DenseNet, ResNet, YOLOv5, and YOLOv8;
- controlled interruption after epoch 1 and successful resume at epoch 2;
- strict DenseNet/YOLO checkpoint inference with normalized class probabilities.

One YOLOv8 integration failure is preserved and fixed in retry 2. Superseded smoke artifacts are not selection evidence.

## Stage B - Baseline Architecture Benchmark

Planned seed-42 runs are:

- `baseline_densenet121_s42`
- `baseline_densenet169_s42`
- `baseline_densenet201_s42`
- `baseline_resnet50_s42`
- `baseline_resnet101_s42`
- `baseline_resnet152_s42`
- `baseline_yolov5_cls_s42`
- `baseline_yolov8_cls_s42`

`baseline_yolov10_cls_s42` remains in the matrix with status `unsupported`: Ultralytics 8.3.168 has YOLOv10 detection YAMLs ending in `v10Detect`, but no genuine YOLOv10 classification YAML/checkpoint. Object detection will not be substituted.

The benchmark runner executes independent rows sequentially, skips already completed configurations, preserves failure directories, updates the ledger/run matrix, and continues after safe independent failures.

## Stage C - Preprocessing Ablation

The completed controls are reused rather than retrained. The representative models are DenseNet169 (best validation macro F1), DenseNet201 (best Viral recall), and ResNet50 (efficient independent residual family). Each is compared under the same four conditions: baseline/no enhancement, CLAHE, global histogram equalization, and exploratory VIRIDIS. Only the nine new enhanced runs are launched:

- `ablation_densenet169_{clahe,histogram_equalization,pseudo_color}_s42`
- `ablation_densenet201_{clahe,histogram_equalization,pseudo_color}_s42`
- `ablation_resnet50_{clahe,histogram_equalization,pseudo_color}_s42`

CLAHE, global equalization, and VIRIDIS pseudo-color are never blindly combined. Visual examples and fixed parameters are saved under `model/results/preprocessing_examples`.

PCA is listed as unsupported/deferred for this stage because raw high-resolution pixel PCA is not scientifically defensible for convolutional transfer learning. A later train-only feature-space PCA experiment may be designed if a clear representation and question justify it. FILM is unsupported/unresolved because no verified meaning or valid conditioning signal exists in the repository.

## Stage D - Focused Generalization Check

After Stage B/C, rank the four completed preprocessing conditions within each representative and record the best completed configuration. The placeholder rows `focused_top1_bestprep_s42` through `focused_top3_bestprep_s42` become selection aliases with status `skipped`; no redundant retraining is allowed because the full representative cross-product already exists.

## Stage E - General Hyperparameter Tuning

Tune the validation-best preprocessing parent for each of the three preregistered focused representatives—DenseNet169, DenseNet201, and ResNet50—with Optuna TPE on validation data only. Each four-trial, six-epoch study begins with the exact parent settings as a preregistered control, followed by bounded trials over optimizer, learning rate, weight decay, scheduler, class weighting, corrected focal versus cross-entropy loss, label smoothing, and conservative versus no augmentation. Input size, architecture-safe batch size, pretrained weights, and preprocessing remain fixed. Dropout and freeze/unfreeze are not mixed into this small common search because they are not exposed consistently across all heads. Every completed, pruned, failed, and interrupted trial must be retained.

This is general Optuna tuning. It is not called SAS Tuning because the repository contains no SAS definition, dependency, or implementation. “Automated grading of chest X-ray images tuning” also remains unresolved.

## Stage F - Multi-Seed Robustness

Retrain the final validation-selected configuration with seeds 42, 123, and 2026. Reuse an exactly matching seed-42 configuration if already completed. Report individual results plus mean and standard deviation. If hardware/time prevents all three, document the incomplete rows rather than estimating metrics.

## Stage G - Locked Test Evaluation

Only after architecture, preprocessing, hyperparameters, selection metric, and checkpoint policy are frozen will `final_locked_test` run with explicit test unlock. Test results are reported separately from validation and may not trigger retraining or configuration changes.

## Matrix and Controlled Statuses

`model/results/run_matrix.csv` contains 49 current rows covering completed diagnostics, the pilot, eight supported baselines, one unsupported YOLO baseline, nine preprocessing ablations, two explicitly unsupported preprocessing ideas, focused selection aliases, tuning, robustness, and final test. Status values are restricted to planned, queued, running, completed, failed, interrupted, unsupported, or skipped.

The machine-readable resource evidence is in `training_resource_estimate.json` and `run_matrix_summary.json`.

## Launch and Resume Commands

From the repository root:

```text
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage baseline
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage baseline --resume
```

Before launch, rerun `nvidia-smi`, `pip check`, unit tests, source immutability verification, and disk-space checks. The locked test must remain unopened.
