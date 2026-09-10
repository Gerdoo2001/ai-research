# Results Interpretation

All values below are regenerated from consolidated per-run artifacts. Validation evidence is kept separate from the locked test result. Planned, failed, interrupted, skipped, and unsupported work remains visible.

## Baseline Architecture Benchmark

| Rank | Run | Architecture | Preprocessing | Macro F1 | Balanced accuracy | Normal recall | Viral recall | Bacterial recall | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline_densenet169_s42 | densenet169 | baseline | 0.840293 | 0.837324 | 0.987342 | 0.618834 | 0.905797 | completed |
| 2 | baseline_densenet201_s42 | densenet201 | baseline | 0.838713 | 0.840279 | 0.983122 | 0.699552 | 0.838164 | completed |
| 3 | baseline_densenet121_s42 | densenet121 | baseline | 0.837745 | 0.827333 | 0.970464 | 0.569507 | 0.942029 | completed |
| 4 | baseline_resnet101_s42 | resnet101 | baseline | 0.837300 | 0.835853 | 0.991561 | 0.636771 | 0.879227 | completed |
| 5 | baseline_resnet50_s42 | resnet50 | baseline | 0.836919 | 0.834447 | 0.987342 | 0.636771 | 0.879227 | completed |
| 6 | baseline_resnet152_s42 | resnet152 | baseline | 0.833275 | 0.832120 | 0.983122 | 0.641256 | 0.871981 | completed |
| 7 | baseline_yolov5_cls_s42 | yolov5_cls | baseline | 0.794705 | 0.793459 | 0.978903 | 0.524664 | 0.876812 | completed |
| 8 | baseline_yolov8_cls_s42 | yolov8_cls | baseline | 0.773396 | 0.774431 | 0.970464 | 0.497758 | 0.855072 | completed |

The current validation leader is `baseline_densenet169_s42` at macro F1 0.840293 and balanced accuracy 0.837324. Its lowest class recall is Viral at 0.618834. This ranking is final only when all eight supported baselines have completed.

## Preprocessing Ablation

| Run | Architecture | Preprocessing | Macro F1 | Balanced accuracy | Normal recall | Viral recall | Bacterial recall | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_densenet201_histogram_equalization_s42 | densenet201 | histogram_equalization | 0.844714 | 0.842814 | 0.983122 | 0.663677 | 0.881643 | completed |
| ablation_densenet201_clahe_s42 | densenet201 | clahe | 0.841733 | 0.843874 | 0.987342 | 0.677130 | 0.867150 | completed |
| ablation_resnet50_pseudo_color_s42 | resnet50 | pseudo_color | 0.838547 | 0.829110 | 0.978903 | 0.578475 | 0.929952 | completed |
| ablation_densenet169_pseudo_color_s42 | densenet169 | pseudo_color | 0.837293 | 0.834623 | 0.978903 | 0.645740 | 0.879227 | completed |
| ablation_densenet201_pseudo_color_s42 | densenet201 | pseudo_color | 0.837062 | 0.829137 | 0.983122 | 0.569507 | 0.934783 | completed |
| ablation_densenet169_clahe_s42 | densenet169 | clahe | 0.833828 | 0.833842 | 0.983122 | 0.668161 | 0.850242 | completed |
| ablation_densenet169_histogram_equalization_s42 | densenet169 | histogram_equalization | 0.829458 | 0.828844 | 0.974684 | 0.659193 | 0.852657 | completed |
| ablation_resnet50_clahe_s42 | resnet50 | clahe | 0.823620 | 0.821028 | 0.974684 | 0.618834 | 0.869565 | completed |
| ablation_resnet50_histogram_equalization_s42 | resnet50 | histogram_equalization | 0.819553 | 0.820232 | 0.940928 | 0.713004 | 0.806763 | completed |

Preprocessing is selected by validation macro F1 with balanced accuracy and minimum class recall as tie-breakers. Pseudo-coloring is not interpreted as adding clinical information.

## General Optuna Tuning

| Study | Trial | Run | Status | Macro F1 | Balanced accuracy | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| tuning_top1 | 0 | tuning_top1_trial_000 | completed | 0.844714 | 0.842814 | - |
| tuning_top1 | 1 | tuning_top1_trial_001 | completed | 0.812990 | 0.807174 | - |
| tuning_top1 | 2 | tuning_top1_trial_002 | completed | 0.799890 | 0.799757 | - |
| tuning_top1 | 3 | tuning_top1_trial_003 | completed | 0.831671 | 0.831888 | - |
| tuning_top2 | 0 | tuning_top2_trial_000 | completed | 0.840293 | 0.837324 | - |
| tuning_top2 | 1 | tuning_top2_trial_001 | completed | 0.808950 | 0.801939 | - |
| tuning_top2 | 2 | tuning_top2_trial_002 | completed | 0.803215 | 0.797832 | - |
| tuning_top2 | 3 | tuning_top2_trial_003 | completed | 0.830025 | 0.827204 | - |
| tuning_top3 | 0 | tuning_top3_trial_000 | completed | 0.838547 | 0.829110 | - |
| tuning_top3 | 1 | tuning_top3_trial_001 | completed | 0.786046 | 0.780075 | - |
| tuning_top3 | 2 | tuning_top3_trial_002 | completed | 0.780904 | 0.772353 | - |
| tuning_top3 | 3 | tuning_top3_trial_003 | completed | 0.819929 | 0.818293 | - |

The bounded TPE search is general validation-only tuning and is not called SAS Tuning. The small trial budget limits optimization claims.

## Multi-Seed Robustness

| Run | Architecture | Preprocessing | Macro F1 | Balanced accuracy | Normal recall | Viral recall | Bacterial recall | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_selected_s42 | densenet201 | histogram_equalization | 0.844714 | 0.842814 | 0.983122 | 0.663677 | 0.881643 | completed |
| final_selected_s123 | densenet201 | histogram_equalization | 0.836618 | 0.832527 | 0.974684 | 0.641256 | 0.881643 | completed |
| final_selected_s2026 | densenet201 | histogram_equalization | 0.835511 | 0.832527 | 0.974684 | 0.641256 | 0.881643 | completed |

Across seeds 42, 123, and 2026, validation macro F1 mean is 0.838948 with sample standard deviation 0.005024.

## Frozen Locked-Test Evaluation

| Run | Architecture | Preprocessing | Macro F1 | Balanced accuracy | Normal recall | Viral recall | Bacterial recall | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_locked_test | densenet201 | histogram_equalization | 0.788772 | 0.785812 | 0.970464 | 0.556054 | 0.830918 | completed |

The one frozen locked-test result is macro F1 0.788772 and balanced accuracy 0.785812. Despite the consolidated column names, this row's source `metrics.json` is explicitly labeled `locked_test`; it was not used for selection.

## Failed, Unsupported, and Unresolved Work

| Run | Stage | Status | Reason |
| --- | --- | --- | --- |
| baseline_yolov10_cls_s42 | baseline | unsupported | Ultralytics 8.3.168 provides v10Detect YAMLs but no YOLOv10 classification YAML/checkpoint |
| ablation_pca_unsupported | preprocessing_ablation | unsupported | raw-image PCA is not scientifically justified; feature-space experiment deferred |
| ablation_film_unsupported | preprocessing_ablation | unsupported | FILM remains unresolved in repository evidence |
| focused_top1_bestprep_s42 | focused_comparison | skipped | reuse completed validation evidence ablation_densenet201_histogram_equalization_s42; redundant retraining prohibited |
| focused_top2_bestprep_s42 | focused_comparison | skipped | reuse completed validation evidence baseline_densenet169_s42; redundant retraining prohibited |
| focused_top3_bestprep_s42 | focused_comparison | skipped | reuse completed validation evidence ablation_resnet50_pseudo_color_s42; redundant retraining prohibited |

| Run | Stage | Status | Reason |
| --- | --- | --- | --- |
| smoke_yolov8_cls_e2e | smoke | failed | 'tuple' object has no attribute 'shape' |

YOLOv10 classification is unsupported because a genuine classification head/checkpoint is absent. Raw-image PCA is deferred, and FILM, SAS Tuning, and automated grading remain unresolved. No missing result is estimated or fabricated.

## Interpretation Limits

Performance comes from one public data source with filename-derived etiology labels and no authoritative patient metadata. Viral/Bacterial confusion, label noise, source/device shift, calibration, subgroup behavior, and external generalization require further study. This is research evidence only and not clinical deployment validation.
