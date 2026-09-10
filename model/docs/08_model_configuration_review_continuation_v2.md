# Model-Specific Configuration Review

Generated: 2026-08-02T12:41:15+08:00

This review uses only the completed validation controls, verified implementation provenance, and the research evidence matrix. It does not access the locked test set.

## What the completed training shows

| Rank | Architecture | Batch | Best epoch | Validation macro F1 | Recall N | Recall V | Recall B | Parameters | Compute-only latency ms/image |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | densenet169 | 12 | 6 | 0.840293 | 0.987 | 0.619 | 0.906 | 12,489,475 | 26.55 |
| 2 | densenet201 | 8 | 10 | 0.838713 | 0.983 | 0.700 | 0.838 | 18,098,691 | 33.04 |
| 3 | densenet121 | 16 | 10 | 0.837745 | 0.970 | 0.570 | 0.942 | 6,956,931 | 30.45 |
| 4 | resnet101 | 8 | 9 | 0.837300 | 0.992 | 0.637 | 0.879 | 42,506,307 | 11.72 |
| 5 | resnet50 | 16 | 5 | 0.836919 | 0.987 | 0.637 | 0.879 | 23,514,179 | 6.86 |
| 6 | resnet152 | 6 | 6 | 0.833275 | 0.983 | 0.641 | 0.872 | 58,149,955 | 26.82 |
| 7 | yolov5_cls | 32 | 11 | 0.794705 | 0.979 | 0.525 | 0.877 | 4,176,323 | 5.83 |
| 8 | yolov8_cls | 32 | 11 | 0.773396 | 0.970 | 0.498 | 0.855 | 1,442,131 | 4.90 |

The clearest recurring weakness is Viral recall, not Normal recall. DenseNet201 reached the highest Viral recall (0.700), while DenseNet169 achieved the best overall macro F1 (0.840293). The margin among the six DenseNet/ResNet baselines is small enough that a single seed cannot prove a universally superior architecture; robustness runs are required after configuration selection.

## Why the batch sizes differ

Batch size was selected by actual CUDA forward/backward resource calibration on the 4 GB RTX 2050, then fixed per architecture: D121 16, D169 12, D201 8, R50 16, R101 8, R152 6, YOLOv5-cls 32, and YOLOv8-cls 32. This avoided out-of-memory failures while preserving meaningful throughput.

The tradeoff is important: the architecture benchmark controlled data, weights type, transforms, loss, optimizer, learning rate, scheduler, maximum epochs, seed, and selection metric, but not numerical batch size. Therefore, architecture differences include a possible batch-noise effect. For preprocessing ablations, every new run must reuse its architecture's completed-control batch size so the within-model comparison is clean.

## Why the preprocessing representatives are DenseNet169, DenseNet201, and ResNet50

- **DenseNet169** is the current validation leader and has the strongest overall macro F1.
- **DenseNet201** is close in macro F1 but has the best Viral recall, so it tests whether enhancement affects the difficult class differently in a deeper DenseNet.
- **ResNet50** adds a different residual-learning family and is the most efficient ResNet control by time, memory, and measured compute-only latency.

DenseNet121 is not rerun as a control and is not the only ablation architecture. Its completed result remains valid, but D169/D201/R50 provide a more informative three-model screen within the available compute budget.

## Transferability by model family

- **DenseNet:** Rahman et al. directly evaluated DenseNet201 on the exact three-class task. Kundu, Arulananth, and CheXNet provide additional but partial evidence for DenseNet121. The original DenseNet paper supplies architecture provenance only.
- **ResNet:** Bakır et al. provides exact-task ResNet feature-extraction evidence. Jaiswal uses ResNet50/101 for localization, which supports backbone relevance but not classifier hyperparameters. The original ResNet paper supplies architecture provenance only.
- **YOLOv5/YOLOv8 classification:** both baselines use genuine official image-classification heads/checkpoints and a verified three-logit adapter. The project does not use their native trainers or detector losses. Their lower validation macro F1 is a measured result of this common controlled pipeline, not proof that native YOLO classification recipes are poor.
- **YOLOv10:** excluded because the verified official system is a detector and no compatible official classifier checkpoint/head was found.

## Configuration decisions before the next runs

| Setting | Decision | Why | Risk/interpretation |
|---|---|---|---|
| Input size 224 | Retain for controlled preprocessing ablations. | Matches all completed controls and standard pretrained recipes; changing it would confound preprocessing effects and increase 4 GB GPU cost. | ImageOps.fit can center-crop edge anatomy in extreme aspect ratios; audit visually and consider a future letterbox-versus-crop ablation only after the current comparison. |
| ImageNet normalization | Retain for all eight architectures. | Matches pretrained-weight expectations and preserves architecture fairness. | CLAHE/HE/VIRIDIS distributions differ from natural RGB; this is exactly why validation ablation is required. |
| Horizontal flip p=0.5 | Retain for controlled ablations. | Mild laterality perturbation already used in all controls. | Laterality markers move; do not add vertical flips because they are anatomically implausible. |
| Rotation +/-5 degrees | Retain for controlled ablations. | Models small acquisition/positioning variation while avoiding the much larger 30-degree range used in some binary studies. | Black corner fill is a synthetic cue; Grad-CAM/error audit should check corner attention. |
| Balanced class-weighted cross entropy | Retain for ablations; include unweighted CE and focal loss only as bounded tuning candidates after preprocessing selection. | Training counts are imbalanced and viral recall is the weakest class across baselines. | Validation loss is currently unweighted; focal implementation must be corrected/audited before a trial. |
| AdamW lr=1e-4, weight decay=1e-4 | Retain as the control optimizer for preprocessing ablation. | This is the actual full-fine-tuning control shared by all baselines; it is an engineering control, not a value copied from a paper. | May not be optimal for every architecture; tune only after a preprocessing winner is selected. |
| Maximum 12 epochs with early stopping | Retain for like-for-like preprocessing comparisons. | Best epochs ranged from 5 to 11 and early stopping already bounded wasted computation. | A new transform may converge differently; inspect learning curves and only extend a run by an explicit preregistered rule. |
| Native Ultralytics classification augmentation | Do not enable for YOLO-only runs in the controlled matrix. | RandomResizedCrop, RandAugment, and random erasing would make the YOLO data distribution incomparable with DenseNet/ResNet. | YOLO baselines may be below their native-recipe potential; that is a separate model-specific optimization question. |
| VIRIDIS pseudo-color | Keep one exploratory ablation, label it exploratory, and never claim clinical information is added. | No direct VIRIDIS evidence was verified; the experiment can determine whether the fixed representation helps validation macro F1. | It creates artificial color statistics and may conflict with ImageNet features or encourage colormap shortcuts. |
| YOLOv10 classification | Exclude. | No genuine official compatible classification head/checkpoint was verified. | Relabeling a detector or generic fork head would be scientifically misleading. |

## Meaning of the core training values

The initial learning rate **0.0001** is the starting step size used by AdamW. It is small because every layer is being fine-tuned from pretrained weights: large steps could destroy useful pretrained features. ReduceLROnPlateau halves it after two validation plateaus.

Weight decay **0.0001** gently penalizes large weights during optimization. It is regularization, not a second learning rate. Its scale is intentionally modest because the dataset is only 4,076 training images and the models already start pretrained.

Maximum **12 epochs** means at most twelve complete passes through the 4,076 training images. It is a ceiling, not a promise to use epoch 12. The best checkpoint is chosen by validation macro F1 and early stopping ends training after four epochs without a meaningful improvement. Observed best epochs were 5-11, confirming the ceiling was active but not uniformly binding.

## Loss and validation detail

The training weights computed from the locked manifest are Normal 1.229563, Viral 1.307668, and Bacterial 0.703244. They upweight the two smaller classes. The metrics and checkpoint selection are unweighted per-example calculations, with macro F1 giving each class equal importance.

One implementation distinction must remain explicit: training loss is weighted cross entropy, but saved validation loss is unweighted cross entropy. Macro F1—not loss—selects the checkpoint. This is acceptable for the completed baseline comparison, but future code should label the two losses clearly. The existing focal-loss implementation must be corrected or formally audited before it is allowed into tuning.

Machine-readable companion: `model/results/audits/model_configuration_review_continuation_v2.json`.
