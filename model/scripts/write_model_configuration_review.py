"""Write the model-specific configuration and transferability review."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
RESULTS = MODEL / "results"
RUNS = RESULTS / "runs"
DOCS = MODEL / "docs"
AUDITS = RESULTS / "audits"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    with (RESULTS / "model_ranking.csv").open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))

    records = []
    for row in ranking:
        config = load_json(RUNS / row["run_id"] / "config.json")
        metrics = load_json(RUNS / row["run_id"] / "metrics.json")
        metadata = load_json(RUNS / row["run_id"] / "model_metadata.json")
        architecture = row["architecture"]
        family = "DenseNet" if architecture.startswith("densenet") else "ResNet" if architecture.startswith("resnet") else "YOLO classification"
        literature = {
            "densenet121": "Kundu et al. (binary ensemble), Arulananth et al. (binary pediatric), and CheXNet provide Partial evidence.",
            "densenet169": "DenseNet architecture provenance plus exact-task transfer evidence from Rahman et al.; no located exact-task paper uniquely establishes D169 settings.",
            "densenet201": "Rahman et al. directly evaluated DenseNet201 on the exact Normal/Viral/Bacterial task.",
            "resnet50": "Bakır et al. supports ResNet feature extraction on the exact task; Jaiswal et al. uses ResNet50 for localization only.",
            "resnet101": "Jaiswal et al. uses ResNet101 for CXR localization; the classification settings remain repository-controlled.",
            "resnet152": "ResNet architecture provenance; no supplied exact-task source uniquely establishes ResNet152 settings.",
            "yolov5_cls": "Official v7.0 classification checkpoint/head provenance; no supplied pneumonia paper establishes these exact settings.",
            "yolov8_cls": "Official v8.3.0 classification checkpoint/head provenance; no formal YOLOv8 paper or supplied pneumonia paper establishes these exact settings.",
        }[architecture]
        records.append(
            {
                "rank": int(row["rank"]),
                "run_id": row["run_id"],
                "family": family,
                "architecture": architecture,
                "pretrained_weight_source": metadata["weight_source"],
                "input_size": config["input_size"],
                "batch_size": config["batch_size"],
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "maximum_epochs": config["epochs"],
                "completed_epochs": int(load_json(RUNS / row["run_id"] / "status.json")["current_epoch"]),
                "best_epoch": metrics["best_epoch"],
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "recall_normal": metrics["per_class"]["Normal"]["recall"],
                "recall_viral": metrics["per_class"]["Viral"]["recall"],
                "recall_bacterial": metrics["per_class"]["Bacterial"]["recall"],
                "expected_calibration_error": metrics["expected_calibration_error"],
                "parameter_count": metrics["parameter_count"],
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
                "latency_ms_per_image_compute_only": metrics["inference"]["latency_ms_per_image"],
                "training_duration_seconds": metrics["training_duration_seconds"],
                "literature_transferability": literature,
                "artifact_verification": "passed",
            }
        )

    selected = ["baseline_densenet169_s42", "baseline_densenet201_s42", "baseline_resnet50_s42"]
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds"),
        "selection_scope": "validation-only configuration review; locked test remains sealed",
        "common_control": {
            "classes": ["Normal", "Viral", "Bacterial"],
            "input_size": 224,
            "weights": "ImageNet/official classification weights",
            "full_network_fine_tuning": True,
            "preprocessing": "baseline RGB plus shared resize/crop/tensor/ImageNet normalization",
            "augmentation": {"horizontal_flip_probability": 0.5, "rotation_degrees": 5.0},
            "loss": "balanced class-weighted cross entropy",
            "class_weights": {"Normal": 1.2295625942684767, "Viral": 1.3076676291305742, "Bacterial": 0.7032436162870945},
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "scheduler": "ReduceLROnPlateau on validation macro F1, factor 0.5, patience 2",
            "early_stopping": "patience 4, minimum delta 0.0001",
            "maximum_epochs": 12,
            "selection_metric": "validation macro F1",
            "seed": 42,
            "amp": True,
        },
        "batch_size_policy": {
            "method": "architecture-specific safe values chosen by CUDA training-step resource calibration on the 4 GB RTX 2050",
            "values": {record["architecture"]: record["batch_size"] for record in records},
            "fairness_constraint": "Every preprocessing comparison must reuse the completed control's batch size for that architecture.",
            "limitation": "Batch size was not numerically identical across architecture baselines and may affect optimization noise; cross-architecture differences are therefore interpreted with that caveat.",
        },
        "representative_preprocessing_models": selected,
        "representative_selection_reason": {
            "baseline_densenet169_s42": "Highest verified validation macro F1 and best DenseNet169 control.",
            "baseline_densenet201_s42": "Second-highest macro F1 and highest viral recall (0.699552), testing a deeper DenseNet response.",
            "baseline_resnet50_s42": "Strong ResNet macro F1 with lowest ResNet training time, memory, and compute-only latency; adds a distinct residual family.",
        },
        "model_records": records,
        "configuration_decisions": [
            {
                "setting": "Input size 224",
                "decision": "Retain for controlled preprocessing ablations.",
                "reason": "Matches all completed controls and standard pretrained recipes; changing it would confound preprocessing effects and increase 4 GB GPU cost.",
                "risk": "ImageOps.fit can center-crop edge anatomy in extreme aspect ratios; audit visually and consider a future letterbox-versus-crop ablation only after the current comparison.",
            },
            {
                "setting": "ImageNet normalization",
                "decision": "Retain for all eight architectures.",
                "reason": "Matches pretrained-weight expectations and preserves architecture fairness.",
                "risk": "CLAHE/HE/VIRIDIS distributions differ from natural RGB; this is exactly why validation ablation is required.",
            },
            {
                "setting": "Horizontal flip p=0.5",
                "decision": "Retain for controlled ablations.",
                "reason": "Mild laterality perturbation already used in all controls.",
                "risk": "Laterality markers move; do not add vertical flips because they are anatomically implausible.",
            },
            {
                "setting": "Rotation +/-5 degrees",
                "decision": "Retain for controlled ablations.",
                "reason": "Models small acquisition/positioning variation while avoiding the much larger 30-degree range used in some binary studies.",
                "risk": "Black corner fill is a synthetic cue; Grad-CAM/error audit should check corner attention.",
            },
            {
                "setting": "Balanced class-weighted cross entropy",
                "decision": "Retain for ablations; include unweighted CE and focal loss only as bounded tuning candidates after preprocessing selection.",
                "reason": "Training counts are imbalanced and viral recall is the weakest class across baselines.",
                "risk": "Validation loss is currently unweighted; focal implementation must be corrected/audited before a trial.",
            },
            {
                "setting": "AdamW lr=1e-4, weight decay=1e-4",
                "decision": "Retain as the control optimizer for preprocessing ablation.",
                "reason": "This is the actual full-fine-tuning control shared by all baselines; it is an engineering control, not a value copied from a paper.",
                "risk": "May not be optimal for every architecture; tune only after a preprocessing winner is selected.",
            },
            {
                "setting": "Maximum 12 epochs with early stopping",
                "decision": "Retain for like-for-like preprocessing comparisons.",
                "reason": "Best epochs ranged from 5 to 11 and early stopping already bounded wasted computation.",
                "risk": "A new transform may converge differently; inspect learning curves and only extend a run by an explicit preregistered rule.",
            },
            {
                "setting": "Native Ultralytics classification augmentation",
                "decision": "Do not enable for YOLO-only runs in the controlled matrix.",
                "reason": "RandomResizedCrop, RandAugment, and random erasing would make the YOLO data distribution incomparable with DenseNet/ResNet.",
                "risk": "YOLO baselines may be below their native-recipe potential; that is a separate model-specific optimization question.",
            },
            {
                "setting": "VIRIDIS pseudo-color",
                "decision": "Keep one exploratory ablation, label it exploratory, and never claim clinical information is added.",
                "reason": "No direct VIRIDIS evidence was verified; the experiment can determine whether the fixed representation helps validation macro F1.",
                "risk": "It creates artificial color statistics and may conflict with ImageNet features or encourage colormap shortcuts.",
            },
            {
                "setting": "YOLOv10 classification",
                "decision": "Exclude.",
                "reason": "No genuine official compatible classification head/checkpoint was verified.",
                "risk": "Relabeling a detector or generic fork head would be scientifically misleading.",
            },
        ],
    }
    AUDITS.mkdir(parents=True, exist_ok=True)
    (AUDITS / "model_configuration_review_continuation_v2.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    table = "\n".join(
        f"| {record['rank']} | {record['architecture']} | {record['batch_size']} | {record['best_epoch']} | {record['validation_macro_f1']:.6f} | {record['recall_normal']:.3f} | {record['recall_viral']:.3f} | {record['recall_bacterial']:.3f} | {record['parameter_count']:,} | {record['latency_ms_per_image_compute_only']:.2f} |"
        for record in records
    )
    decision_rows = "\n".join(
        f"| {item['setting']} | {item['decision']} | {item['reason']} | {item['risk']} |"
        for item in payload["configuration_decisions"]
    )
    md = f"""# Model-Specific Configuration Review

Generated: {payload['generated_at']}

This review uses only the completed validation controls, verified implementation provenance, and the research evidence matrix. It does not access the locked test set.

## What the completed training shows

| Rank | Architecture | Batch | Best epoch | Validation macro F1 | Recall N | Recall V | Recall B | Parameters | Compute-only latency ms/image |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

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
{decision_rows}

## Meaning of the core training values

The initial learning rate **0.0001** is the starting step size used by AdamW. It is small because every layer is being fine-tuned from pretrained weights: large steps could destroy useful pretrained features. ReduceLROnPlateau halves it after two validation plateaus.

Weight decay **0.0001** gently penalizes large weights during optimization. It is regularization, not a second learning rate. Its scale is intentionally modest because the dataset is only 4,076 training images and the models already start pretrained.

Maximum **12 epochs** means at most twelve complete passes through the 4,076 training images. It is a ceiling, not a promise to use epoch 12. The best checkpoint is chosen by validation macro F1 and early stopping ends training after four epochs without a meaningful improvement. Observed best epochs were 5-11, confirming the ceiling was active but not uniformly binding.

## Loss and validation detail

The training weights computed from the locked manifest are Normal 1.229563, Viral 1.307668, and Bacterial 0.703244. They upweight the two smaller classes. The metrics and checkpoint selection are unweighted per-example calculations, with macro F1 giving each class equal importance.

One implementation distinction must remain explicit: training loss is weighted cross entropy, but saved validation loss is unweighted cross entropy. Macro F1—not loss—selects the checkpoint. This is acceptable for the completed baseline comparison, but future code should label the two losses clearly. The existing focal-loss implementation must be corrected or formally audited before it is allowed into tuning.

Machine-readable companion: `model/results/audits/model_configuration_review_continuation_v2.json`.
"""
    (DOCS / "08_model_configuration_review_continuation_v2.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
