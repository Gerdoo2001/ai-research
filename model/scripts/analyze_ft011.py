"""Pre-freeze FT-011 controls and analyze validation outcomes under fixed gates.

This module reads prediction tables and manifest metadata, never image files or
test predictions.  ``freeze_controls`` must run before candidate outcomes are
opened; its returned object belongs in the immutable experiment protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))
from multiclass.metrics import calculate_metrics
from finalize_ft004_ensemble import FT010_RUNS, FT004_MEMBERS

CLASSES = ("Normal", "Viral", "Bacterial")
SEEDS = (42, 123, 2026)
PROBABILITY_COLUMNS = tuple(f"probability_{name.lower()}" for name in CLASSES)
METRIC_NAMES = ("macro_f1", "accuracy", "balanced_accuracy", "normal_recall", "viral_recall", "bacterial_recall")
REFERENCES = (
    ("Yosinski et al. (2014), transferability of neural-network features", "https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html"),
    ("Long et al. (2015), transferable features", "https://proceedings.mlr.press/v37/long15.html"),
    ("Cawley and Talbot (2010), model-selection bias", "https://www.jmlr.org/papers/v11/cawley10a.html"),
    ("TRIPOD+AI (2024), transparent prediction-model reporting", "https://www.bmj.com/content/385/bmj-2023-078378"),
    ("Lakshminarayanan et al. (2017), deep ensembles", "https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html"),
    ("Riley et al. (2024), external validation", "https://www.bmj.com/content/384/bmj-2023-074820"),
    ("FT-011 protocol", "https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit"),
    ("Completed fine-tuning report", "https://docs.google.com/document/d/1mdLbxZMtv2icHKNs-VSjpZiWb5_2lG-S/edit"),
    ("Dataset preparation record", "https://docs.google.com/document/d/1I9-egb-SZRzbGQ-PXxM8UjvGmc_CCcsZ/edit"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required comparison artifact missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metric_scalars(metrics: dict[str, Any]) -> dict[str, float]:
    result = {name: float(metrics[name]) for name in METRIC_NAMES[:3]}
    for name in CLASSES:
        result[f"{name.lower()}_recall"] = float(metrics["per_class"][name]["recall"])
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result.values()):
        raise ValueError("invalid classification metric")
    return result


def check_metrics(saved: dict[str, Any], calculated: dict[str, Any], label: str) -> None:
    """Ensure saved decision metrics agree with the original evaluation code."""
    actual = metric_scalars(calculated)
    for name in METRIC_NAMES:
        if name in METRIC_NAMES[:3]:
            previous = saved.get(name)
        else:
            previous = saved.get("per_class", {}).get(name.removesuffix("_recall").title(), {}).get("recall")
        if previous is not None and not math.isclose(float(previous), actual[name], abs_tol=1e-12, rel_tol=0):
            raise RuntimeError(f"saved metric/prediction mismatch for {label}: {name}")
    if "confusion_matrix" in saved and saved["confusion_matrix"] != calculated["confusion_matrix"]:
        raise RuntimeError(f"saved confusion matrix/prediction mismatch for {label}")


def validation_manifest(repo_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    bundle = repo_root / "model/data_manifests"
    summary = read_json(bundle / "split_summary.json")
    if summary.get("class_order") != list(CLASSES):
        raise RuntimeError("locked manifest class order mismatch")
    # Deliberately avoid opening or hashing the held-out test manifest.
    for filename in ("train_manifest.csv", "val_manifest.csv"):
        if sha256(bundle / filename) != summary["manifest_hashes"][filename]:
            raise RuntimeError(f"locked manifest digest mismatch: {filename}")
    train = read_csv(bundle / "train_manifest.csv")
    rows = read_csv(bundle / "val_manifest.csv")
    if len(train) != 4076 or len(rows) != 874:
        raise RuntimeError("FT-011 requires exactly 4076 training and 874 validation assignments")
    if any(row["assigned_split"] != "train" for row in train) or any(row["assigned_split"] != "val" for row in rows):
        raise RuntimeError("assigned_split mismatch in locked training/validation manifests")
    if set(row["split_group_id"] for row in train) & set(row["split_group_id"] for row in rows):
        raise RuntimeError("training/validation proxy group overlap")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate relative_path in validation manifest")
    for row in rows:
        index = int(row["class_index"])
        if index not in range(3) or row["class_name"] != CLASSES[index] or not row["split_group_id"]:
            raise RuntimeError("validation label/group mismatch")
    return rows, summary


def aligned_predictions(path: Path, manifest: list[dict[str, str]]) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = read_csv(path)
    required = {"sample_id", "relative_path", "true_index", "true_class", *PROBABILITY_COLUMNS}
    if not rows or required - rows[0].keys():
        raise RuntimeError(f"incomplete validation prediction columns: {path}")
    by_path = {row["relative_path"]: row for row in rows}
    if len(rows) != len(manifest) or len(by_path) != len(rows) or len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate/missing validation prediction identity: {path}")
    aligned = []
    for index, assigned in enumerate(manifest):
        row = by_path.get(assigned["relative_path"])
        expected = (f"validation_{index:05d}", assigned["class_name"], int(assigned["class_index"]))
        if row is None or (row["sample_id"], row["true_class"], int(row["true_index"])) != expected:
            raise RuntimeError(f"sample_id/relative_path/label does not match assigned validation manifest: {path}")
        aligned.append(row)
    labels = np.asarray([int(row["true_index"]) for row in aligned], dtype=np.int64)
    probabilities = np.asarray([[float(row[name]) for name in PROBABILITY_COLUMNS] for row in aligned], dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError(f"invalid validation probabilities: {path}")
    # Historical mixed-precision CSVs have sub-per-mille rounding of row sums.
    if not np.allclose(probabilities.sum(axis=1), 1, atol=0.01, rtol=0):
        raise RuntimeError(f"validation probability sums invalid: {path}")
    return aligned, labels, probabilities


def aggregate_metrics(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    scalars = {seed: metric_scalars(metrics) for seed, metrics in per_seed.items()}
    return {
        name: {"mean": statistics.mean(item[name] for item in scalars.values()),
               "sample_std": statistics.stdev(item[name] for item in scalars.values()) if len(scalars) > 1 else None,
               "values_by_seed": {str(seed): item[name] for seed, item in scalars.items()}}
        for name in METRIC_NAMES
    }


def freeze_controls(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Return exact baseline metrics, hashes and gates before candidate inspection.

    This function performs no writes and never looks inside an FT-011 run.
    Missing metrics are recovered from saved probabilities via the original
    ``calculate_metrics`` function; any contradictory saved metric stops work.
    """
    repo = Path(repo_root or REPO_ROOT).resolve()
    manifest, summary = validation_manifest(repo)
    artifacts = [record_artifact(repo / "model/data_manifests" / name) for name in ("train_manifest.csv", "val_manifest.csv", "split_summary.json", "class_mapping.json")]
    systems: dict[str, Any] = {}
    ensemble_summary_path = repo / "model/results/ensemble_confirmation/summary.json"
    ensemble_summary = read_json(ensemble_summary_path)
    artifacts.append(record_artifact(ensemble_summary_path))
    for name in ("FT010", "FT004"):
        per_seed = {}
        prediction_artifacts = {}
        for seed in SEEDS:
            if name == "FT010":
                run = repo / "model/results/runs" / FT010_RUNS[seed]
                status = read_json(run / "status.json")
                config = read_json(run / "config.json")
                if status.get("status") != "completed" or config.get("seed") != seed:
                    raise RuntimeError(f"incomplete/mismatched FT-010 control seed {seed}")
                if config.get("class_names") != list(CLASSES) or config.get("task_mode") != "multiclass":
                    raise RuntimeError("FT-010 control is not original three-class experiment")
                path = run / "predictions.csv"
                saved_path = run / "metrics.json"
                saved = read_json(saved_path) if saved_path.exists() else {}
                if saved and saved.get("split") != "validation":
                    raise RuntimeError("FT-010 control metrics are not validation metrics")
                artifacts.extend(record_artifact(run / filename) for filename in ("config.json", "status.json", "history.csv", "best_checkpoint.pt"))
                if saved:
                    artifacts.append(record_artifact(saved_path))
            else:
                path = repo / "model/results/ensemble_confirmation" / f"ensemble_predictions_s{seed}.csv"
                saved = next(item["metrics"] for item in ensemble_summary["per_seed"] if item["seed"] == seed)
            _, labels, probabilities = aligned_predictions(path, manifest)
            computed = calculate_metrics(labels, probabilities, CLASSES)
            check_metrics(saved, computed, f"{name} seed {seed}")
            per_seed[seed] = computed
            entry = record_artifact(path)
            artifacts.append(entry)
            prediction_artifacts[str(seed)] = entry
        systems[name] = {"per_seed": {str(seed): value for seed, value in per_seed.items()}, "aggregate": aggregate_metrics(per_seed), "predictions": prediction_artifacts}
    incumbent_path = repo / "model/results/final_ft004_ensemble/frozen_selection.json"
    incumbent = read_json(incumbent_path)
    if incumbent.get("deployable_seed") != 42 or incumbent.get("class_order") != list(CLASSES) or len(incumbent.get("members", [])) != 3:
        raise RuntimeError("frozen incumbent package mismatch")
    artifacts.append(record_artifact(incumbent_path))
    artifacts.append(record_artifact(repo / "model/configs/final_ft004_ensemble.json"))
    for member, run_id in zip(incumbent["members"], FT004_MEMBERS[42]):
        checkpoint = repo / "model/results/runs" / run_id / "best_checkpoint.pt"
        entry = record_artifact(checkpoint)
        if member["run_id"] != run_id or member["checkpoint_sha256"] != entry["sha256"]:
            raise RuntimeError("frozen incumbent checkpoint changed")
        artifacts.append(entry)
    control42 = metric_scalars(systems["FT010"]["per_seed"]["42"])
    screening = [{"metric": name, "operator": ">=", "threshold": control42[name] + (0.005 if name == "macro_f1" else 0), "basis": "FT010 seed42"} for name in ("macro_f1", "balanced_accuracy", "viral_recall", "normal_recall", "bacterial_recall")]
    replacement = []
    for name in ("macro_f1", "balanced_accuracy", "viral_recall", "normal_recall", "bacterial_recall"):
        comparator = "FT010" if name in ("normal_recall", "bacterial_recall") else "FT004"
        replacement.append({"metric": name, "operator": ">" if name == "macro_f1" else ">=", "threshold": systems[comparator]["aggregate"][name]["mean"], "basis": f"{comparator} three-seed arithmetic mean"})
    return {
        "schema_version": 1, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation", "locked_test_accessed": False,
        "class_order": list(CLASSES), "seeds": list(SEEDS), "validation_rows": len(manifest),
        "manifest_version": summary["manifest_version"], "systems": systems,
        "screening_gates": screening, "replacement_gates": replacement,
        "artifact_hashes": artifacts, "frozen_incumbent": record_artifact(incumbent_path),
        "metric_recovery": "exact metrics recomputed using original multiclass.metrics.calculate_metrics and checked against available saved metrics",
        "bootstrap": {"iterations": 10000, "seed": 20261013, "unit": "class-stratified split_group_id proxy clusters; whole groups retained"},
    }


def evaluate_gates(values: dict[str, float], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for gate in gates:
        value = float(values[gate["metric"]])
        threshold = float(gate["threshold"])
        if gate["operator"] not in (">", ">=") or not math.isfinite(value):
            raise ValueError("invalid decision gate/value")
        passed = value > threshold if gate["operator"] == ">" else value >= threshold
        results.append({**gate, "observed": value, "difference": value - threshold, "passed": bool(passed)})
    return results


def cluster_strata(manifest: list[dict[str, str]]) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(manifest):
        if not row.get("split_group_id"):
            raise RuntimeError("missing proxy split_group_id prevents specified bootstrap")
        groups.setdefault(row["split_group_id"], []).append(index)
    strata: dict[tuple[int, ...], list[np.ndarray]] = {}
    for indices in groups.values():
        labels = tuple(sorted({int(manifest[index]["class_index"]) for index in indices}))
        strata.setdefault(labels, []).append(np.asarray(indices, dtype=np.int64))
    return list(strata.values()), {
        "unit": "derived split_group_id proxy cluster", "group_count": len(groups),
        "repeated_image_group_count": sum(len(indices) > 1 for indices in groups.values()),
        "strata": {",".join(CLASSES[i] for i in names): len(values) for names, values in strata.items()},
        "method": "sample the original number of groups with replacement within each class stratum; retain every image from each sampled group",
        "mixed_label_group_rule": "if present, stratify intact mixed-label groups by their class-membership tuple",
        "limitation": "Identifiers are derived filename/hash groups, not verified patient IDs. Proxy clustering does not establish patient independence. Resampled image counts may vary because group sizes vary.",
    }


def draw_cluster_indices(strata: list[list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([group for members in strata for group in (members[i] for i in rng.integers(0, len(members), size=len(members)))])


def fast_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Argmax decision metrics mathematically identical to original evaluator."""
    predictions = probabilities.argmax(axis=1)
    matrix = np.bincount(labels * 3 + predictions, minlength=9).reshape(3, 3)
    diagonal = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
    denominator = support + matrix.sum(axis=0)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(3), where=denominator != 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(diagonal.sum() / len(labels)), "balanced_accuracy": float(recall.mean()), "normal_recall": float(recall[0]), "viral_recall": float(recall[1]), "bacterial_recall": float(recall[2])}


def paired_bootstrap(labels: np.ndarray, systems: dict[str, dict[int, np.ndarray]], manifest: list[dict[str, str]], seeds: tuple[int, ...], iterations: int = 10000, random_seed: int = 20261013) -> dict[str, Any]:
    if set(systems) != {"FT011", "FT010", "FT004"} or not seeds:
        raise ValueError("bootstrap requires FT011 and both matched comparator systems")
    if any(set(runs) != set(seeds) for runs in systems.values()):
        raise ValueError("bootstrap seed alignment mismatch")
    if any(array.shape != (len(labels), 3) for runs in systems.values() for array in runs.values()):
        raise ValueError("bootstrap prediction shape mismatch")
    strata, unit = cluster_strata(manifest)
    points = {name: {metric: statistics.mean(fast_metrics(labels, probabilities)[metric] for probabilities in runs.values()) for metric in METRIC_NAMES} for name, runs in systems.items()}
    samples = {comparator: {metric: np.empty(iterations) for metric in METRIC_NAMES} for comparator in ("FT010", "FT004")}
    rng = np.random.default_rng(random_seed)
    for iteration in range(iterations):
        # The same draw is shared by every system, comparator and seed.
        indices = draw_cluster_indices(strata, rng)
        sampled = {name: [fast_metrics(labels[indices], runs[seed][indices]) for seed in seeds] for name, runs in systems.items()}
        averages = {name: {metric: statistics.mean(row[metric] for row in values) for metric in METRIC_NAMES} for name, values in sampled.items()}
        for comparator in samples:
            for metric in METRIC_NAMES:
                samples[comparator][metric][iteration] = averages["FT011"][metric] - averages[comparator][metric]
    comparisons = []
    for comparator, distributions in samples.items():
        output = {}
        for metric, values in distributions.items():
            lower, upper = np.quantile(values, [0.025, 0.975])
            output[metric] = {"point_difference": points["FT011"][metric] - points[comparator][metric], "bootstrap_mean_difference": float(values.mean()), "bootstrap_standard_error": float(values.std(ddof=1)), "ci_95_percentile": [float(lower), float(upper)], "bootstrap_probability_difference_above_zero": float(np.mean(values > 0))}
        comparisons.append({"candidate": "FT011", "comparator": comparator, "metrics": output})
    return {"iterations": iterations, "random_seed": random_seed, "seeds": list(seeds), "paired_unit": unit, "seed_handling": "calculate metric independently for each fixed seed, then arithmetic mean; no cross-seed probability averaging", "interpretation_boundary": "Validation sample uncertainty for the observed seeds. This is not independent external validation or proof of population-level random-seed performance.", "comparisons": comparisons}


def verify_frozen_controls(controls: dict[str, Any]) -> None:
    if controls.get("locked_test_accessed") is not False or controls.get("seeds") != list(SEEDS):
        raise RuntimeError("protocol lacks valid frozen controls")
    for artifact in controls["artifact_hashes"]:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError(f"frozen control artifact changed: {artifact['path']}")


def verify_candidate_configuration(run: Path, seed: int, protocol: dict[str, Any], metrics: dict[str, Any], history: list[dict[str, str]]) -> None:
    wrapper = read_json(run / "candidate_configuration.json")
    unhashed = dict(wrapper)
    expected_hash = unhashed.pop("candidate_configuration_hash")
    if canonical_hash(unhashed) != expected_hash or wrapper.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("candidate configuration/protocol hash mismatch")
    if wrapper.get("learning_rate_policy") != {"backbone": 1e-5, "classifier": 1e-4}:
        raise RuntimeError("candidate differential learning-rate policy mismatch")
    base = dict(protocol["control_configuration"])
    base.update(seed=seed, run_id=f"ft011_densenet201_differential_lr_s{seed}")
    if wrapper.get("base_training_configuration") != base or read_json(run / "config.json") != base:
        raise RuntimeError("candidate training configuration changed beyond intended seed/run identity")
    status = read_json(run / "status.json")
    if status.get("protocol_hash") != protocol["protocol_hash"] or metrics.get("protocol_hash") != protocol["protocol_hash"]:
        raise RuntimeError("completed candidate protocol provenance mismatch")
    if metrics.get("smoke_execution_only") is not False or status.get("smoke_execution_only") is not False:
        raise RuntimeError("smoke outcomes cannot enter candidate selection")
    initialization = read_json(run / "initialization.json")
    if initialization.get("fresh_imagenet_initialization") is not True or initialization.get("seed") != seed:
        raise RuntimeError("candidate did not record fresh seed-matched ImageNet initialization")
    if any(len(initialization.get(field, "")) != 64 for field in ("model_state_sha256", "classifier_state_sha256")):
        raise RuntimeError("candidate initialization hashes are missing")
    groups = read_json(run / "optimizer_groups.json")
    backbone, classifier = groups["backbone_parameter_names"], groups["classifier_parameter_names"]
    if groups.get("disjoint") is not True or groups.get("exhaustive") is not True or set(backbone) & set(classifier):
        raise RuntimeError("optimizer group audit is not disjoint and exhaustive")
    if len(backbone) != len(set(backbone)) or len(classifier) != len(set(classifier)):
        raise RuntimeError("duplicate optimizer parameter names")
    if not backbone or not classifier or any(name.startswith("classifier.") for name in backbone) or any(not name.startswith("classifier.") for name in classifier):
        raise RuntimeError("optimizer backbone/classifier group mismatch")
    if groups["initial_learning_rates"] != [1e-5, 1e-4] or groups["backbone_parameters"] + groups["classifier_parameters"] != metrics["trainable_parameter_count"]:
        raise RuntimeError("optimizer learning-rate/parameter coverage mismatch")
    scheduler = read_json(run / "scheduler_initial.json")
    for field, expected in protocol["scheduler_defaults"].items():
        recorded_field = "min_lrs" if field == "min_lr" else field
        if field != "monitored_metric" and scheduler.get(recorded_field) != expected:
            raise RuntimeError(f"candidate scheduler setting mismatch: {field}")
    best_score, best_epoch = -math.inf, 0
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)) or len(history) > base["epochs"]:
        raise RuntimeError("candidate history epoch sequence/budget mismatch")
    if [float(history[0][name]) for name in ("backbone_lr_used", "classifier_lr_used")] != [1e-5, 1e-4]:
        raise RuntimeError("candidate first-epoch learning rates differ from fixed policy")
    for row in history:
        score = float(row["val_macro_f1"])
        if score > best_score + base["minimum_delta"]:
            best_score, best_epoch = score, int(row["epoch"])
        if float(row["backbone_lr_used"]) <= 0 or float(row["classifier_lr_used"]) <= 0:
            raise RuntimeError("epoch history lacks valid both-group learning rates")
    if metrics["best_epoch"] != best_epoch or status["best_epoch"] != best_epoch or not math.isclose(metrics["macro_f1"], best_score, abs_tol=1e-12, rel_tol=0):
        raise RuntimeError("candidate selected checkpoint disagrees with fixed within-run selection rule")


def write_graphs(output: Path, runs: dict[int, dict[str, Any]], histories: dict[int, list[dict[str, str]]], controls: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for seed, history in histories.items():
        if not history:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [int(row["epoch"]) for row in history]
        for column, label in (("train_loss", "Training"), ("val_loss", "Validation")):
            axes[0].plot(epochs, [float(row[column]) for row in history], label=label)
        for column, label in (("val_macro_f1", "Macro F1"), ("val_balanced_accuracy", "Balanced accuracy"), ("val_accuracy", "Accuracy")):
            axes[1].plot(epochs, [float(row[column]) for row in history], label=label)
        for axis in axes:
            axis.set_xlabel("Epoch"); axis.legend(); axis.grid(alpha=0.25)
        axes[0].set_ylabel("Loss"); axes[1].set_ylabel("Validation metric [0, 1]")
        figure.suptitle(f"FT-011 seed {seed}: recorded training history")
        figure.tight_layout()
        path = output / f"seed_{seed}_training_curves.png"
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(str(path))
        if all("backbone_lr_used" in row and "classifier_lr_used" in row for row in history):
            figure, axis = plt.subplots(figsize=(7, 4))
            for column, label in (("backbone_lr_used", "Backbone"), ("classifier_lr_used", "Classifier")):
                axis.plot(epochs, [float(row[column]) for row in history], marker="o", label=label)
            axis.set_xlabel("Epoch"); axis.set_ylabel("Learning rate used"); axis.set_yscale("log"); axis.legend()
            figure.tight_layout(); path = output / f"seed_{seed}_learning_rates.png"
            figure.savefig(path, dpi=160); plt.close(figure); paths.append(str(path))
        matrix = np.asarray(runs[seed]["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Greys")
        axis.set_xticks(range(3), CLASSES); axis.set_yticks(range(3), CLASSES)
        axis.set_xlabel("Predicted class"); axis.set_ylabel("True class")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        figure.colorbar(image, ax=axis); figure.tight_layout()
        path = output / f"seed_{seed}_confusion_matrix.png"
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(str(path))
    figure, axis = plt.subplots(figsize=(9, 4))
    seeds = sorted(runs)
    offsets = np.arange(len(seeds))
    for group, name in enumerate(("FT010", "FT004", "FT011")):
        values = [runs[seed]["macro_f1"] if name == "FT011" else controls["systems"][name]["per_seed"][str(seed)]["macro_f1"] for seed in seeds]
        axis.bar(offsets + (group - 1) * 0.25, values, 0.25, label=name)
    axis.set_xticks(offsets, [f"Seed {seed}" for seed in seeds]); axis.set_ylabel("Validation macro F1 [0, 1]")
    axis.set_ylim(0, 1); axis.legend(); figure.tight_layout()
    path = output / "matched_seed_model_comparison.png"
    figure.savefig(path, dpi=160); plt.close(figure); paths.append(str(path))
    return paths


def gate_table(gates: list[dict[str, Any]]) -> str:
    lines = ["| Metric | Rule | Full-precision observed | Pass/fail |", "|---|---|---:|---|"]
    for gate in gates:
        lines.append(f"| {gate['metric']} | {gate['operator']} {gate['threshold']!r} | {gate['observed']!r} | {'PASS' if gate['passed'] else 'FAIL'} |")
    return "\n".join(lines)


def write_report(output: Path, summary: dict[str, Any]) -> Path:
    lines = ["# FT-011 final controlled learning-rate experiment", "", f"Decision: **{summary['decision']}**.", "", "The single DenseNet-201 candidate changes the initial backbone learning rate to 0.00001 while the classifier remains at 0.0001. Transferability research provides a rationale for testing different learning rates; it does not establish that this particular rate will improve pneumonia classification. [1] [2] The fixed screening and replacement rules guard against selecting a favorable result after inspecting it. [3] [7]", "", "## Exact decision gates", "", "Seed-42 screening:", "", gate_table(summary["screening_gates"]), ""]
    if summary["replacement_gates"]:
        lines.extend(["Three-seed replacement:", "", gate_table(summary["replacement_gates"]), ""])
    else:
        lines.extend(["Three-seed replacement gates were not evaluated. A single completed seed cannot establish the three-seed final selection.", ""])
    lines.extend(["## Actual validation results", "", "| Seed | Accuracy | Balanced accuracy | Macro F1 | Normal recall | Viral recall | Bacterial recall |", "|---|---:|---:|---:|---:|---:|---:|"])
    for seed, metrics in summary["runs"].items():
        values = metric_scalars(metrics)
        lines.append("| " + str(seed) + " | " + " | ".join(repr(values[name]) for name in ("accuracy", "balanced_accuracy", "macro_f1", "normal_recall", "viral_recall", "bacterial_recall")) + " |")
    for seed, metrics in summary["runs"].items():
        lines.extend(["", f"Seed {seed}, selected epoch {metrics['best_epoch']}:", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
        for name in CLASSES:
            values = metrics["per_class"][name]
            lines.append(f"| {name} | {values['precision']!r} | {values['recall']!r} | {values['f1']!r} | {values['support']} |")
        inference = metrics.get("inference", {})
        lines.extend(["", f"Recorded active training time: {metrics.get('training_duration_seconds')} seconds; peak allocated GPU memory: {metrics.get('peak_gpu_memory_bytes')} bytes. Inference latency: {inference.get('latency_ms_per_image')} ms per image, throughput: {inference.get('throughput_images_per_second')} images/second. The latency benchmark excludes preprocessing and uses the recorded measurement settings.", ""])
    lines.extend(["", "Each seed's metric is calculated separately. Means and sample standard deviations are in the machine-readable analysis.json; no probability vectors are averaged across confirmation seeds.", "", "## Uncertainty and interpretation", "", "Paired bootstrap uses 10,000 resamples. Each draw is shared by FT-011, FT-010 and FT-004 and every included seed. Entire derived split_group_id groups are sampled with replacement within class strata. These are proxy filename/hash clusters, not verified patient identifiers; they cannot establish patient independence. The bootstrap measures uncertainty on the already used validation cases. It does not produce independent external performance. [3] [4] [6]", "", "The incumbent FT-004 uses a previously frozen equal-probability ensemble, with separate seed-matched ensemble predictions for comparison. Ensemble research explains its rationale without proving superiority for this dataset. [5] [8]", "", "## Artifacts and next step", "", "The original locked train/validation assignments and preparation record were preserved. [9] No test images or test metrics were accessed. Candidate artifacts, exact gate evidence, hashes, bootstrap intervals and actual graphs are recorded alongside this report. The live website is unchanged. Independent evaluation of the frozen system is the next step. [4] [6]", ""])
    for path in summary.get("graphs", []):
        lines.append(f"![Actual result graph]({Path(path).name})")
    lines.extend(["", "## Numbered references", ""])
    for index, (title, url) in enumerate(REFERENCES, 1):
        lines.append(f"[{index}] [{title}]({url})")
    lines.append("")
    lines.extend(f"[{index}]: {url}" for index, (_, url) in enumerate(REFERENCES, 1))
    lines.extend(["", "Protocol references provide methodological rationale and reporting guidance; their use is not a claim of verified efficacy for the chosen FT-011 configuration.", ""])
    report = output / "FT011_Final_Report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def analyze_candidate(root: Path | str, terminal: str | None = None) -> dict[str, Any]:
    """Analyze completed matched candidates after immutable control freeze.

    ``root`` may be the experiment directory or the original repository. Every
    analysis gets a new output directory, preserving previous reports.
    """
    supplied = Path(root).resolve()
    experiment = supplied if (supplied / "protocol.json").is_file() else supplied / "model/results/ft011"
    protocol_path = experiment / "protocol.json"
    protocol = read_json(protocol_path)
    controls = protocol.get("controls")
    if not isinstance(controls, dict):
        raise RuntimeError("freeze control artifacts and exact gates in protocol.json before candidate outcome inspection")
    unhashed_protocol = dict(protocol)
    expected_protocol_hash = unhashed_protocol.pop("protocol_hash")
    if canonical_hash(unhashed_protocol) != expected_protocol_hash:
        raise RuntimeError("frozen experiment protocol digest mismatch")
    verify_frozen_controls(controls)
    repo = Path(next(artifact["path"] for artifact in controls["artifact_hashes"] if Path(artifact["path"]).name == "val_manifest.csv")).parents[2]
    manifest, _ = validation_manifest(repo)
    systems: dict[str, dict[int, np.ndarray]] = {"FT011": {}, "FT010": {}, "FT004": {}}
    runs: dict[int, dict[str, Any]] = {}
    histories = {}
    artifacts = [record_artifact(protocol_path)]
    screening: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = experiment / "runs" / f"ft011_densenet201_differential_lr_s{seed}"
        if not (run / "status.json").is_file() or read_json(run / "status.json").get("status") != "completed":
            break
        if seed != 42 and not all(gate["passed"] for gate in screening):
            raise RuntimeError("confirmation outcome found after failed seed-42 screening")
        _, labels, probabilities = aligned_predictions(run / "predictions.csv", manifest)
        saved = read_json(run / "metrics.json")
        if saved.get("split") != "validation":
            raise RuntimeError("candidate metrics split is not validation")
        computed = calculate_metrics(labels, probabilities, CLASSES)
        check_metrics(saved, computed, f"FT011 seed {seed}")
        runs[seed] = {**saved, **computed}
        histories[seed] = read_csv(run / "history.csv")
        if not histories[seed]:
            raise RuntimeError("completed candidate lacks epoch history")
        verify_candidate_configuration(run, seed, protocol, saved, histories[seed])
        for name in ("metrics.json", "predictions.csv", "history.csv", "status.json", "config.json", "candidate_configuration.json", "optimizer_groups.json", "scheduler_initial.json", "initialization.json", "best_checkpoint.pt"):
            artifacts.append(record_artifact(run / name))
        systems["FT011"][seed] = probabilities
        for name in ("FT010", "FT004"):
            path = Path(controls["systems"][name]["predictions"][str(seed)]["path"])
            _, comparator_labels, comparator_probabilities = aligned_predictions(path, manifest)
            if not np.array_equal(labels, comparator_labels):
                raise RuntimeError("paired comparison label mismatch")
            systems[name][seed] = comparator_probabilities
        if seed == 42:
            screening = evaluate_gates(metric_scalars(computed), controls["screening_gates"])
            if not all(gate["passed"] for gate in screening):
                break
    if 42 not in runs:
        raise RuntimeError("no complete seed-42 candidate available; training/selection is not complete")
    aggregate = aggregate_metrics(runs)
    replacement = evaluate_gates({name: value["mean"] for name, value in aggregate.items()}, controls["replacement_gates"]) if tuple(runs) == SEEDS else []
    if not all(gate["passed"] for gate in screening):
        decision, complete = "retain FT-004", True
    elif replacement:
        decision, complete = ("select FT-011" if all(gate["passed"] for gate in replacement) else "retain FT-004"), True
    else:
        decision, complete = "pending confirmation; retain frozen FT-004 meanwhile", False
    bootstrap_policy = controls["bootstrap"]
    if bootstrap_policy["iterations"] != 10000:
        raise RuntimeError("fixed final bootstrap must use 10000 resamples")
    bootstrap = paired_bootstrap(labels, systems, manifest, tuple(runs), 10000, bootstrap_policy["seed"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = experiment / "analysis" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "terminal_status": terminal,
               "decision": decision, "terminal_decision_complete": complete, "closed_tuning_phase": complete,
               "screening_passed": all(gate["passed"] for gate in screening),
               "screening_gates": screening, "replacement_gates": replacement, "completed_seeds": list(runs),
               "three_seed_confirmation_completed": tuple(runs) == SEEDS, "runs": {str(seed): value for seed, value in runs.items()},
               "aggregate": aggregate, "bootstrap": bootstrap, "artifact_hashes": artifacts,
               "locked_test_accessed": False, "live_website_changed": False,
               "frozen_incumbent": controls["frozen_incumbent"], "selected_deployment_seed": 42,
               "selected_candidate_checkpoint": record_artifact(experiment / "runs/ft011_densenet201_differential_lr_s42/best_checkpoint.pt") if decision == "select FT-011" else None,
               "interpretation_boundary": "Validation findings only. Confirmation metrics are averaged separately per seed. Seed42 deployment is predeclared, not retrospectively chosen.",
               "next_step": "independent evaluation of the frozen system", "output_directory": str(output)}
    summary["selected_package"] = {"architecture": "densenet201", "preprocessing": "VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE", "class_order": list(CLASSES), "prediction_rule": "argmax of one model's three class probabilities", "seed": 42, "checkpoint": summary["selected_candidate_checkpoint"]} if decision == "select FT-011" else {"system": "FT004", "frozen_selection": controls["frozen_incumbent"]}
    summary["graphs"] = write_graphs(output, runs, histories, controls)
    summary["report_path"] = str(write_report(output, summary))
    path = output / "analysis.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "model/results/ft011")
    parser.add_argument("--freeze-controls", action="store_true")
    parser.add_argument("--terminal")
    arguments = parser.parse_args()
    result = freeze_controls() if arguments.freeze_controls else analyze_candidate(arguments.root, arguments.terminal)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
