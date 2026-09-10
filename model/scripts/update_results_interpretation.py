"""Regenerate live results interpretation from consolidated actual artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import _multiclass_bootstrap


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "not available"


def table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if key.startswith("validation_") or key.startswith("recall_"):
                value = metric(value)
            values.append(str(value).replace("|", "/") if value != "" else "-")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    ranking = read_csv(results / "model_ranking.csv")
    ablation_all = read_csv(results / "preprocessing_ablation.csv")
    completed_ablation = [
        row for row in ablation_all
        if row["stage"] == "preprocessing_ablation" and row["status"] == "completed"
    ]
    ablation = (
        [row for row in ablation_all if row["run_id"] == "baseline_densenet121_s42"]
        + completed_ablation
        if completed_ablation
        else []
    )
    trials = read_csv(results / "tuning_trials.csv")
    robustness = [row for row in read_csv(results / "seed_robustness.csv") if row["status"] == "completed"]
    test = [row for row in read_csv(results / "test_results.csv") if row["status"] == "completed"]
    summary = read_csv(results / "summary.csv")
    unsupported = [row for row in summary if row["status"] in {"unsupported", "skipped"}]
    failed = [row for row in summary if row["status"] in {"failed", "interrupted"}]
    columns = [
        ("Run", "run_id"), ("Architecture", "architecture"), ("Preprocessing", "preprocessing"),
        ("Macro F1", "validation_macro_f1"), ("Balanced accuracy", "validation_balanced_accuracy"),
        ("Normal recall", "recall_normal"), ("Viral recall", "recall_viral"),
        ("Bacterial recall", "recall_bacterial"), ("Status", "status"),
    ]
    lines = [
        "# Results Interpretation",
        "",
        "All values below are regenerated from consolidated per-run artifacts. Validation evidence is kept separate from the locked test result. Planned, failed, interrupted, skipped, and unsupported work remains visible.",
        "",
        "## Baseline Architecture Benchmark",
        "",
    ]
    if ranking:
        lines.extend(table(ranking, [("Rank", "rank"), *columns]))
        best = ranking[0]
        recalls = {
            "Normal": float(best["recall_normal"]),
            "Viral": float(best["recall_viral"]),
            "Bacterial": float(best["recall_bacterial"]),
        }
        hardest = min(recalls, key=recalls.get)
        lines.extend(
            [
                "",
                f"The current validation leader is `{best['run_id']}` at macro F1 {metric(best['validation_macro_f1'])} and balanced accuracy {metric(best['validation_balanced_accuracy'])}. Its lowest class recall is {hardest} at {metric(recalls[hardest])}. This ranking is final only when all eight supported baselines have completed.",
            ]
        )
    else:
        lines.append("No controlled baseline has completed.")
    lines.extend(["", "## Preprocessing Ablation", ""])
    if ablation:
        lines.extend(table(sorted(ablation, key=lambda row: float(row["validation_macro_f1"]), reverse=True), columns))
        lines.extend(["", "Preprocessing is selected by validation macro F1 with balanced accuracy and minimum class recall as tie-breakers. Pseudo-coloring is not interpreted as adding clinical information."])
    else:
        lines.append("No supported preprocessing ablation has completed.")
    lines.extend(["", "## General Optuna Tuning", ""])
    if trials:
        trial_columns = [
            ("Study", "study_id"), ("Trial", "trial_number"), ("Run", "run_id"),
            ("Status", "status"), ("Macro F1", "validation_macro_f1"),
            ("Balanced accuracy", "validation_balanced_accuracy"), ("Reason", "reason"),
        ]
        lines.extend(table(trials, trial_columns))
        lines.extend(["", "The bounded TPE search is general validation-only tuning and is not called SAS Tuning. The small trial budget limits optimization claims."])
    else:
        lines.append("No tuning trial has run. SAS Tuning remains unresolved.")
    lines.extend(["", "## Multi-Seed Robustness", ""])
    if robustness:
        lines.extend(table(robustness, columns))
        if len(robustness) == 3:
            import statistics

            scores = [float(row["validation_macro_f1"]) for row in robustness]
            lines.extend(["", f"Across seeds 42, 123, and 2026, validation macro F1 mean is {statistics.mean(scores):.6f} with sample standard deviation {statistics.stdev(scores):.6f}."])
    else:
        lines.append("No final robustness retrain has completed.")
    lines.extend(["", "## Frozen Locked-Test Evaluation", ""])
    if test:
        row = test[0]
        lines.extend(table(test, columns))
        lines.extend(["", f"The one frozen locked-test result is macro F1 {metric(row['validation_macro_f1'])} and balanced accuracy {metric(row['validation_balanced_accuracy'])}. Despite the consolidated column names, this row's source `metrics.json` is explicitly labeled `locked_test`; it was not used for selection."])
    else:
        lines.append("The locked test manifest remains sealed; no test metric exists.")
    lines.extend(["", "## Failed, Unsupported, and Unresolved Work", ""])
    if unsupported:
        lines.extend(table(unsupported, [("Run", "run_id"), ("Stage", "stage"), ("Status", "status"), ("Reason", "reason")]))
    if failed:
        lines.extend(["", *table(failed, [("Run", "run_id"), ("Stage", "stage"), ("Status", "status"), ("Reason", "reason")])])
    lines.extend(
        [
            "",
            "YOLOv10 classification is unsupported because a genuine classification head/checkpoint is absent. Raw-image PCA is deferred, and FILM, SAS Tuning, and automated grading remain unresolved. No missing result is estimated or fabricated.",
            "",
            "## Interpretation Limits",
            "",
            "Performance comes from one public data source with filename-derived etiology labels and no authoritative patient metadata. Viral/Bacterial confusion, label noise, source/device shift, calibration, subgroup behavior, and external generalization require further study. This is research evidence only and not clinical deployment validation.",
            "",
        ]
    )
    output = repo / "model/docs/05_results_interpretation.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote live interpretation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
