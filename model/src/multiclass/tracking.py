"""Crash-resistant run status, ledger, history, and journal updates."""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


LEDGER_COLUMNS = [
    "run_id", "config_hash", "architecture", "preprocessing", "task_mode", "seed",
    "manifest_hash", "status", "created_at", "started_at", "updated_at", "finished_at",
    "best_epoch", "best_validation_macro_f1", "run_dir", "failure_reason", "resume_command",
]
FAILURE_COLUMNS = [
    "timestamp", "run_id", "phase", "command", "error_type", "error_message",
    "run_dir", "resolved", "issue_id",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def replace_with_retry(source: Path, destination: Path, *, attempts: int = 20) -> None:
    """Atomically replace a file despite brief Windows reader/antivirus locks."""

    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.5, 0.025 * (2**attempt)))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replace_with_retry(temp, path)


def append_csv(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    columns = list(history[0])
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temp, path)


def update_ledger(record: dict[str, Any]) -> None:
    csv_path = REPO_ROOT / "model/results/experiment_ledger.csv"
    json_path = REPO_ROOT / "model/results/experiment_ledger.json"
    records: list[dict[str, Any]] = []
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    by_id = {item["run_id"]: item for item in records if item.get("run_id")}
    existing = by_id.get(record["run_id"], {})
    existing.update({key: value for key, value in record.items() if value is not None})
    by_id[record["run_id"]] = existing
    ordered = [by_id[key] for key in sorted(by_id)]
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in ordered:
            writer.writerow({column: item.get(column, "") for column in LEDGER_COLUMNS})
    replace_with_retry(temp, csv_path)
    atomic_write_json(
        json_path,
        {"schema_version": 1, "updated_at": now_iso(), "experiments": ordered},
    )
    matrix_path = REPO_ROOT / "model/results/run_matrix.csv"
    if matrix_path.exists() and matrix_path.stat().st_size:
        with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
            matrix_rows = list(csv.DictReader(handle))
            fieldnames = list(matrix_rows[0]) if matrix_rows else []
        changed = False
        for row in matrix_rows:
            if row.get("run_id") == record["run_id"]:
                row["status"] = str(record.get("status", row.get("status", "")))
                if record.get("failure_reason"):
                    row["reason"] = str(record["failure_reason"])
                changed = True
        if changed:
            temp_matrix = matrix_path.with_suffix(".csv.tmp")
            with temp_matrix.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(matrix_rows)
            replace_with_retry(temp_matrix, matrix_path)


def append_failure(run_id: str, run_dir: Path, phase: str, command: str, error: BaseException) -> None:
    append_csv(
        REPO_ROOT / "model/results/failures.csv",
        {
            "timestamp": now_iso(),
            "run_id": run_id,
            "phase": phase,
            "command": command,
            "error_type": type(error).__name__,
            "error_message": str(error).replace("\n", " "),
            "run_dir": str(run_dir),
            "resolved": "false",
            "issue_id": "",
        },
        FAILURE_COLUMNS,
    )


def append_journal(title: str, fields: dict[str, Any]) -> None:
    path = REPO_ROOT / "model/docs/TRAINING_JOURNAL.md"
    lines = ["", f"## {now_iso()} - {title}", ""]
    for key, value in fields.items():
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        elif isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"- **{key}:** {rendered}")
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def update_current_status(run_id: str, event: str) -> None:
    ledger_path = REPO_ROOT / "model/results/experiment_ledger.csv"
    records: list[dict[str, Any]] = []
    if ledger_path.exists() and ledger_path.stat().st_size:
        with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    matrix_path = REPO_ROOT / "model/results/run_matrix.csv"
    matrix: list[dict[str, Any]] = []
    if matrix_path.exists() and matrix_path.stat().st_size:
        with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
            matrix = list(csv.DictReader(handle))
    matrix_by_id = {row["run_id"]: row for row in matrix}
    matrix_run_id = run_id
    if matrix_run_id not in matrix_by_id and "_trial_" in matrix_run_id:
        matrix_run_id = matrix_run_id.split("_trial_", 1)[0]
    stage = matrix_by_id.get(matrix_run_id, {}).get("stage", "training")
    phase_labels = {
        "baseline": "Phase 6 - Stage B baseline architecture benchmark",
        "preprocessing_ablation": "Phase 6 - Stage C preprocessing ablation",
        "focused_comparison": "Phase 6 - Stage D focused comparison",
        "tuning": "Phase 6 - Stage E Optuna general tuning",
        "robustness": "Phase 6 - Stage F multi-seed robustness",
        "locked_test": "Phase 7 - Stage G frozen locked-test evaluation",
    }
    task_labels = {
        "baseline": "Execute supported baselines sequentially on the locked training/validation manifests.",
        "preprocessing_ablation": "Compare baseline, CLAHE, global histogram equalization, and exploratory VIRIDIS on DenseNet169, DenseNet201, and ResNet50.",
        "focused_comparison": "Test the validation-selected preprocessing on the top baseline architectures.",
        "tuning": "Run bounded Optuna TPE studies using validation macro F1 only.",
        "robustness": "Independently retrain the frozen selected configuration at seeds 42, 123, and 2026.",
        "locked_test": "Perform the single explicitly unlocked evaluation after validation-only selection was frozen.",
    }
    excluded_prefixes = ("smoke_", "pilot_")
    model_records = [
        record for record in records
        if not record.get("run_id", "").startswith(excluded_prefixes)
        and record.get("run_id") not in {"final_locked_test", "tuning_top1", "tuning_top2", "tuning_top3"}
    ]
    completed = [record for record in model_records if record.get("status") == "completed"]
    failed = [record for record in model_records if record.get("status") == "failed"]
    active = [record for record in records if record.get("status") in {"queued", "running", "interrupted"}]
    pending_rows = [
        row for row in matrix
        if row.get("stage") not in {"smoke", "pilot", "locked_test"}
        and row.get("status") in {"planned", "queued", "running", "interrupted"}
    ]
    best = None
    for record in completed:
        try:
            score = float(record.get("best_validation_macro_f1", ""))
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, record)
    active_record = next((record for record in records if record.get("run_id") == run_id), None)
    active_text = "None"
    if active:
        active_text = ", ".join(f"{record['run_id']} ({record['status']})" for record in active)
    elif active_record:
        active_text = f"{run_id} ({active_record.get('status', 'unknown')})"
    best_text = "Not available; no controlled validation run has completed."
    if best:
        best_text = f"{best[1]['run_id']} - validation macro F1 {best[0]:.6f}"
    default_resume = {
        "baseline": '.venv\\Scripts\\python.exe model\\scripts\\run_benchmark.py --stage baseline --resume',
        "preprocessing_ablation": '.venv\\Scripts\\python.exe model\\scripts\\run_benchmark.py --stage preprocessing_ablation --resume',
        "focused_comparison": '.venv\\Scripts\\python.exe model\\scripts\\resolve_run_matrix.py --stage focused',
        "tuning": '.venv\\Scripts\\python.exe model\\scripts\\run_research_pipeline.py --from-stage tuning',
        "robustness": '.venv\\Scripts\\python.exe model\\scripts\\run_research_pipeline.py --from-stage robustness',
        "locked_test": '.venv\\Scripts\\python.exe model\\scripts\\run_research_pipeline.py --from-stage test',
    }.get(stage, '.venv\\Scripts\\python.exe model\\scripts\\run_research_pipeline.py --from-stage post_baseline')
    resume = (
        active_record.get("resume_command")
        if active_record and active_record.get("resume_command")
        else default_resume
    )
    locked_test = next((record for record in records if record.get("run_id") == "final_locked_test"), None)
    test_text = (
        "The single frozen locked-test evaluation has completed."
        if locked_test and locked_test.get("status") == "completed"
        else "The locked test manifest remains sealed."
    )
    text = f"""# Current Status

- **Current phase:** {phase_labels.get(stage, "Phase 6 - controlled training execution")}
- **Current task:** {task_labels.get(stage, "Execute the next validation-controlled research stage.")}
- **Last completed action:** {event}
- **Active or most recent training run:** {active_text}
- **Completed model count:** {len(completed)} controlled trained configurations
- **Failed model count:** {len(failed)} controlled trained configurations
- **Pending model count:** {len(pending_rows)} planned/queued/running controlled matrix rows; YOLOv10 classification remains unsupported
- **Current best validation model and metric:** {best_text}
- **Known blockers:** FILM, SAS Tuning, and automated grading remain unresolved; YOLOv10 classification lacks a genuine supported head; PCA is deferred to a justified train-only feature-space design. Word COM rendering and locked-file save fallback are verified; LibreOffice is absent.
- **Exact resume command:** `{resume}`
- **Last updated:** {now_iso()}

The immutable source remains unchanged. Training and validation manifests are in use. {test_text}
"""
    path = REPO_ROOT / "model/docs/CURRENT_STATUS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    replace_with_retry(temporary, path)
