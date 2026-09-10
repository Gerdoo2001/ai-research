"""Materialize readable run.log files from crash-resistant status/history artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import _multiclass_bootstrap


def main() -> int:
    runs = _multiclass_bootstrap.REPO_ROOT / "model/results/runs"
    count = 0
    for run_dir in sorted(path for path in runs.iterdir() if path.is_dir()):
        status_path = run_dir / "status.json"
        config_path = run_dir / "config.json"
        if not status_path.exists() or not config_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        history = []
        history_path = run_dir / "history.csv"
        if history_path.exists() and history_path.stat().st_size:
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                history = list(csv.DictReader(handle))
        lines = [
            "Materialized from persisted config.json, status.json, and history.csv; no batch events are reconstructed.",
            f"materialized_at={datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"run_id={status.get('run_id')}",
            f"status={status.get('status')}",
            f"configuration_hash={status.get('config_hash')}",
            f"manifest_hash={status.get('manifest_hash')}",
            f"architecture={config.get('architecture')}",
            f"preprocessing={config.get('preprocessing')}",
            f"seed={config.get('seed')}",
            f"device={config.get('device')}",
            f"created_at={status.get('created_at')}",
            f"started_at={status.get('started_at')}",
            f"updated_at={status.get('updated_at')}",
            f"finished_at={status.get('finished_at')}",
            f"best_epoch={status.get('best_epoch')}",
            f"best_validation_macro_f1={status.get('best_validation_macro_f1')}",
            f"failure_reason={status.get('failure_reason')}",
            f"resume_command={status.get('resume_command')}",
        ]
        for row in history:
            lines.append(
                "epoch={epoch}/{total} train_loss={train_loss} val_loss={val_loss} "
                "val_accuracy={val_accuracy} val_balanced_accuracy={val_balanced_accuracy} "
                "val_macro_f1={val_macro_f1} recall_normal={recall_normal} "
                "recall_viral={recall_viral_or_pneumonia} recall_bacterial={recall_bacterial} "
                "learning_rate={learning_rate} elapsed_seconds={elapsed_seconds} "
                "peak_gpu_memory_bytes={peak_gpu_memory_bytes} best_epoch={best_epoch} "
                "best_validation_macro_f1={best_validation_macro_f1}".format(
                    total=status.get("total_epochs"), **row
                )
            )
        (run_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        count += 1
    print(f"Materialized run.log for {count} run directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
