"""Run a resumable Optuna TPE study against locked validation data only.

Each trial is a normal crash-resistant training run with its own directory and
checkpoints. The Optuna database is the study source of truth; a flat CSV/JSON
export is regenerated after every trial so failed and interrupted work remains
visible without requiring Optuna to inspect it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import _multiclass_bootstrap
from multiclass.config import ExperimentConfig
from multiclass.tracking import append_journal, now_iso, update_current_status, update_ledger
from multiclass.trainer import run_training


TRIAL_COLUMNS = [
    "study_id",
    "trial_id",
    "trial_number",
    "parent_run_id",
    "run_id",
    "status",
    "parameters_json",
    "validation_macro_f1",
    "validation_balanced_accuracy",
    "minimum_per_class_recall",
    "started_at",
    "completed_at",
    "duration_seconds",
    "run_dir",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run-id", required=True, help="Matrix row such as tuning_top1")
    parser.add_argument("--parent-run-id", required=True, help="Completed baseline/focused run to tune")
    parser.add_argument("--trials", type=int, default=4, help="Target total trials, including existing trials")
    parser.add_argument("--epochs", type=int, default=6, help="Equal budget used by every tuning trial")
    parser.add_argument("--resume", action="store_true", help="Continue an existing SQLite study")
    return parser.parse_args()


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trial_state_name(trial) -> str:
    name = trial.state.name.lower()
    return {"complete": "completed", "fail": "failed", "pruned": "pruned"}.get(name, name)


def remaining_optimization_trials(study, target_trials: int) -> int:
    """Return how many trials ``study.optimize`` must finish to reach the target.

    Enqueued trials are stored in Optuna with the WAITING state before they have
    run. Counting those rows as already executed causes an off-by-one error: a
    target of four with one queued control would execute only three trials.  The
    optimization budget must therefore be based on terminal trials, because an
    invocation of ``study.optimize(..., n_trials=N)`` includes WAITING trials it
    consumes among those N finished attempts.
    """
    terminal_states = {
        "COMPLETE",
        "FAIL",
        "PRUNED",
    }
    terminal_count = sum(trial.state.name in terminal_states for trial in study.trials)
    return max(0, target_trials - terminal_count)


def export_trials(study, *, study_id: str, parent_run_id: str, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frozen in study.get_trials(deepcopy=False):
        run_id = frozen.user_attrs.get("run_id", f"{study_id}_trial_{frozen.number:03d}")
        metrics = frozen.user_attrs.get("metrics", {})
        recalls = [
            metrics.get("per_class", {}).get(name, {}).get("recall")
            for name in ("Normal", "Viral", "Bacterial")
        ]
        recalls = [float(value) for value in recalls if value is not None]
        duration = None
        if frozen.datetime_start and frozen.datetime_complete:
            duration = (frozen.datetime_complete - frozen.datetime_start).total_seconds()
        rows.append(
            {
                "study_id": study_id,
                "trial_id": f"{study_id}:{frozen.number}",
                "trial_number": frozen.number,
                "parent_run_id": parent_run_id,
                "run_id": run_id,
                "status": trial_state_name(frozen),
                "parameters_json": json.dumps(frozen.params, sort_keys=True),
                "validation_macro_f1": frozen.value if frozen.value is not None else "",
                "validation_balanced_accuracy": metrics.get("balanced_accuracy", ""),
                "minimum_per_class_recall": min(recalls) if recalls else "",
                "started_at": frozen.datetime_start.isoformat() if frozen.datetime_start else "",
                "completed_at": frozen.datetime_complete.isoformat() if frozen.datetime_complete else "",
                "duration_seconds": duration if duration is not None else "",
                "run_dir": frozen.user_attrs.get("run_dir", ""),
                "reason": frozen.user_attrs.get("failure_reason", ""),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "trials.json").write_text(
        json.dumps({"schema_version": 1, "study_id": study_id, "trials": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> int:
    import optuna

    args = parse_args()
    if args.trials < 1 or args.epochs < 1:
        raise ValueError("--trials and --epochs must be positive")

    repo = _multiclass_bootstrap.REPO_ROOT
    results_root = repo / "model/results"
    parent_dir = results_root / "runs" / args.parent_run_id
    parent_status_path = parent_dir / "status.json"
    parent_config_path = parent_dir / "config.json"
    if not parent_status_path.exists() or not parent_config_path.exists():
        raise FileNotFoundError(f"parent run artifacts not found: {parent_dir}")
    parent_status = json.loads(parent_status_path.read_text(encoding="utf-8"))
    if parent_status.get("status") != "completed":
        raise RuntimeError(f"parent run must be completed, found {parent_status.get('status')!r}")
    parent_payload = json.loads(parent_config_path.read_text(encoding="utf-8"))

    study_id = args.matrix_run_id
    study_dir = results_root / "tuning" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    database_path = study_dir / "study.db"
    if database_path.exists() and not args.resume:
        raise FileExistsError(f"study already exists; pass --resume: {database_path}")

    study_payload = {
        "schema_version": 1,
        "study_id": study_id,
        "parent_run_id": args.parent_run_id,
        "sampler": "Optuna TPESampler",
        "sampler_seed": 42,
        "direction": "maximize validation macro F1",
        "target_trials": args.trials,
        "trial_epochs": args.epochs,
        "locked_manifest_hash": parent_status["manifest_hash"],
        "test_access": "prohibited",
        "pre_registered_control_trial": True,
        "search_space_file": "model/configs/tuning_search_space_continuation_v2.json",
    }
    study_hash = stable_hash(study_payload)
    (study_dir / "study_config.json").write_text(json.dumps(study_payload, indent=2) + "\n", encoding="utf-8")
    resume_command = (
        f".venv\\Scripts\\python.exe model\\scripts\\run_tuning.py --matrix-run-id {study_id} "
        f"--parent-run-id {args.parent_run_id} --trials {args.trials} --epochs {args.epochs} --resume"
    )
    started_at = now_iso()
    ledger_base = {
        "run_id": study_id,
        "config_hash": study_hash,
        "architecture": parent_payload["architecture"],
        "preprocessing": parent_payload["preprocessing"],
        "task_mode": parent_payload["task_mode"],
        "seed": parent_payload["seed"],
        "manifest_hash": parent_status["manifest_hash"],
        "created_at": started_at,
        "started_at": started_at,
        "updated_at": started_at,
        "run_dir": str(study_dir),
        "resume_command": resume_command,
    }
    update_ledger({**ledger_base, "status": "running"})
    update_current_status(study_id, f"Started Optuna study {study_id} from parent {args.parent_run_id}.")
    append_journal(
        f"Optuna study {study_id} started",
        {
            "Run ID": study_id,
            "Parent run": args.parent_run_id,
            "Configuration hash": study_hash,
            "Architecture": parent_payload["architecture"],
            "Preprocessing": parent_payload["preprocessing"],
            "Manifest hash": parent_status["manifest_hash"],
            "Target trials": args.trials,
            "Epochs per trial": args.epochs,
            "Selection metric": "validation macro F1",
            "Locked test access": "prohibited",
            "Status": "running",
        },
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(3, args.trials))
    study = optuna.create_study(
        study_name=study_id,
        storage=f"sqlite:///{database_path.as_posix()}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=args.resume,
    )
    if not study.trials:
        control_learning_rate_key = (
            "learning_rate_sgd" if parent_payload["optimizer"] == "sgd" else "learning_rate_adamw"
        )
        study.enqueue_trial(
            {
                "optimizer": parent_payload["optimizer"],
                control_learning_rate_key: parent_payload["learning_rate"],
                "weight_decay": parent_payload["weight_decay"],
                "scheduler": parent_payload["scheduler"],
                "loss": parent_payload["loss"],
                "class_weighting": parent_payload["class_weighting"],
                "label_smoothing": parent_payload["label_smoothing"],
                "augmentation": (
                    "none"
                    if parent_payload["augmentation"]["horizontal_flip_probability"] == 0
                    and parent_payload["augmentation"]["rotation_degrees"] == 0
                    else "conservative"
                ),
            }
        )

    def objective(trial) -> float:
        optimizer = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
        if optimizer == "sgd":
            learning_rate = trial.suggest_float("learning_rate_sgd", 1e-4, 3e-3, log=True)
        else:
            learning_rate = trial.suggest_float("learning_rate_adamw", 3e-5, 3e-4, log=True)
        augmentation = trial.suggest_categorical("augmentation", ["none", "conservative"])
        payload = dict(parent_payload)
        payload.update(
            {
                "run_id": f"{study_id}_trial_{trial.number:03d}",
                "epochs": args.epochs,
                "seed": 42,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "scheduler": trial.suggest_categorical("scheduler", ["reduce_on_plateau", "cosine"]),
                "loss": trial.suggest_categorical("loss", ["cross_entropy", "focal"]),
                "class_weighting": trial.suggest_categorical("class_weighting", ["balanced", "none"]),
                "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.05]),
                "augmentation": {
                    "horizontal_flip_probability": 0.0 if augmentation == "none" else 0.5,
                    "rotation_degrees": 0.0 if augmentation == "none" else 5.0,
                },
                "controlled_stop_after_epoch": None,
                "smoke_test": False,
                "max_train_samples": None,
                "max_val_samples": None,
            }
        )
        config = ExperimentConfig(**payload)
        run_dir = config.results_path / config.effective_run_id()
        trial.set_user_attr("run_id", config.effective_run_id())
        trial.set_user_attr("run_dir", str(run_dir))
        try:
            checkpoint = run_dir / "last_checkpoint.pt"
            run_training(config, resume=checkpoint.exists())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            trial.set_user_attr("metrics", metrics)
            return float(metrics["macro_f1"])
        except Exception as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            raise

    def after_trial(study, frozen) -> None:
        rows = export_trials(study, study_id=study_id, parent_run_id=args.parent_run_id, output_dir=study_dir)
        append_journal(
            f"Optuna study {study_id} trial {frozen.number} {trial_state_name(frozen)}",
            {
                "Run ID": frozen.user_attrs.get("run_id", f"{study_id}_trial_{frozen.number:03d}"),
                "Parent run": args.parent_run_id,
                "Trial": frozen.number,
                "Parameters": frozen.params,
                "Validation macro F1": frozen.value if frozen.value is not None else "not available",
                "Status": trial_state_name(frozen),
                "Error": frozen.user_attrs.get("failure_reason", "none"),
                "Persisted trial count": len(rows),
            },
        )

    try:
        remaining = remaining_optimization_trials(study, args.trials)
        if remaining:
            study.optimize(objective, n_trials=remaining, callbacks=[after_trial], catch=(Exception,))
        rows = export_trials(study, study_id=study_id, parent_run_id=args.parent_run_id, output_dir=study_dir)
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("study has no completed trial")
        best = study.best_trial
        best_run_dir = Path(best.user_attrs["run_dir"])
        shutil.copy2(best_run_dir / "config.json", study_dir / "best_config.json")
        summary = {
            **study_payload,
            "completed_at": now_iso(),
            "persisted_trial_count": len(rows),
            "completed_trial_count": len(completed),
            "best_trial_number": best.number,
            "best_run_id": best.user_attrs["run_id"],
            "best_validation_macro_f1": best.value,
            "best_parameters": best.params,
            "best_run_dir": str(best_run_dir),
        }
        (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        finished = now_iso()
        update_ledger(
            {
                **ledger_base,
                "status": "completed",
                "updated_at": finished,
                "finished_at": finished,
                "best_epoch": best.user_attrs.get("metrics", {}).get("best_epoch", ""),
                "best_validation_macro_f1": best.value,
            }
        )
        update_current_status(study_id, f"Completed Optuna study {study_id}; its best trial was frozen.")
        append_journal(
            f"Optuna study {study_id} finished",
            {
                "Run ID": study_id,
                "Parent run": args.parent_run_id,
                "Completed trials": len(completed),
                "Best trial": best.number,
                "Best trial run": best.user_attrs["run_id"],
                "Best validation macro F1": best.value,
                "Status": "completed",
            },
        )
        print(json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        update_ledger({**ledger_base, "status": "interrupted", "updated_at": now_iso()})
        update_current_status(study_id, f"Optuna study {study_id} was interrupted after persisted trials.")
        export_trials(study, study_id=study_id, parent_run_id=args.parent_run_id, output_dir=study_dir)
        raise
    except Exception as error:
        finished = now_iso()
        update_ledger(
            {
                **ledger_base,
                "status": "failed",
                "updated_at": finished,
                "finished_at": finished,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        )
        update_current_status(study_id, f"Optuna study {study_id} failed; its database and trial exports were preserved.")
        export_trials(study, study_id=study_id, parent_run_id=args.parent_run_id, output_dir=study_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
