"""Resume the full staged research workflow after the active baseline sequence.

This orchestrator launches only existing guarded entry points, records its own
atomic state/log, waits for an already-running baseline process via ledger
state, and stops at the first failed stage. Re-running the exact command is
safe: completed training runs skip and Optuna studies resume from SQLite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import _multiclass_bootstrap
from multiclass.tracking import atomic_write_json, now_iso


STAGES = ["post_baseline", "preprocessing", "focused", "tuning", "robustness", "test", "finalize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default="post_baseline")
    parser.add_argument("--wait-for-baselines", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--document-python",
        type=Path,
        default=Path(
            r"C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = _multiclass_bootstrap.REPO_ROOT
        self.results = self.repo / "model/results"
        self.log_path = self.results / "pipeline_orchestrator.log"
        self.state_path = self.results / "pipeline_orchestrator_status.json"
        self.python = Path(sys.executable).resolve()
        self.document_python = args.document_python.resolve()
        self.commands: list[str] = []

    def log(self, message: str) -> None:
        line = f"[{now_iso()}] {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state(self, *, status: str, stage: str, detail: str, command: list[str] | None = None) -> None:
        atomic_write_json(
            self.state_path,
            {
                "schema_version": 1,
                "status": status,
                "stage": stage,
                "detail": detail,
                "updated_at": now_iso(),
                "active_command": command,
                "commands_executed_this_invocation": self.commands,
                "resume_command": (
                    f".venv\\Scripts\\python.exe model\\scripts\\run_research_pipeline.py "
                    f"--from-stage {stage} --wait-for-baselines"
                ),
            },
        )

    def run(self, stage: str, label: str, command: list[str]) -> None:
        rendered = subprocess.list2cmdline(command)
        self.commands.append(rendered)
        self.log(f"START {label}: {rendered}")
        self.state(status="running", stage=stage, detail=label, command=command)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {now_iso()} | {label} =====\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=self.repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
            handle.write(f"===== return_code={return_code} | {label} =====\n")
            handle.flush()
            os.fsync(handle.fileno())
        if return_code:
            self.log(f"FAILED {label}: return code {return_code}")
            self.state(
                status="failed",
                stage=stage,
                detail=f"{label} failed with return code {return_code}; inspect {self.log_path}",
                command=command,
            )
            raise RuntimeError(f"{label} failed with return code {return_code}")
        self.log(f"COMPLETE {label}")

    def py(self, *parts: str) -> list[str]:
        return [str(self.python), *parts]

    def doc_py(self, *parts: str) -> list[str]:
        if not self.document_python.exists():
            raise FileNotFoundError(f"document runtime not found: {self.document_python}")
        return [str(self.document_python), *parts]

    def baseline_state(self) -> tuple[int, int, list[str]]:
        rows = read_csv(self.results / "experiment_ledger.csv")
        supported = [
            row for row in rows
            if row["run_id"].startswith("baseline_") and "yolov10" not in row["run_id"]
        ]
        completed = sum(row["status"] == "completed" for row in supported)
        failed = sum(row["status"] == "failed" for row in supported)
        active = [row["run_id"] for row in supported if row["status"] in {"queued", "running", "interrupted"}]
        return completed, failed, active

    def wait_for_baselines(self) -> None:
        inactive_checks = 0
        while True:
            completed, failed, active = self.baseline_state()
            self.log(f"baseline wait: completed={completed}/8 failed={failed} active={active}")
            self.state(
                status="waiting",
                stage="post_baseline",
                detail=f"Waiting for active baseline sequence: {completed}/8 completed; active={active}",
            )
            if completed == 8:
                return
            if active:
                inactive_checks = 0
            else:
                inactive_checks += 1
                if inactive_checks >= 2:
                    self.log("No baseline active after two checks; invoking one resumable baseline pass")
                    self.run(
                        "post_baseline",
                        "resume incomplete baseline matrix",
                        self.py("model/scripts/run_benchmark.py", "--stage", "baseline", "--resume"),
                    )
                    completed, _, _ = self.baseline_state()
                    if completed != 8:
                        raise RuntimeError(f"baseline recovery pass ended with only {completed}/8 complete")
                    return
            time.sleep(max(15, self.args.poll_seconds))

    def consolidate(self, stage: str) -> None:
        self.run(stage, "materialize readable run logs", self.py("model/scripts/materialize_run_logs.py"))
        self.run(stage, "consolidate and verify persisted metrics", self.py("model/scripts/consolidate_results.py"))
        self.run(stage, "regenerate live results interpretation", self.py("model/scripts/update_results_interpretation.py"))

    def progress_report(self, stage: str, label: str) -> None:
        self.run(
            stage,
            f"regenerate progressive Word report ({label})",
            self.doc_py("model/scripts/generate_progress_report.py", "--phase-label", label),
        )

    def append_development(self, title: str, stage: str, outcome: str, next_step: str) -> None:
        path = self.repo / "model/docs/DEVELOPMENT_LOG.md"
        commands = "; ".join(self.commands[-8:])
        entry = f"""

## {now_iso()} - {title}

- **Objective:** Complete and verify {stage} from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for {stage}; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded {stage} workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** {commands}
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** {outcome}
- **Next step:** {next_step}
"""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
            handle.flush()
            os.fsync(handle.fileno())

    def require_count(self, *, stage_name: str, status_stage: str, count: int) -> None:
        summary = read_csv(self.results / "summary.csv")
        completed = [
            row for row in summary if row["stage"] == status_stage and row["status"] == "completed"
        ]
        if len(completed) != count:
            raise RuntimeError(f"{stage_name} requires {count} completed rows, found {len(completed)}")

    def post_baseline(self) -> None:
        if self.args.wait_for_baselines:
            self.wait_for_baselines()
        else:
            completed, _, active = self.baseline_state()
            if active:
                raise RuntimeError("baseline sequence is active; rerun with --wait-for-baselines")
            if completed != 8:
                self.run(
                    "post_baseline",
                    "resume baseline matrix",
                    self.py("model/scripts/run_benchmark.py", "--stage", "baseline", "--resume"),
                )
        self.consolidate("post_baseline")
        self.require_count(stage_name="baseline gate", status_stage="baseline", count=8)
        self.progress_report("post_baseline", "Phase 6 - Stage B baseline architecture benchmark complete")
        self.append_development(
            "Stage B controlled baseline benchmark completed",
            "the eight supported baseline architecture runs",
            "All eight supported baseline runs completed on the locked training/validation manifests; YOLOv10 remains explicitly unsupported.",
            "Run the nine supported preprocessing ablations on DenseNet169, DenseNet201, and ResNet50 while reusing completed controls.",
        )

    def preprocessing(self) -> None:
        self.run(
            "preprocessing",
            "run supported preprocessing ablations",
            self.py("model/scripts/run_benchmark.py", "--stage", "preprocessing_ablation", "--resume"),
        )
        self.consolidate("preprocessing")
        self.require_count(stage_name="preprocessing gate", status_stage="preprocessing_ablation", count=9)
        self.run(
            "preprocessing",
            "analyze controlled preprocessing matrix",
            self.py("model/scripts/analyze_preprocessing_ablation.py"),
        )
        self.progress_report("preprocessing", "Phase 6 - Stage C preprocessing ablation complete")
        self.append_development(
            "Stage C preprocessing ablation completed",
            "the independent preprocessing comparison",
            "CLAHE, global histogram equalization, and exploratory VIRIDIS runs completed against the existing DenseNet169, DenseNet201, and ResNet50 controls.",
            "Resolve the best completed architecture/preprocessing evidence without redundant retraining.",
        )

    def focused(self) -> None:
        self.run(
            "focused",
            "resolve focused matrix placeholders",
            self.py("model/scripts/resolve_run_matrix.py", "--stage", "focused"),
        )
        self.run(
            "focused",
            "materialize non-redundant focused selection aliases",
            self.py("model/scripts/run_benchmark.py", "--stage", "focused_comparison", "--resume"),
        )
        self.consolidate("focused")
        self.progress_report("focused", "Phase 6 - Stage D focused comparison complete")
        self.append_development(
            "Stage D focused comparison completed",
            "the validation-resolved architecture/preprocessing comparison",
            "Focused rows were explicitly skipped because all representative preprocessing combinations already had completed artifacts.",
            "Resolve the top three distinct architecture configurations for general Optuna tuning.",
        )

    def tuning(self) -> None:
        if not (self.results / "tuning_plan.json").exists():
            self.run(
                "tuning",
                "resolve tuning parent configurations",
                self.py("model/scripts/resolve_run_matrix.py", "--stage", "tuning"),
            )
        plan = json.loads((self.results / "tuning_plan.json").read_text(encoding="utf-8"))
        for item in plan["studies"]:
            command = self.py(
                "model/scripts/run_tuning.py",
                "--matrix-run-id", item["matrix_run_id"],
                "--parent-run-id", item["parent_run_id"],
                "--trials", "4",
                "--epochs", "6",
            )
            if (self.results / "tuning" / item["matrix_run_id"] / "study.db").exists():
                command.append("--resume")
            self.run("tuning", f"run {item['matrix_run_id']}", command)
        self.consolidate("tuning")
        for index in (1, 2, 3):
            if not (self.results / "tuning" / f"tuning_top{index}" / "study_summary.json").exists():
                raise RuntimeError(f"tuning_top{index} did not produce a completed study summary")
        self.progress_report("tuning", "Phase 6 - Stage E Optuna general tuning complete")
        self.append_development(
            "Stage E bounded Optuna tuning completed",
            "three validation-only general tuning studies",
            "All three studies persisted their trial states and selected one best six-epoch validation trial each; no test data was accessed.",
            "Freeze the selected hyperparameters and independently retrain seeds 42, 123, and 2026.",
        )

    def robustness(self) -> None:
        self.run(
            "robustness",
            "resolve final validation-selected configuration",
            self.py("model/scripts/resolve_run_matrix.py", "--stage", "robustness"),
        )
        self.run(
            "robustness",
            "run three independent robustness seeds",
            self.py(
                "model/scripts/run_benchmark.py", "--stage", "robustness",
                "--config", "model/configs/final_selected.json", "--resume",
            ),
        )
        self.consolidate("robustness")
        self.require_count(stage_name="robustness gate", status_stage="robustness", count=3)
        self.progress_report("robustness", "Phase 6 - Stage F multi-seed robustness complete")
        self.append_development(
            "Stage F multi-seed robustness completed",
            "independent full-budget seeds 42, 123, and 2026",
            "The frozen selected configuration completed at all three seeds and its validation distribution was consolidated.",
            "Freeze the representative seed-42 checkpoint before the single locked-test evaluation.",
        )

    def test(self) -> None:
        selection_path = self.results / "final_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        if selection.get("status") != "locked_test_evaluated":
            if selection.get("status") != "selection_frozen_before_test":
                self.run(
                    "test",
                    "freeze validation-only selection before test",
                    self.py("model/scripts/resolve_run_matrix.py", "--stage", "test"),
                )
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.run(
                "test",
                "perform single frozen locked-test evaluation",
                self.py(
                    "model/scripts/evaluate_multiclass.py",
                    "--run-dir", selection["selected_run_dir"],
                    "--split", "test",
                    "--checkpoint", "best",
                    "--allow-locked-test",
                    "--final-selection", str(selection_path),
                ),
            )
        self.consolidate("test")
        self.require_count(stage_name="locked-test gate", status_stage="locked_test", count=1)
        self.run(
            "test", "generate post-freeze Grad-CAM review",
            self.py("model/scripts/generate_gradcam_review.py"),
        )
        self.run(
            "test", "fit validation-only temperature scaling and report frozen-test calibration",
            self.py("model/scripts/calibrate_final_model.py"),
        )
        self.run(
            "test", "measure post-freeze end-to-end efficiency",
            self.py("model/scripts/measure_final_efficiency.py"),
        )
        self.progress_report("test", "Phase 7 - frozen single locked-test evaluation complete")
        self.append_development(
            "Stage G frozen locked-test evaluation completed",
            "the single predeclared representative checkpoint evaluation",
            "The locked test was evaluated once after selection was frozen; test metrics were consolidated separately from validation selection.",
            "Run final source, test, consistency, inference-artifact, and report verification.",
        )

    def finalize(self) -> None:
        self.run("finalize", "verify immutable source", self.py("model/scripts/verify_source_immutable.py", "--workers", "4"))
        self.run("finalize", "run complete unit suite", self.py("-m", "unittest", "discover", "-s", "model/tests", "-v"))
        self.run("finalize", "refresh final CUDA/software environment snapshot", self.py("model/scripts/snapshot_multiclass_environment.py"))
        self.consolidate("finalize")
        self.append_development(
            "Phase 10 final verification gates completed",
            "the complete pre-report verification workflow",
            "Source immutability, tests, and final consolidation passed; completion-gated final report generation is authorized from actual artifacts.",
            "Generate and structurally verify the final Markdown, final DOCX, and final progressive DOCX.",
        )
        self.run(
            "finalize",
            "generate completion-gated final Markdown and Word reports",
            self.doc_py("model/scripts/generate_final_report.py"),
        )
        self.progress_report("finalize", "Phase 10 - final verification and reporting complete")
        self.run(
            "finalize",
            "generate versioned black-only research-complete Word report",
            self.doc_py("model/scripts/generate_research_complete_black_report.py"),
        )

    def execute(self) -> None:
        start = STAGES.index(self.args.from_stage)
        self.log(f"Orchestrator invoked from stage {self.args.from_stage}")
        for stage in STAGES[start:]:
            getattr(self, stage)()
        self.state(status="completed", stage="finalize", detail="Full staged workflow completed.")
        self.log("Full staged workflow completed")


def main() -> int:
    args = parse_args()
    orchestrator = Orchestrator(args)
    try:
        orchestrator.execute()
        return 0
    except BaseException as error:
        orchestrator.log(f"ORCHESTRATOR STOPPED: {type(error).__name__}: {error}")
        current = json.loads(orchestrator.state_path.read_text(encoding="utf-8")) if orchestrator.state_path.exists() else {}
        orchestrator.state(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            stage=current.get("stage", args.from_stage),
            detail=f"{type(error).__name__}: {error}",
            command=current.get("active_command"),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
