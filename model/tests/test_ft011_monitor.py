"""Display-only FT-011 monitor contracts; no ML or GPU imports are required."""
from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "model/scripts/monitor_ft011.py"
SPEC = importlib.util.spec_from_file_location("ft011_monitor_test_module", SCRIPT)
MONITOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MONITOR
SPEC.loader.exec_module(MONITOR)

NOW = datetime(2026, 10, 13, 0, 0, 0, tzinfo=timezone.utc)


def telemetry(age=0, pause=False, stopped=False):
    return {
        "reading": {
            "sample_completed_utc": (NOW - timedelta(seconds=age)).isoformat(),
            "gpu_name": "NVIDIA GeForce RTX 2050",
            "gpu_temperature_c": 62, "gpu_utilization_percent": 80,
            "gpu_vram_used_bytes": 1024**3, "gpu_vram_total_bytes": 4 * 1024**3,
            "cpu_utilization_percent": 23, "cpu_temperature_c": None,
            "cpu_temperature_status": "unavailable: no reliable package sensor",
            "ram_available_bytes": 3 * 1024**3, "ram_total_bytes": 16 * 1024**3,
            "disk_free_bytes": 10 * 1024**3, "ac_connected": True,
            "battery_percent": 40, "gpu_thermal_throttling": False,
            "errors": [],
        },
        "policy": {"heartbeat_max_age_seconds": 6, "gpu_warning_c": 80,
                   "gpu_pause_c": 84, "gpu_critical_c": 92, "gpu_resume_c": 75,
                   "cooldown_seconds": 120},
        "pause_active": pause, "terminal_stop": stopped,
        "cooling_pauses": int(pause), "monitoring_gaps": 0,
        "disk_reserve_bytes": 1536 * 1024**2,
        "training": {"seed": 42, "epoch": 2, "batch": 100,
                     "total_batches": 510, "phase": "running:train",
                     "backbone_lr": 1e-5, "classifier_lr": 1e-4},
        "peaks": {"gpu_temperature_c": 64},
    }


def snapshot():
    base = {
        "architecture": "densenet201", "weights": "imagenet",
        "class_names": ["Normal", "Viral", "Bacterial"],
        "batch_size": 8, "epochs": 12, "input_size": 224,
        "optimizer": "adamw", "weight_decay": 1e-4,
        "learning_rate": 1e-4, "loss": "cross_entropy",
        "label_smoothing": 0.05, "class_weighting": "balanced",
        "early_stopping_patience": 4, "minimum_delta": 1e-4,
        "amp": True, "num_workers": 0, "freeze_backbone": False,
        "preprocessing": "pseudo_color_then_histogram_equalization_then_clahe",
        "augmentation": {"horizontal_flip_probability": 0.5, "rotation_degrees": 5},
    }
    return {
        "protocol": {"experiment": "FT-011", "protocol_hash": "a" * 64,
                     "control_configuration": base,
                     "learning_rate_policy": {"backbone": 1e-5, "classifier": 1e-4},
                     "seeds": [42, 123, 2026],
                     "scheduler_defaults": {"mode": "max", "factor": 0.5, "patience": 2},
                     "checkpoint_rule": "strictly greater macro F1 than best + 0.0001",
                     "controls": {"screening_gates": [{"metric": "macro_f1", "operator": ">=", "threshold": 0.8613107953930739}]}},
        "queue": {"status": "training", "seed": 42, "updated_at": NOW.isoformat()},
        "run": {"status": "running", "seed": 42, "run_id": "ft011_densenet201_differential_lr_s42",
                "epoch": 2, "phase": "train", "batch": 100, "total_batches": 510,
                "best_epoch": 1, "best_validation_macro_f1": 0.82,
                "elapsed_seconds": 900, "updated_at": NOW.isoformat(),
                "learning_rates": {"backbone": 1e-5, "classifier": 1e-4}},
        "history": [{"epoch": 1, "train_loss": 0.6, "val_loss": 0.4,
                     "val_accuracy": 0.83, "val_balanced_accuracy": 0.8,
                     "val_macro_f1": 0.82, "recall_normal": 0.98,
                     "recall_viral": 0.6, "recall_bacterial": 0.82,
                     "backbone_lr_used": 1e-5, "classifier_lr_used": 1e-4,
                     "elapsed_seconds": 600, "best_epoch": 1,
                     "best_validation_macro_f1": 0.82}],
        "metrics": {}, "telemetry": telemetry(),
        "config": {"base_training_configuration": base,
                   "learning_rate_policy": {"backbone": 1e-5, "classifier": 1e-4}},
        "run_dir": str(REPO_ROOT / "model/results/ft011/runs/ft011_densenet201_differential_lr_s42"),
        "seed_statuses": {"42": {"status": "running"}, "123": {}, "2026": {}},
        "weights": {}, "initialization": {}, "metadata": {},
        "split_summary": {"split_total_counts": {"train": 4076, "val": 874}},
        "smoke": {"status": "passed"}, "decision": {},
    }


class FormattingTests(unittest.TestCase):
    def test_missing_percent_and_duration_are_explicit(self):
        self.assertIn("unavailable", MONITOR.pct(None).lower())
        self.assertIn("unavailable", MONITOR.duration(None).lower())

    def test_probability_is_displayed_as_percentage(self):
        result = MONITOR.pct(0.8563107953930739)
        self.assertIn("85.631", result)
        self.assertIn("%", result)

    def test_progress_bar_reports_actual_fraction_and_handles_unknown_total(self):
        result = MONITOR.progress_bar(50, 100, width=28)
        self.assertRegex(result, r"50(?:\.0+)?%")
        unknown = MONITOR.progress_bar(0, 0, width=28)
        self.assertIsInstance(unknown, str)
        self.assertNotIn("nan", unknown.lower())

    def test_metric_value_reads_saved_per_class_and_preserves_missing_values(self):
        values = {"macro_f1": 0.85, "per_class": {"Viral": {"recall": 0.66}}}
        self.assertEqual(MONITOR.metric_value(values, "macro_f1"), 0.85)
        self.assertEqual(MONITOR.metric_value(values, "viral_recall"), 0.66)
        self.assertIsNone(MONITOR.metric_value(values, "normal_recall"))


class StatusPriorityTests(unittest.TestCase):
    def test_final_queue_decision_has_priority_over_old_watchdog_state(self):
        queue = {"status": "terminal_decision", "decision": "retain FT-004"}
        result = MONITOR.display_status({"status": "running"}, telemetry(age=120, stopped=True), queue, now=NOW)
        self.assertIn("retain FT-004", result)

    def test_protective_terminal_stop_has_priority_over_running(self):
        result = MONITOR.display_status({"status": "running"}, telemetry(stopped=True), {"status": "training"}, now=NOW)
        self.assertIn("STOP", result.upper())

    def test_interrupted_run_is_identified(self):
        result = MONITOR.display_status({"status": "interrupted"}, telemetry(), {"status": "training"}, now=NOW)
        self.assertIn("INTERRUPT", result.upper())

    def test_completed_run_is_not_labeled_running(self):
        result = MONITOR.display_status({"status": "completed"}, telemetry(), {"status": "training"}, now=NOW)
        self.assertIn("COMPLETE", result.upper())

    def test_stale_heartbeat_does_not_show_normal_training(self):
        result = MONITOR.display_status({"status": "running"}, telemetry(age=20), {"status": "training"}, now=NOW)
        self.assertIn("STAL", result.upper())

    def test_active_cooling_pause_is_visible(self):
        result = MONITOR.display_status({"status": "running"}, telemetry(pause=True), {"status": "training"}, now=NOW)
        self.assertIn("PAUS", result.upper())


class DashboardTests(unittest.TestCase):
    def test_tied_f1_history_reports_recorded_best_epoch_not_latest_tie(self):
        state = snapshot()
        state["metrics"] = {"macro_f1": 0.82}
        state["history"] = [
            {"epoch": 1, "val_macro_f1": 0.8, "best_epoch": 1},
            {"epoch": 2, "val_macro_f1": 0.82, "best_epoch": 2},
            {"epoch": 3, "val_macro_f1": 0.82, "best_epoch": 2},
            {"epoch": 4, "val_macro_f1": 0.82, "best_epoch": 2},
        ]
        metrics, best_epoch = MONITOR.verified_metrics(state)
        self.assertEqual(metrics["macro_f1"], 0.82)
        self.assertEqual(best_epoch, 2)
        self.assertNotEqual(best_epoch, state["history"][-1]["epoch"])

    def test_completed_overview_does_not_label_checkpoint_result_interim(self):
        state = snapshot()
        state["run"].update(status="completed", phase="completed", best_epoch=1)
        state["metrics"] = {"macro_f1": 0.82, "best_epoch": 1}
        state["seed_statuses"]["42"] = {"status": "completed"}
        result = MONITOR.render_dashboard(state, view="overview")
        self.assertIn("run complete", result.lower())
        self.assertNotIn("INTERIM", result.upper())
        self.assertIn("controller", result.lower())

    def test_every_view_renders_missing_or_partial_files_gracefully(self):
        for view in ("overview", "config", "results", "hardware", "files", "all"):
            with self.subTest(view=view):
                result = MONITOR.render_dashboard({}, view=view)
                self.assertIsInstance(result, str)
                self.assertTrue(result.strip())
                self.assertNotIn("Traceback", result)

    def test_full_dashboard_names_preprocessing_and_both_learning_rates(self):
        result = MONITOR.render_dashboard(snapshot(), view="all")
        upper = result.upper()
        for name in ("FT-011", "DENSENET", "VIRIDIS", "GLOBAL HISTOGRAM EQUALIZATION", "CLAHE", "BACKBONE", "CLASSIFIER"):
            self.assertIn(name, upper)
        self.assertTrue("0.000010" in result or "0.00001" in result or "1e-05" in result)
        self.assertTrue("0.000100" in result or "0.0001" in result or "1e-04" in result)

    def test_confirmation_queue_is_described_as_conditional(self):
        result = MONITOR.render_dashboard(snapshot(), view="overview")
        self.assertIn("123", result)
        self.assertIn("2026", result)
        self.assertRegex(result.lower(), r"conditional|if .*gate|only .*pass")

    def test_missing_cpu_temperature_is_explicit_in_hardware_view(self):
        result = MONITOR.render_dashboard(snapshot(), view="hardware")
        self.assertIn("unavailable", result.lower())
        self.assertIn("CPU", result.upper())

    def test_display_script_has_no_machine_learning_imports(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertFalse(names & {"torch", "torchvision", "numpy", "multiclass", "pynvml", "psutil"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
