from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import optuna


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "model/scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OrchestrationTests(unittest.TestCase):
    def test_optuna_target_counts_enqueued_control_as_unexecuted(self):
        module = load_script("run_tuning")
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"x": 0.5})

        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state.name, "WAITING")
        self.assertEqual(module.remaining_optimization_trials(study, 4), 4)

        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)
        self.assertEqual(module.remaining_optimization_trials(study, 4), 1)

    def test_optuna_export_retains_completed_and_failed_trials(self):
        module = load_script("run_tuning")
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
        )
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

        def fail(_trial):
            raise RuntimeError("expected contract-test failure")

        study.optimize(fail, n_trials=1, catch=(RuntimeError,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = module.export_trials(
                study, study_id="contract", parent_run_id="parent", output_dir=output
            )
            self.assertEqual([row["status"] for row in rows], ["completed", "failed"])
            self.assertTrue((output / "trials.csv").exists())
            self.assertTrue((output / "trials.json").exists())

    def test_stage_resolution_requires_all_predecessors(self):
        module = load_script("resolve_run_matrix")
        with self.assertRaisesRegex(RuntimeError, "requires 8 completed rows; found 1"):
            module.require_completed([{"status": "completed"}], 8, "baseline gate")


if __name__ == "__main__":
    unittest.main()
