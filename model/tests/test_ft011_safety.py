"""Simulated protective policy tests; these never allocate GPU memory."""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))
from multiclass.ft011_safety import GIB, MIB, SafetyPolicy, SafetyWatchdog, evaluate_reading


def policy() -> SafetyPolicy:
    return SafetyPolicy.from_hardware({"gpu_target_c": 87, "gpu_slowdown_c": 97, "gpu_shutdown_c": 100})


def healthy(now=0.0) -> dict:
    return {"monotonic_time": now, "gpu_temperature_c": 50,
            "gpu_utilization_percent": 5, "gpu_vram_used_bytes": GIB,
            "gpu_vram_total_bytes": 4 * GIB, "gpu_thermal_throttling": False,
            "ram_total_bytes": 16 * GIB, "ram_available_bytes": 3 * GIB,
            "disk_free_bytes": 5 * GIB, "ac_connected": True, "errors": []}


class PolicyTests(unittest.TestCase):
    def test_device_anchored_thresholds(self):
        p = policy()
        self.assertEqual((p.gpu_warning_c, p.gpu_pause_c, p.gpu_critical_c, p.gpu_resume_c), (80, 84, 92, 75))
        self.assertEqual(p.disk_reserve_bytes(219 * MIB), 438 * MIB + GIB)
        with self.assertRaises(RuntimeError):
            SafetyPolicy.from_hardware({"gpu_target_c": None, "gpu_slowdown_c": 97, "gpu_shutdown_c": 100})

    def test_all_required_readings_fail_closed(self):
        base = healthy()
        self.assertEqual(evaluate_reading(base, policy(), 219 * MIB, now=0)["action"], "continue")
        for key in base:
            if key == "errors":
                continue
            with self.subTest(key=key):
                reading = {**base, key: None}
                self.assertEqual(evaluate_reading(reading, policy(), 219 * MIB, now=0)["action"], "pause")

    def test_stale_ac_memory_disk_and_throttle_interrupt(self):
        cases = [{"ac_connected": False}, {"ram_available_bytes": 512 * MIB},
                 {"disk_free_bytes": GIB}, {"gpu_vram_used_bytes": 4 * GIB},
                 {"gpu_thermal_throttling": True}]
        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(evaluate_reading({**healthy(), **values}, policy(), 219 * MIB, now=0)["action"], "pause")
        self.assertEqual(evaluate_reading(healthy(), policy(), 219 * MIB, now=7)["action"], "pause")
        self.assertEqual(evaluate_reading({**healthy(), "gpu_temperature_c": 92}, policy(), 219 * MIB, now=0)["action"], "stop")


class WatchdogTests(unittest.TestCase):
    def test_transient_hazard_is_latched_and_smoke_simulation_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            temperatures = iter([85, 50])
            def sampler():
                return {**healthy(time.monotonic()), "gpu_temperature_c": next(temperatures)}
            watchdog = SafetyWatchdog(path, policy(), 219 * MIB, path, sampler=sampler)
            watchdog.sample()
            watchdog.sample()
            self.assertEqual(watchdog.poll()["action"], "pause")
            self.assertEqual(watchdog.status["overheat_events"], 1)
            with self.assertRaises(ValueError):
                watchdog.simulate_once()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            watchdog = SafetyWatchdog(path, policy(), 219 * MIB, path)
            watchdog._reading = healthy(time.monotonic())
            watchdog.simulate_once(after_polls=2, smoke=True)
            self.assertEqual(watchdog.poll()["action"], "continue")
            self.assertEqual(watchdog.poll()["action"], "pause")
            self.assertTrue(list((path / "events").glob("*simulated_smoke_trigger_applied.json")))
            self.assertEqual(watchdog.status["actual_overheat_events"], 0)
            self.assertEqual(watchdog.status["simulated_pause_events"], 1)
            checkpoint = path / "complete_checkpoint.pt"
            checkpoint.write_bytes(b"simulated full checkpoint")
            watchdog.begin_cooling(checkpoint)
            watchdog._reading = healthy(0)
            watchdog.poll_resume(checkpoint_verified=True, now=0)
            watchdog._reading = healthy(120)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=120)["action"], "resume")
            watchdog._reading = {**healthy(121), "gpu_temperature_c": 85}
            self.assertEqual(watchdog.poll(now=121)["action"], "pause")
            self.assertEqual(watchdog.status["actual_overheat_events"], 1)
            self.assertEqual(watchdog.status["simulated_pause_events"], 1)

    def test_two_minutes_continuous_cooling_and_repeat_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            watchdog = SafetyWatchdog(path / "logs", policy(), 219 * MIB, path)
            watchdog._reading = {**healthy(), "gpu_temperature_c": 85}
            self.assertEqual(watchdog.poll(now=0)["action"], "pause")
            checkpoint = path / "complete_checkpoint.pt"
            checkpoint.write_bytes(b"simulated full checkpoint")
            watchdog.begin_cooling(checkpoint)
            watchdog._reading = healthy(0)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=0)["action"], "pause")
            watchdog._reading = healthy(119)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=119)["action"], "pause")
            watchdog._reading = healthy(120)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=120)["action"], "resume")
            watchdog._reading = {**healthy(121), "gpu_temperature_c": 85}
            self.assertEqual(watchdog.poll(now=121)["action"], "stop")
            self.assertEqual(watchdog.status["overheat_events"], 2)

    def test_monitoring_gap_resets_cooling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            watchdog = SafetyWatchdog(path / "logs", policy(), 219 * MIB, path)
            checkpoint = path / "complete.pt"
            checkpoint.write_bytes(b"checkpoint")
            watchdog.begin_cooling(checkpoint)
            watchdog._reading = healthy(0)
            watchdog.poll_resume(checkpoint_verified=True, now=0)
            watchdog._reading = {**healthy(100), "ac_connected": None}
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=100)["cooling_seconds"], 0)
            watchdog._reading = healthy(120)
            watchdog.poll_resume(checkpoint_verified=True, now=120)
            watchdog._reading = healthy(239)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=239)["action"], "pause")
            watchdog._reading = healthy(240)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=240)["action"], "resume")

    def test_checkpoint_verification_and_missing_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            watchdog = SafetyWatchdog(path / "logs", policy(), 219 * MIB, path)
            checkpoint = path / "complete.pt"
            checkpoint.write_bytes(b"checkpoint")
            watchdog.begin_cooling(checkpoint)
            watchdog._reading = healthy(0)
            self.assertEqual(watchdog.poll_resume(now=0)["action"], "pause")
            with self.assertRaises(RuntimeError):
                watchdog.begin_cooling(path / "missing.pt")
            watchdog._reading = healthy(120)
            self.assertEqual(watchdog.poll_resume(checkpoint_verified=True, now=120)["action"], "stop")

    def test_timestamped_atomic_samples_peaks_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            watchdog = SafetyWatchdog(path, policy(), 219 * MIB, path, sampler=lambda: healthy())
            watchdog.sample()
            self.assertTrue((path / "samples/00000001.json").is_file())
            self.assertTrue((path / "telemetry_status.json").is_file())
            self.assertTrue((path / "telemetry.jsonl").is_file())
            self.assertTrue((path / "telemetry_parts/00000001.jsonl").is_file())
            self.assertEqual(watchdog.status["peaks"]["gpu_temperature_c"], 50)
            self.assertFalse(list(path.rglob("*.tmp")))
            second = SafetyWatchdog(path, policy(), 219 * MIB, path, sampler=lambda: healthy(time.monotonic()))
            second.sample()
            self.assertTrue((path / "samples/00000002.json").is_file())
            self.assertEqual(len(list((path / "samples").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
