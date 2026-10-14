"""Fail-closed FT-011 telemetry and protective policy, independent of torch.

Training integration: start SafetyWatchdog, inspect poll() at each safe batch
boundary, atomically save the *complete* trainer state when action is pause or
stop, then begin_cooling(checkpoint_path). poll_resume(checkpoint_verified=True)
permits restoration only after 120 continuous seconds of healthy readings.
The caller must verify/restore model, optimizer, scheduler, scaler, RNG, early
stopping and data-position state; this module never claims a weight-only file
is a resumable checkpoint. A second overheating episode stops the experiment.
No hardware/fan settings are changed. CPU temperature is explicitly unavailable
unless a trustworthy sensor has been configured by the caller.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


MIB = 1024 * 1024
GIB = 1024 * MIB


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: Any) -> None:
    """One-writer atomic replacement; readers never see partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_jsonl_record(path: Path, value: Any) -> None:
    """Atomic one-line JSONL segment; telemetry.jsonl is only a convenience index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def number(value: str | None) -> float | None:
    if not value or value.strip() in {"N/A", "Not Supported", "Unknown"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def query_gpu(gpu_index: int = 0) -> dict[str, Any]:
    """XML avoids version-dependent CSV query-field aliases."""
    result = subprocess.run(
        ["nvidia-smi", "-q", "-x", "-i", str(gpu_index)],
        capture_output=True, text=True, timeout=4, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    gpu = ET.fromstring(result.stdout).find("gpu")
    if gpu is None:
        raise RuntimeError("nvidia-smi returned no selected GPU")
    temperatures = gpu.find("temperature")
    def temperature(*names: str) -> float | None:
        for name in names:
            value = number(temperatures.findtext(name)) if temperatures is not None else None
            if value is not None:
                return value
        return None
    events = gpu.find("clocks_event_reasons")
    if events is None:
        events = gpu.find("clocks_throttle_reasons")
    thermal_values: list[str] = []
    if events is not None:
        thermal_values = [child.text.strip() for child in events
                          if "thermal_slowdown" in child.tag and child.text]
    thermal = None
    if thermal_values and all(value in {"Active", "Not Active"} for value in thermal_values):
        thermal = any(value == "Active" for value in thermal_values)
    total = number(gpu.findtext("fb_memory_usage/total"))
    used = number(gpu.findtext("fb_memory_usage/used"))
    return {
        "gpu_name": gpu.findtext("product_name"),
        "gpu_temperature_c": temperature("gpu_temp"),
        "gpu_target_c": temperature("gpu_target_temperature", "gpu_temp_target"),
        "gpu_slowdown_c": temperature("gpu_temp_slow_threshold"),
        "gpu_shutdown_c": temperature("gpu_temp_max_threshold"),
        "gpu_utilization_percent": number(gpu.findtext("utilization/gpu_util")),
        "gpu_vram_used_bytes": round(used * MIB) if used is not None else None,
        "gpu_vram_total_bytes": round(total * MIB) if total is not None else None,
        "gpu_thermal_throttling": thermal,
        "gpu_thermal_reason_values": thermal_values,
        "gpu_power_watts": number(gpu.findtext("power_readings/power_draw")),
        "gpu_driver_version": ET.fromstring(result.stdout).findtext("driver_version"),
    }


@dataclass(frozen=True)
class SafetyPolicy:
    gpu_warning_c: float
    gpu_pause_c: float
    gpu_critical_c: float
    gpu_resume_c: float
    hardware_target_c: float
    hardware_slowdown_c: float
    hardware_shutdown_c: float
    continuous_cooling_seconds: float = 120.0
    heartbeat_max_age_seconds: float = 6.0
    sample_interval_seconds: float = 2.0
    ram_warning_available_bytes: int = 2 * GIB
    ram_pause_available_bytes: int = GIB
    ram_resume_available_bytes: int = 2 * GIB
    vram_warning_fraction: float = 0.95
    vram_pause_fraction: float = 0.98
    vram_resume_fraction: float = 0.92
    checkpoint_copies_reserved: int = 2
    disk_additional_reserve_bytes: int = GIB

    @classmethod
    def from_hardware(cls, gpu: dict[str, Any]) -> "SafetyPolicy":
        limits = [gpu.get(key) for key in
                  ("gpu_target_c", "gpu_slowdown_c", "gpu_shutdown_c")]
        if any(value is None or not math.isfinite(float(value)) for value in limits):
            raise RuntimeError("Required GPU target/slowdown/shutdown limits unavailable")
        target, slowdown, shutdown = map(float, limits)
        if not 0 < target < slowdown < shutdown:
            raise RuntimeError("GPU thermal limits are inconsistent; cannot derive policy")
        warning = min(target - 7, slowdown - 17, shutdown - 20)
        pause = min(target - 3, slowdown - 13, shutdown - 16)
        critical = min(slowdown - 5, shutdown - 8)
        resume = min(target - 12, warning - 5)
        if not 0 < resume < warning < pause < critical < shutdown:
            raise RuntimeError("Derived GPU thresholds are inconsistent")
        return cls(warning, pause, critical, resume, target, slowdown, shutdown)

    def disk_reserve_bytes(self, checkpoint_bytes: int) -> int:
        if checkpoint_bytes <= 0:
            raise ValueError("Actual/estimated full resumable checkpoint size is required")
        return checkpoint_bytes * self.checkpoint_copies_reserved + self.disk_additional_reserve_bytes


class WindowsSystemSampler:
    """Use native Windows counters; no questionable ACPI CPU temperature proxy."""
    def __init__(self) -> None:
        self.previous_cpu: tuple[int, int, int] | None = None

    def sample(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cpu_temperature_c": None,
            "cpu_temperature_status": "unavailable: no verified package/core temperature sensor",
            "cpu_utilization_percent": None,
            "ram_total_bytes": None, "ram_available_bytes": None,
            "ac_connected": None, "battery_percent": None,
        }
        if os.name != "nt":
            raise RuntimeError("FT-011 requires the audited Windows laptop telemetry")
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
        memory = Memory()
        memory.length = ctypes.sizeof(Memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("GlobalMemoryStatusEx failed")
        data.update(ram_total_bytes=int(memory.total_phys), ram_available_bytes=int(memory.avail_phys))
        class Power(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                        ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                        ("life", wintypes.DWORD), ("full", wintypes.DWORD)]
        power = Power()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
            raise OSError("GetSystemPowerStatus failed")
        data["ac_connected"] = bool(power.ac) if power.ac in (0, 1) else None
        data["battery_percent"] = int(power.percent) if power.percent <= 100 else None
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            def ticks(value: Any) -> int:
                return (value.dwHighDateTime << 32) | value.dwLowDateTime
            current = (ticks(idle), ticks(kernel), ticks(user))
            if self.previous_cpu is not None:
                deltas = [a - b for a, b in zip(current, self.previous_cpu)]
                total = deltas[1] + deltas[2]
                if total > 0:
                    data["cpu_utilization_percent"] = max(0.0, min(100.0, 100 * (total - deltas[0]) / total))
            self.previous_cpu = current
        return data


class HardwareSampler:
    def __init__(self, disk_path: Path | str, gpu_index: int = 0) -> None:
        self.disk_path = Path(disk_path)
        self.gpu_index = gpu_index
        self.system = WindowsSystemSampler()

    def __call__(self) -> dict[str, Any]:
        reading: dict[str, Any] = {"timestamp_utc": utc_now(), "monotonic_time": time.monotonic(), "errors": []}
        try:
            reading.update(query_gpu(self.gpu_index))
        except Exception as error:
            reading["errors"].append(f"GPU telemetry: {type(error).__name__}: {error}")
        try:
            reading.update(self.system.sample())
        except Exception as error:
            reading["errors"].append(f"System telemetry: {type(error).__name__}: {error}")
        try:
            disk = shutil.disk_usage(self.disk_path)
            reading.update(disk_free_bytes=disk.free, disk_total_bytes=disk.total)
        except Exception as error:
            reading["errors"].append(f"Disk telemetry: {type(error).__name__}: {error}")
        # Timestamp completion, so a slow/hung required query cannot hide its age.
        reading["sample_completed_utc"] = utc_now()
        return reading


def evaluate_reading(reading: dict[str, Any], policy: SafetyPolicy,
                     checkpoint_bytes: int, now: float | None = None,
                     resume: bool = False) -> dict[str, Any]:
    """Pure policy function for simulation and safe-boundary enforcement."""
    now = time.monotonic() if now is None else now
    reading = dict(reading)
    reasons: list[str] = []
    warnings: list[str] = []
    required = ("monotonic_time", "gpu_temperature_c", "gpu_utilization_percent",
                "gpu_vram_used_bytes", "gpu_vram_total_bytes", "gpu_thermal_throttling",
                "ram_total_bytes", "ram_available_bytes", "disk_free_bytes", "ac_connected")
    for key in required:
        value = reading.get(key)
        valid = (isinstance(value, bool) if key in {"gpu_thermal_throttling", "ac_connected"}
                 else isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
        if not valid:
            reasons.append(f"Required monitoring unavailable: {key}")
            reading[key] = None
    if reading.get("errors"):
        reasons.extend(reading["errors"])
    sampled = reading.get("monotonic_time")
    age = now - sampled if isinstance(sampled, (int, float)) and math.isfinite(sampled) else 0.0
    if age > policy.heartbeat_max_age_seconds or age < -1:
        reasons.append(f"Monitoring heartbeat stale/invalid: {age:.1f}s")
    if reading.get("ac_connected") is False:
        reasons.append("AC power disconnected")
    temperature = reading.get("gpu_temperature_c")
    utilization = reading.get("gpu_utilization_percent")
    if utilization is not None and not 0 <= utilization <= 100:
        reasons.append("GPU utilization measurement invalid")
    critical = temperature is not None and temperature >= policy.gpu_critical_c
    if temperature is not None:
        threshold = policy.gpu_resume_c if resume else policy.gpu_pause_c
        if temperature >= threshold:
            reasons.append(f"GPU temperature {temperature:.1f}C >= {'resume' if resume else 'pause'} threshold {threshold:.1f}C")
        elif temperature >= policy.gpu_warning_c:
            warnings.append(f"GPU temperature warning: {temperature:.1f}C")
    if reading.get("gpu_thermal_throttling") is True:
        reasons.append("GPU thermal throttling active")
    available = reading.get("ram_available_bytes")
    ram_total = reading.get("ram_total_bytes")
    if ram_total is not None and (ram_total <= 0 or (available is not None and not 0 <= available <= ram_total)):
        reasons.append("System RAM measurement invalid")
    if available is not None:
        threshold = policy.ram_resume_available_bytes if resume else policy.ram_pause_available_bytes
        if available < threshold:
            reasons.append(f"System RAM available below {'resume' if resume else 'pause'} budget")
        elif available < policy.ram_warning_available_bytes:
            warnings.append("System RAM available below warning budget")
    total, used = reading.get("gpu_vram_total_bytes"), reading.get("gpu_vram_used_bytes")
    if total is not None and used is not None:
        if total <= 0 or used < 0 or used > total:
            reasons.append("GPU VRAM measurement invalid")
        else:
            fraction = used / total
            threshold = policy.vram_resume_fraction if resume else policy.vram_pause_fraction
            if fraction >= threshold:
                reasons.append(f"GPU VRAM pressure: {fraction:.1%}")
            elif fraction >= policy.vram_warning_fraction:
                warnings.append(f"GPU VRAM warning: {fraction:.1%}")
    free = reading.get("disk_free_bytes")
    reserve = policy.disk_reserve_bytes(checkpoint_bytes)
    if free is not None and free < reserve:
        reasons.append(f"Checkpoint disk reserve insufficient: {free} < {reserve} bytes")
    return {"action": "stop" if critical else "pause" if reasons else "continue",
            "reasons": reasons, "warnings": warnings, "critical_overheat": critical,
            "overheat": bool(temperature is not None and temperature >= policy.gpu_pause_c)
                        or reading.get("gpu_thermal_throttling") is True,
            "heartbeat_age_seconds": age, "disk_reserve_bytes": reserve}


class SafetyWatchdog:
    """Two-second background sampler plus nonblocking safe-boundary API."""
    def __init__(self, log_dir: Path | str, policy: SafetyPolicy,
                 checkpoint_bytes: int, disk_path: Path | str, gpu_index: int = 0,
                 sampler: Callable[[], dict[str, Any]] | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.policy = policy
        self.checkpoint_bytes = checkpoint_bytes
        policy.disk_reserve_bytes(checkpoint_bytes)
        self.sampler = sampler or HardwareSampler(disk_path, gpu_index)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reading: dict[str, Any] = {}
        self._peaks: dict[str, float] = {}
        self._pause_active = False
        self._terminal = False
        self._overheat_events = 0
        self._cooling_start: float | None = None
        self._checkpoint: Path | None = None
        self._event_sequence = 0
        self._sample_sequence = 0
        self._monitoring_gaps = 0
        self._throttle_samples = 0
        self._pause_count = 0
        self._simulated_pause_events = 0
        self._progress: dict[str, Any] = {}
        self._pause_reason: dict[str, Any] | None = None
        self._pause_was_overheat = False
        self._simulation: dict[str, Any] | None = None
        self._poll_count = 0
        # Resume a log stream without replacing earlier timestamped records.
        for directory, attribute in (("samples", "_sample_sequence"), ("events", "_event_sequence")):
            numbers = [int(path.stem.split("_")[0]) for path in (self.log_dir / directory).glob("*.json")
                       if path.stem.split("_")[0].isdigit()]
            if numbers:
                setattr(self, attribute, max(numbers))
        previous_status = self.log_dir / "telemetry_status.json"
        if previous_status.is_file():
            try:
                previous = json.loads(previous_status.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise RuntimeError("Existing watchdog history is unreadable; preserve and inspect it") from error
            if previous.get("policy") != asdict(policy):
                raise RuntimeError("Existing watchdog policy mismatch; do not overwrite prior logs")
            self._peaks = dict(previous.get("peaks", {}))
            self._overheat_events = int(previous.get("overheat_events", 0))
            self._monitoring_gaps = int(previous.get("monitoring_gaps", 0))
            self._throttle_samples = int(previous.get("thermal_throttle_samples", 0))
            self._pause_count = int(previous.get("cooling_pauses", 0))
            self._simulated_pause_events = int(previous.get("simulated_pause_events", 0))
            self._terminal = bool(previous.get("terminal_stop", False))

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"timestamp_utc": utc_now(), "sample_sequence": self._sample_sequence,
                    "reading": dict(self._reading), "policy": asdict(self.policy),
                    "checkpoint_bytes": self.checkpoint_bytes,
                    "disk_reserve_bytes": self.policy.disk_reserve_bytes(self.checkpoint_bytes),
                    "peaks": dict(self._peaks), "training": dict(self._progress),
                    "overheat_events": self._overheat_events, "monitoring_gaps": self._monitoring_gaps,
                    "actual_overheat_events": self._overheat_events,
                    "simulated_pause_events": self._simulated_pause_events,
                    "cooling_pauses": self._pause_count,
                    "thermal_throttle_samples": self._throttle_samples,
                    "pause_active": self._pause_active, "terminal_stop": self._terminal,
                    "cooling_started_monotonic": self._cooling_start,
                    "complete_checkpoint_path": str(self._checkpoint) if self._checkpoint else None}

    def update_progress(self, **values: Any) -> None:
        with self._lock:
            self._progress.update(values)

    def _event(self, event: str, **values: Any) -> None:
        self._event_sequence += 1
        record = {"timestamp_utc": utc_now(), "event": event, **values}
        atomic_json(self.log_dir / "events" / f"{self._event_sequence:06d}_{event}.json", record)

    def _latch_decision(self, decision: dict[str, Any]) -> None:
        if decision["action"] == "continue":
            return
        new_episode = not self._pause_active
        simulated = bool(decision.get("simulated", False))
        new_actual_overheat = decision["overheat"] and not simulated and not self._pause_was_overheat
        self._pause_active = True
        if new_episode:
            self._pause_count += 1
            if simulated:
                self._simulated_pause_events += 1
        if new_actual_overheat:
            self._overheat_events += 1
            self._pause_was_overheat = True
        self._terminal = self._terminal or (decision["critical_overheat"] and not simulated) or self._overheat_events >= 2
        decision["action"] = "stop" if self._terminal else "pause"
        if self._overheat_events >= 2:
            decision["reasons"].append("Repeated overheating: second event stops FT-011")
        self._pause_reason = decision
        if new_episode or new_actual_overheat or self._terminal:
            self._event("protective_stop" if self._terminal else "protective_pause", decision=decision)

    def simulate_once(self, reason: str = "gpu_temperature", after_polls: int = 4,
                      smoke: bool = False) -> None:
        """Smoke only: inject one labelled policy reading without touching hardware.

        The real checkpoint and 120-second cooling path still execute. Never
        call this for a full candidate; the explicit smoke flag is mandatory.
        """
        if not smoke:
            raise ValueError("Safety simulation is permitted only for smoke=True")
        if reason not in {"gpu_temperature", "ac_power", "monitoring_failure", "disk"}:
            raise ValueError("Unknown simulated safety reason")
        if after_polls < 1:
            raise ValueError("after_polls must be positive")
        with self._lock:
            if self._simulation is not None:
                raise RuntimeError("A smoke simulation is already scheduled")
            self._simulation = {"reason": reason, "trigger_poll": self._poll_count + after_polls}
            self._event("simulated_smoke_trigger_scheduled", **self._simulation)

    def sample(self) -> dict[str, Any]:
        with self._lock:
            previous_time = self._reading.get("monotonic_time")
            try:
                reading = self.sampler()
            except Exception as error:
                reading = {"timestamp_utc": utc_now(), "monotonic_time": time.monotonic(),
                           "errors": [f"Sampler failed: {type(error).__name__}: {error}"]}
            self._reading = reading
            self._sample_sequence += 1
            decision = evaluate_reading(reading, self.policy, self.checkpoint_bytes)
            self._latch_decision(decision)
            if self._checkpoint is not None:
                resume_decision = evaluate_reading(reading, self.policy, self.checkpoint_bytes, resume=True)
                if resume_decision["action"] != "continue" or (previous_time is not None and
                        reading.get("monotonic_time", 0) - previous_time > self.policy.heartbeat_max_age_seconds):
                    self._cooling_start = None
            if any("unavailable" in reason.lower() or "telemetry" in reason.lower()
                   or "heartbeat" in reason.lower() or "sampler" in reason.lower() for reason in decision["reasons"]):
                self._monitoring_gaps += 1
            if reading.get("gpu_thermal_throttling") is True:
                self._throttle_samples += 1
            for key in ("gpu_temperature_c", "gpu_utilization_percent", "gpu_vram_used_bytes", "cpu_utilization_percent", "cpu_temperature_c"):
                value = reading.get(key)
                if value is not None:
                    self._peaks[key] = max(self._peaks.get(key, value), value)
            if reading.get("ram_total_bytes") is not None and reading.get("ram_available_bytes") is not None:
                used = reading["ram_total_bytes"] - reading["ram_available_bytes"]
                self._peaks["ram_used_bytes"] = max(self._peaks.get("ram_used_bytes", used), used)
            # Each timestamped sample is atomic; append-only JSONL is an index.
            sample_path = self.log_dir / "samples" / f"{self._sample_sequence:08d}.json"
            record = {**reading, "decision": decision, "training": dict(self._progress)}
            atomic_json(sample_path, record)
            atomic_jsonl_record(self.log_dir / "telemetry_parts" / f"{self._sample_sequence:08d}.jsonl", record)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(self.log_dir / "telemetry_status.json", self.status)
            return record

    def start(self) -> "SafetyWatchdog":
        if self._thread is not None:
            raise RuntimeError("Watchdog already started")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.log_dir / "safety_policy.json", {
            "policy": asdict(self.policy), "checkpoint_bytes": self.checkpoint_bytes,
            "disk_reserve_bytes": self.policy.disk_reserve_bytes(self.checkpoint_bytes),
            "derivation": "Warning target-7C; pause target-3C; critical min(slowdown-5C,shutdown-8C); resume target-12C. Conservative project margins, not manufacturer temperature guarantees.",
            "cpu_temperature": "Unavailable is explicit and optional; GPU thermal telemetry is required.",
        })
        self.sample()
        startup = self.poll()
        if startup["action"] != "continue":
            raise RuntimeError("Required safety preflight failed: " + "; ".join(startup["reasons"]))
        self._event("watchdog_started")
        def loop() -> None:
            while not self._stop.wait(self.policy.sample_interval_seconds):
                try:
                    self.sample()
                except Exception as error:
                    with self._lock:
                        self._reading = {"monotonic_time": time.monotonic(),
                                         "errors": [f"Monitoring log/sample failure: {error}"]}
        self._thread = threading.Thread(target=loop, name="FT011SafetySampler", daemon=True)
        self._thread.start()
        return self

    def poll(self, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            self._poll_count += 1
            reading = self._reading
            simulated = False
            if self._simulation is not None and self._poll_count >= self._simulation["trigger_poll"]:
                simulated = True
                reading = dict(reading)
                reason = self._simulation["reason"]
                if reason == "gpu_temperature":
                    reading["gpu_temperature_c"] = self.policy.gpu_pause_c
                elif reason == "ac_power":
                    reading["ac_connected"] = False
                elif reason == "monitoring_failure":
                    reading["gpu_temperature_c"] = None
                else:
                    reading["disk_free_bytes"] = self.policy.disk_reserve_bytes(self.checkpoint_bytes) - 1
                self._event("simulated_smoke_trigger_applied", reason=reason, simulated_reading=reading)
                self._simulation = None
            decision = evaluate_reading(reading, self.policy, self.checkpoint_bytes, now)
            if simulated:
                decision["simulated"] = True
            self._latch_decision(decision)
            if self._pause_active:
                decision = dict(self._pause_reason or decision)
                decision["action"] = "stop" if self._terminal else "pause"
            return decision

    def poll_should_pause(self) -> bool:
        return self.poll()["action"] != "continue"

    def begin_cooling(self, checkpoint_path: Path | str) -> None:
        with self._lock:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                self._terminal = True
                raise RuntimeError("Protective pause requires a saved complete atomic checkpoint")
            self._checkpoint = checkpoint
            self._cooling_start = None
            self._pause_active = True
            self._event("checkpoint_saved_cooling", checkpoint_path=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)

    def poll_resume(self, checkpoint_verified: bool = False,
                    now: float | None = None, checkpoint_restored: bool | None = None) -> dict[str, Any]:
        """Nonblocking: caller sleeps briefly, verifies full state, then restores."""
        with self._lock:
            now = time.monotonic() if now is None else now
            if checkpoint_restored is not None:
                checkpoint_verified = checkpoint_restored
            decision = evaluate_reading(self._reading, self.policy, self.checkpoint_bytes, now, resume=True)
            if self._terminal or decision["critical_overheat"]:
                self._terminal = True
                return {**decision, "action": "stop", "cooling_seconds": 0.0}
            if not self._checkpoint or not self._checkpoint.is_file() or not checkpoint_verified:
                self._cooling_start = None
                return {**decision, "action": "pause", "cooling_seconds": 0.0,
                        "reasons": decision["reasons"] + ["Complete resumable checkpoint not verified"]}
            if decision["action"] != "continue":
                self._cooling_start = None
                return {**decision, "action": "pause", "cooling_seconds": 0.0}
            if self._cooling_start is None:
                self._cooling_start = now
            duration = now - self._cooling_start
            if duration < self.policy.continuous_cooling_seconds:
                return {**decision, "action": "pause", "cooling_seconds": duration}
            self._pause_active = False
            self._pause_was_overheat = False
            self._pause_reason = None
            self._cooling_start = None
            self._event("safe_resume_allowed", cooling_seconds=duration)
            return {**decision, "action": "resume", "cooling_seconds": duration}

    def wait_until_resume(self, checkpoint_verified: bool = False) -> dict[str, Any]:
        """Convenience wrapper; trainer can prefer nonblocking poll_resume."""
        while True:
            decision = self.poll_resume(checkpoint_verified=checkpoint_verified)
            if decision["action"] in {"resume", "stop"}:
                return decision
            self._stop.wait(self.policy.sample_interval_seconds)
            if self._stop.is_set():
                return {"action": "stop", "reasons": ["Watchdog closed while cooling"]}

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        with self._lock:
            self._event("watchdog_closed")
            atomic_json(self.log_dir / "telemetry_status.json", self.status)

