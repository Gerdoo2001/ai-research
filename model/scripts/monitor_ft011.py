"""Read-only, paged FT-011 console dashboard; safety remains in the trainer.

Standard library only: no PyTorch, image/checkpoint loading or training control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

MANILA = timezone(timedelta(hours=8))
VIEWS = ("overview", "config", "results", "hardware", "files")
LABELS = {"macro_f1": "Macro F1", "balanced_accuracy": "Balanced accuracy",
          "accuracy": "Accuracy", "normal_recall": "Normal recall",
          "viral_recall": "Viral recall", "bacterial_recall": "Bacterial recall"}
WIDTH = 108


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def read_history(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [{key: float(value) for key, value in row.items() if value not in (None, "")}
                for row in rows]
    except (OSError, ValueError, TypeError):
        return []


def number(value, digits=4):
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def pct(value):
    return "unavailable" if value is None else f"{float(value) * 100:.3f}%"


def gib(value):
    return "unavailable" if value is None else f"{float(value) / 1024 ** 3:.2f} GiB"


def duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "unavailable"
    seconds = max(0, int(float(seconds)))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def age(timestamp, now=None):
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            return None
        return max(0.0, ((now or datetime.now(timezone.utc)) - parsed).total_seconds())
    except (TypeError, ValueError):
        return None


def local_time(timestamp):
    try:
        return datetime.fromisoformat(timestamp).astimezone(MANILA).strftime("%Y-%m-%d %H:%M:%S Manila")
    except (TypeError, ValueError):
        return "unavailable"


def progress_bar(batch, total, width=28):
    total, batch = max(0, int(total or 0)), max(0, int(batch or 0))
    fraction = min(1.0, batch / total) if total else 0.0
    filled = int(width * fraction)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:6.2f}% ({batch}/{total})"


def metric_value(metrics, key):
    if key in metrics:
        return metrics[key]
    if key.endswith("_recall"):
        return metrics.get("per_class", {}).get(key[:-7].capitalize(), {}).get("recall")
    return None


def display_status(run, telemetry, queue, now=None):
    if queue.get("status") == "terminal_decision":
        return "PHASE CLOSED: " + queue.get("decision", "decision saved")
    status = run.get("status", "")
    if telemetry.get("terminal_stop") or status == "protective_stopped":
        return "PROTECTIVE STOP - saved state; review required"
    if status == "interrupted":
        return "INTERRUPTED - saved state; not running"
    if status == "completed":
        return "RUN COMPLETE - controller analysis/selection pending"
    reading = telemetry.get("reading", {})
    seconds = age(reading.get("sample_completed_utc", reading.get("timestamp_utc")), now)
    if seconds is None:
        return "MONITORING UNAVAILABLE - last-known status: " + (status or "waiting")
    if seconds > telemetry.get("policy", {}).get("heartbeat_max_age_seconds", 6):
        return "MONITORING STALE - last-known status: " + (status or "waiting")
    if telemetry.get("pause_active") or status in ("cooling", "protective_pause"):
        return "PROTECTIVE PAUSE - cooling/resource recovery"
    return "RUNNING" if status == "running" else (status or queue.get("status", "waiting")).upper()


def load_snapshot(log_dir, progress_path=None):
    root = (Path(progress_path).parent if progress_path else Path(log_dir).parent).resolve()
    protocol, queue = read_json(root / "protocol.json"), read_json(progress_path) if progress_path else {}
    run_dir = Path(queue["run_dir"]) if queue.get("run_dir") else None
    statuses, seed_metrics = {}, {}
    for seed in protocol.get("seeds", [42, 123, 2026]):
        candidate = read_json(root / "configs" / f"seed{seed}.json")
        run_id = candidate.get("base_training_configuration", {}).get("run_id",
                      f"ft011_densenet201_differential_lr_s{seed}")
        candidate_dir = root / "runs" / run_id
        statuses[str(seed)] = read_json(candidate_dir / "status.json")
        seed_metrics[str(seed)] = read_json(candidate_dir / "metrics.json")
        if run_dir is None and statuses[str(seed)]:
            run_dir = candidate_dir
    if run_dir is None:
        run_dir = root / "smoke"
    run = read_json(run_dir / "status.json")
    seed = run.get("seed", queue.get("seed", 42))
    candidate = read_json(root / "configs" / f"seed{seed}.json")
    repo = Path(__file__).resolve().parents[2]
    return {"root": str(root), "run_dir": str(run_dir), "protocol": protocol, "queue": queue,
        "run": run, "history": read_history(run_dir / "history.csv"),
        "metrics": read_json(run_dir / "metrics.json"),
        "telemetry": read_json(Path(log_dir) / "telemetry_status.json"),
        "config": candidate.get("base_training_configuration", protocol.get("control_configuration", {})),
        "candidate": candidate, "seed_statuses": statuses, "seed_metrics": seed_metrics,
        "weights": read_json(run_dir / "class_weights.json"),
        "initialization": read_json(run_dir / "initialization.json"),
        "metadata": read_json(run_dir / "model_metadata.json"),
        "split_summary": read_json(repo / "model/data_manifests/split_summary.json"),
        "smoke": read_json(root / "smoke_verified.json"),
        "decision": read_json(root / "final_decision.json"),
        "interruption": read_json(run_dir / "interruption.json")}


def queue_lines(snapshot):
    decision = snapshot.get("decision", {})
    queue = snapshot.get("queue", {})
    lines = []
    for seed in snapshot.get("protocol", {}).get("seeds", [42, 123, 2026]):
        status = snapshot.get("seed_statuses", {}).get(str(seed), {}).get("status")
        if status:
            label = status.upper()
        elif queue.get("status") == "terminal_decision" or decision.get("terminal_decision_complete"):
            label = "NOT RUN / phase closed"
        elif seed == 42:
            label = "PENDING execution checks"
        else:
            label = "CONDITIONAL - only after ALL 5 seed-42 gates pass"
        lines.append(f"Seed {seed:<5}: {label}")
    return lines


def verified_metrics(snapshot):
    metrics, history = snapshot.get("metrics", {}), snapshot.get("history", [])
    if not metrics:
        return {}, None
    matches = [row for row in history if row.get("val_macro_f1") == metrics.get("macro_f1")]
    best_epoch = metrics.get("best_epoch") or (
        int(matches[-1].get("best_epoch", matches[0]["epoch"])) if matches else None)
    if history and not matches and snapshot.get("run", {}).get("status") != "completed":
        return {}, None
    return metrics, best_epoch


def header(snapshot, view):
    run, telemetry, queue = (snapshot.get(k, {}) for k in ("run", "telemetry", "queue"))
    reading = telemetry.get("reading", {})
    sample_age = age(reading.get("sample_completed_utc", reading.get("timestamp_utc")))
    rates = run.get("learning_rates", {})
    return ["FT-011 LIVE MONITOR | " + datetime.now(MANILA).strftime("%Y-%m-%d %H:%M:%S Manila"),
        "=" * WIDTH,
        "1 Progress | 2 Full configuration | 3 Results/gates | 4 Hardware/safety | 5 Files/hashes | A All",
        "VIEW: " + view.upper() + " | Refresh ~2 seconds | Display only - never controls GPU training",
        "STATUS: " + display_status(run, telemetry, queue),
        f"Seed: {run.get('seed', queue.get('seed', '?'))} | Phase: {run.get('phase', 'waiting')} | "
        f"Hardware sample age: {number(sample_age, 1)}s | Run update age: {number(age(run.get('updated_at')), 1)}s",
        f"CURRENT LR: backbone {number(rates.get('backbone'), 7)} | classifier {number(rates.get('classifier'), 7)}",
        "-" * WIDTH]


def overview(snapshot):
    run, config, telemetry = (snapshot.get(k, {}) for k in ("run", "config", "telemetry"))
    reading = telemetry.get("reading", {})
    history = snapshot.get("history", [])
    metrics, best_epoch = verified_metrics(snapshot)
    current = history[-1] if history else {}
    phase = run.get("phase", "waiting")
    complete = run.get("status") == "completed"
    return ["MODEL: DenseNet-201 | Fresh ImageNet + new classifier | All layers trainable | 3-class classification",
        "CLASS ORDER: Normal -> Viral -> Bacterial | Full-image classification, no ROI/segmentation",
        "PIPELINE: VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE | 224 x 224",
        f"CURRENT EPOCH: {run.get('epoch', '?')}/{config.get('epochs', 12)} maximum | Completed epochs: {len(history)}",
        (f"{phase.upper()} BATCHES: " + progress_bar(run.get("batch"), run.get("total_batches")))
          if phase in ("train", "validation") else f"PHASE: {phase} (no training-batch bar in this phase)",
        "REPORTED RUN TIME: " + duration(run.get("elapsed_seconds")) + " | ETA: unknown; early stopping may finish sooner",
        "", "LATEST COMPLETED EPOCH: " + str(int(current["epoch"]) if current else "waiting for first epoch"),
        f"Train loss {number(current.get('train_loss'))} | Validation loss {number(current.get('val_loss'))} | "
        f"Epoch duration {duration(current.get('elapsed_seconds'))}",
        f"Accuracy {pct(current.get('val_accuracy'))} | Balanced accuracy {pct(current.get('val_balanced_accuracy'))} | "
        f"Macro F1 {pct(current.get('val_macro_f1'))}",
        ("BEST SELECTED CHECKPOINT (run complete): epoch " if complete else
         "BEST SELECTED CHECKPOINT SO FAR (INTERIM until run completion): epoch ") + str(best_epoch or "waiting"),
        f"Accuracy {pct(metric_value(metrics, 'accuracy'))} | Balanced accuracy {pct(metric_value(metrics, 'balanced_accuracy'))} | "
        f"Macro F1 {pct(metric_value(metrics, 'macro_f1'))}",
        f"Recall: Normal {pct(metric_value(metrics, 'normal_recall'))} | Viral {pct(metric_value(metrics, 'viral_recall'))} | "
        f"Bacterial {pct(metric_value(metrics, 'bacterial_recall'))}",
        ("Run completed; final model selection requires the controller's applicable gate checks." if complete else
         "No terminal gate/model-selection decision is implied by these interim scores."),
        "", "SEQUENTIAL QUEUE (at most 3 successful full runs):", *queue_lines(snapshot),
        "Incumbent: FT-004 stays selected unless every required replacement gate passes.",
        "", f"GPU {reading.get('gpu_name', 'unavailable')} | {number(reading.get('gpu_temperature_c'), 1)} C | "
        f"Utilization {number(reading.get('gpu_utilization_percent'), 1)}% | VRAM {gib(reading.get('gpu_vram_used_bytes'))}/{gib(reading.get('gpu_vram_total_bytes'))}",
        f"RAM available {gib(reading.get('ram_available_bytes'))}/{gib(reading.get('ram_total_bytes'))} | "
        f"Disk free {gib(reading.get('disk_free_bytes'))} | AC {reading.get('ac_connected', 'unavailable')} | Battery {reading.get('battery_percent', '?')}%",
        f"GPU peak {number(telemetry.get('peaks', {}).get('gpu_temperature_c'), 1)} C | "
        f"Actual overheating {telemetry.get('actual_overheat_events', 0)} | "
        f"Simulated smoke pauses {telemetry.get('simulated_pause_events', 0)} | Monitoring gaps {telemetry.get('monitoring_gaps', 0)}",
        "Smoke check: " + snapshot.get("smoke", {}).get("status", "not verified") +
        " | Training unchanged | Locked test not used for tuning | Website unchanged"]


def configuration(snapshot):
    config, protocol = snapshot.get("config", {}), snapshot.get("protocol", {})
    aug = config.get("augmentation", {})
    rates = protocol.get("learning_rate_policy", {})
    metadata, weights = snapshot.get("metadata", {}), snapshot.get("weights", {})
    counts = snapshot.get("split_summary", {}).get("split_total_counts", {})
    scheduler = protocol.get("scheduler_defaults", {})
    return ["FIXED EXPERIMENT - ONLY INTENDED CHANGE FROM FT-010: differential learning rates",
        "Architecture: DenseNet-201, one model, Normal / Viral / Bacterial; no ensemble",
        "Initialization: fresh ImageNet-pretrained backbone + new seed-matched 3-output classifier",
        "No FT-010 checkpoint continuation; no freezing/warm-up/ROI/extra augmentation/threshold tuning",
        f"Parameters: {metadata.get('parameter_count', 'unavailable')} | Trainable: {metadata.get('trainable_parameter_count', 'unavailable')}",
        "", "DATA AND PREPROCESSING (exact existing implementation):",
        f"Locked assigned_split: train {counts.get('train', 4076):,}; validation {counts.get('val', 874):,}; "
        f"sealed test {counts.get('test', 874):,} (not opened for tuning)",
        "Read original image -> RGB -> VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE",
        "VIRIDIS: OpenCV color map on grayscale; GE: PIL ImageOps.equalize on grayscale",
        "CLAHE: Contrast Limited Adaptive Histogram Equalization; clip 2.0, tile grid 8 x 8",
        "GE/CLAHE convert to grayscale; final grayscale is replicated into RGB channels",
        "Train only: horizontal flip p=" + str(aug.get("horizontal_flip_probability", 0.5)) +
        "; random rotation +/-" + str(aug.get("rotation_degrees", 5.0)) + " degrees (bilinear, black fill)",
        "No translation, scale jitter or vertical flip; validation has no random augmentation",
        "Then aspect-ratio-preserving fit/center crop to 224 x 224 (bilinear)",
        "Float32 pixels /255 -> HWC to CHW -> ImageNet normalization -> tensor 3 x 224 x 224",
        "ImageNet mean [0.485, 0.456, 0.406]; std [0.229, 0.224, 0.225]",
        "", "OPTIMIZATION AND RUNTIME:",
        f"AdamW | weight decay {config.get('weight_decay', 0.0001)} | batch {config.get('batch_size', 8)} | max epochs {config.get('epochs', 12)}",
        f"Initial LR: backbone {number(rates.get('backbone', 1e-5), 7)}; classifier {number(rates.get('classifier', 1e-4), 7)}",
        "Control FT-010 used 0.0001000 for BOTH groups; optimizer groups are disjoint/exhaustive",
        "Training loss: balanced class-weighted cross-entropy + label smoothing " + str(config.get("label_smoothing", 0.05)),
        "Class weights (Normal/Viral/Bacterial): " + ("/".join(number(v, 8) for v in weights.get("balanced_class_weights", [])) or "unavailable"),
        "Validation loss: ordinary unweighted cross-entropy; not directly the same loss as training",
        f"ReduceLROnPlateau: mode max, factor {scheduler.get('factor', 0.5)}, patience {scheduler.get('patience', 2)}; monitors macro F1",
        "Scheduler: relative threshold 0.0001, cooldown 0, minimum LR [0,0], epsilon 1e-8",
        f"Best checkpoint: macro F1 > previous best + {config.get('minimum_delta', 0.0001)}",
        f"Early stopping: {config.get('early_stopping_patience', 4)} epochs without sufficient improvement",
        f"CUDA | mixed precision {config.get('amp', True)} | workers {config.get('num_workers', 0)} | TensorBoard {config.get('tensorboard', True)}",
        "BatchNorm remains trainable; original deterministic seed and sample-order handling retained",
        "Prediction rule: softmax -> argmax; no probability-threshold tuning",
        "Validation evaluation preserves control AMP; full-precision gates use unrounded metrics",
        "Python " + protocol.get("python", "unavailable") + " | " + "; ".join(f"{k} {v}" for k, v in protocol.get("packages", {}).items())]


def results(snapshot):
    history = snapshot.get("history", [])
    metrics, best_epoch = verified_metrics(snapshot)
    run = snapshot.get("run", {})
    complete = run.get("status") == "completed"
    controls = snapshot.get("protocol", {}).get("controls", {})
    lines = ["FULL RUN VALIDATION ONLY - smoke scores excluded; sealed test untouched",
        "Epoch | Train loss | Val loss | Accuracy | Balanced acc | Macro F1 | N/V/B recall | Duration"]
    for row in history[-12:]:
        lines.append(f"{int(row['epoch']):5d} | {number(row.get('train_loss')):>10} | {number(row.get('val_loss')):>8} | "
            f"{pct(row.get('val_accuracy')):>8} | {pct(row.get('val_balanced_accuracy')):>12} | {pct(row.get('val_macro_f1')):>8} | "
            + "/".join(pct(row.get('recall_' + name)) for name in ('normal', 'viral', 'bacterial')) +
            " | " + duration(row.get("elapsed_seconds")))
    if not history:
        lines.append("Waiting for first completed full-training epoch; no result is invented.")
    lines.extend(["", "BEST CHECKPOINT: epoch " + str(best_epoch or "waiting") +
        (" (run complete; controller verification still applies)" if complete else " (INTERIM - training unfinished)"),
        "Class      Precision     Recall        F1       Specificity     Support"])
    for name in ("Normal", "Viral", "Bacterial"):
        value = metrics.get("per_class", {}).get(name, {})
        lines.append(f"{name:<10} {pct(value.get('precision')):>11} {pct(value.get('recall')):>11} "
            f"{pct(value.get('f1')):>11} {pct(value.get('specificity')):>14} {str(value.get('support', 'unavailable')):>10}")
    lines.extend([f"Accuracy {pct(metrics.get('accuracy'))} | Macro F1 {pct(metrics.get('macro_f1'))} | "
        f"Balanced accuracy {pct(metrics.get('balanced_accuracy'))}",
        f"ROC AUC macro {number(metrics.get('roc_auc_ovr_macro'))} | PR AUC macro {number(metrics.get('pr_auc_macro'))}",
        "Confusion matrix (rows=true, columns=predicted; order Normal/Viral/Bacterial): " +
        json.dumps(metrics.get("confusion_matrix", "waiting")), "", "SEED-42 SCREENING GATES (ALL five required):",
        "Metric                Current        Exact target [0,1]       Delta pp       Status"])
    s42 = metrics if run.get("seed") == 42 else snapshot.get("seed_metrics", {}).get("42", {})
    s42complete = snapshot.get("seed_statuses", {}).get("42", run if run.get("seed") == 42 else {}).get("status") == "completed"
    for gate in controls.get("screening_gates", []):
        value, target = metric_value(s42, gate["metric"]), gate["threshold"]
        reached = value is not None and value >= target
        label = ("MEETS" if reached else "BELOW") if value is not None else "WAITING"
        label += " / pending verification" if s42complete else " / INTERIM"
        delta = "unavailable" if value is None else f"{(value-target)*100:+.3f}"
        lines.append(f"{LABELS.get(gate['metric'], gate['metric']):<21} {pct(value):>10}  >= {target:.16f}  {delta:>9}  {label}")
    lines.extend(["Accuracy is reported but is NOT one of these five screening gates.",
        "Do not advance based on interim scores; only the verified completed selected checkpoint counts.",
        "", "FINAL THREE-SEED REPLACEMENT GATES (arithmetic per-seed means; never ensemble seeds):"])
    seed_statuses, seed_metrics = snapshot.get("seed_statuses", {}), snapshot.get("seed_metrics", {})
    for gate in controls.get("replacement_gates", []):
        values = [metric_value(seed_metrics.get(str(s), {}), gate["metric"]) for s in (42, 123, 2026)]
        all_done = all(seed_statuses.get(str(s), {}).get("status") == "completed" for s in (42, 123, 2026))
        mean = sum(values) / 3 if all_done and all(v is not None for v in values) else None
        lines.append(f"Mean {LABELS.get(gate['metric'], gate['metric']):<21} {pct(mean):>11}  "
                     f"{gate['operator']} {gate['threshold']:.16f} | {gate.get('basis', '')}")
    lines.append("OFFICIAL DECISION: " + snapshot.get("decision", {}).get("decision", "PENDING - FT-004 remains incumbent"))
    lines.append("Bootstrap: 10,000 paired resamples after results; proxy clusters, not verified patient independence")
    return lines


def hardware(snapshot):
    telemetry, run = snapshot.get("telemetry", {}), snapshot.get("run", {})
    reading, policy, peaks = (telemetry.get(k, {}) for k in ("reading", "policy", "peaks"))
    lines = ["LIVE HARDWARE / PROTECTIVE WATCHDOG - display does not change safety thresholds",
        "Hardware sample: " + local_time(reading.get("sample_completed_utc")),
        "Run-status update: " + local_time(run.get("updated_at")),
        f"GPU: {reading.get('gpu_name', 'unavailable')} | Driver {reading.get('gpu_driver_version', 'unavailable')}",
        f"GPU temperature: {number(reading.get('gpu_temperature_c'), 1)} C | Utilization {number(reading.get('gpu_utilization_percent'), 1)}%",
        f"Device VRAM: {gib(reading.get('gpu_vram_used_bytes'))} used / {gib(reading.get('gpu_vram_total_bytes'))} total",
        f"CPU utilization: {number(reading.get('cpu_utilization_percent'), 1)}% | CPU temperature: "
          + ("unavailable" if reading.get('cpu_temperature_c') is None else number(reading['cpu_temperature_c'], 1) + " C"),
        "CPU sensor: " + reading.get('cpu_temperature_status', 'unavailable - no verified package/core sensor'),
        f"System RAM: {gib(reading.get('ram_available_bytes'))} available / {gib(reading.get('ram_total_bytes'))} total",
        f"Disk: {gib(reading.get('disk_free_bytes'))} free | Checkpoint reserve {gib(telemetry.get('disk_reserve_bytes'))}",
        f"AC power: {reading.get('ac_connected', 'unavailable')} | Battery {reading.get('battery_percent', 'unavailable')}%",
        f"GPU thermal throttling: {reading.get('gpu_thermal_throttling', 'unavailable')}",
        "", "HARDWARE-ANCHORED POLICY (project margins, not universal safe-temperature claims):",
        f"GPU warn {policy.get('gpu_warning_c', '?')} C | Pause {policy.get('gpu_pause_c', '?')} C | "
        f"Critical stop {policy.get('gpu_critical_c', '?')} C | Resume below {policy.get('gpu_resume_c', '?')} C",
        f"Hardware target {policy.get('hardware_target_c', '?')} C / slowdown {policy.get('hardware_slowdown_c', '?')} C / shutdown {policy.get('hardware_shutdown_c', '?')} C",
        f"Available RAM: warn below {gib(policy.get('ram_warning_available_bytes'))}; pause below {gib(policy.get('ram_pause_available_bytes'))}; "
        f"resume at least {gib(policy.get('ram_resume_available_bytes'))}",
        f"VRAM use: warn {pct(policy.get('vram_warning_fraction'))}; pause {pct(policy.get('vram_pause_fraction'))}; "
        f"resume below {pct(policy.get('vram_resume_fraction'))}",
        f"Resume: {policy.get('continuous_cooling_seconds', 120):g} continuous healthy seconds + restored checkpoint + AC + sufficient disk",
        f"Heartbeat max age {policy.get('heartbeat_max_age_seconds', 6):g}s | Sample every {policy.get('sample_interval_seconds', 2):g}s",
        f"Protection active: pause {telemetry.get('pause_active', False)} | terminal stop {telemetry.get('terminal_stop', False)}"]
    started = telemetry.get("cooling_started_monotonic")
    if telemetry.get("pause_active"):
        sample_age = age(reading.get("sample_completed_utc"))
        if started is not None and sample_age is not None and sample_age <= policy.get("heartbeat_max_age_seconds", 6):
            healthy = max(0.0, reading.get("monotonic_time", started) - started)
            lines.append("Continuous healthy cooling: " + duration(healthy) + " | Remaining " +
                duration(max(0.0, policy.get("continuous_cooling_seconds", 120) - healthy)))
        else:
            lines.append("Cooling HOLD - continuous safe conditions not established / countdown unverified")
    lines.extend(["", "COUNTERS / PEAKS (shared history includes the completed smoke check):",
        f"Actual overheating {telemetry.get('actual_overheat_events', 0)} | Simulated smoke pauses {telemetry.get('simulated_pause_events', 0)}",
        f"Total cooling pauses including smoke {telemetry.get('cooling_pauses', 0)} | Monitoring gaps {telemetry.get('monitoring_gaps', 0)} | "
        f"Thermal-throttle samples {telemetry.get('thermal_throttle_samples', 0)}",
        f"Peak GPU {number(peaks.get('gpu_temperature_c'), 1)} C; device VRAM {gib(peaks.get('gpu_vram_used_bytes'))}; "
        f"RAM used {gib(peaks.get('ram_used_bytes'))}; CPU {number(peaks.get('cpu_utilization_percent'), 1)}%",
        "Peak PyTorch allocated memory (latest completed epoch): " +
        gib(snapshot.get("history", [{}])[-1].get("peak_gpu_memory_bytes") if snapshot.get("history") else None),
        "Required telemetry errors: " + ("; ".join(reading.get("errors", [])) or "none reported"),
        "Last recorded terminal interruption: " + json.dumps(snapshot.get("interruption", {}) or "none"),
        "Safe batch boundary -> complete atomic model/optimizer/scheduler/scaler/RNG/order checkpoint -> cooling",
        "Repeated actual overheating stops the job. Unsupported optional CPU temperature is not 0 degrees.",
        "Keep AC connected and vents clear on a hard ventilated surface. No applications are auto-closed."])
    return lines


def files(snapshot):
    protocol, candidate = snapshot.get("protocol", {}), snapshot.get("candidate", {})
    root, run_dir = Path(snapshot.get("root", ".")), Path(snapshot.get("run_dir", "."))
    lines = ["AUDIT / FILES - only JSON/CSV metadata is read; no images or checkpoints are loaded",
        "Experiment directory: " + str(root), "Current run directory: " + str(run_dir),
        "Run ID: " + snapshot.get("config", {}).get("run_id", "unavailable"),
        "Protocol SHA256: " + protocol.get("protocol_hash", "unavailable"),
        "Candidate configuration SHA256: " + candidate.get("candidate_configuration_hash", "unavailable"),
        "Manifest version SHA256: " + protocol.get("manifest_version_hash", "unavailable"),
        "ImageNet weights: " + protocol.get("weight_identifier", "unavailable"),
        "ImageNet SHA256: " + protocol.get("weights_sha256", "unavailable"),
        "Fresh initialization: " + str(snapshot.get("initialization", {}).get("fresh_imagenet_initialization", "unavailable")),
        "Initial model SHA256: " + snapshot.get("initialization", {}).get("model_state_sha256", "unavailable"),
        "Initial classifier SHA256: " + snapshot.get("initialization", {}).get("classifier_state_sha256", "unavailable"),
        "", "ARTIFACTS (existence/size only; large checkpoint hashes are not repeatedly recalculated):"]
    for name in ("candidate_configuration.json", "config.json", "initialization.json", "optimizer_groups.json",
                 "scheduler_initial.json", "class_weights.json", "history.csv", "status.json",
                 "last_checkpoint.pt", "best_checkpoint.pt", "metrics.json", "predictions.csv",
                 "confusion_matrix.csv", "confusion_matrix.png", "classification_report.csv", "training_curves.png"):
        path = run_dir / name
        try:
            label = f"saved, {path.stat().st_size:,} bytes"
        except OSError:
            label = "pending / not yet saved"
        lines.append(f"{name:<31}: {label}")
    lines.extend(["", "Console log: " + str(root / "queue_console.log"),
        "Timestamped hardware/safety logs: " + str(root / "monitoring"),
        "Live dashboard snapshot/journal: " + str(root / "dashboard"),
        "Protected smoke result: " + str(root / "smoke_verified.json"),
        "Preserved interrupted check: " + str(root / "interrupted_checks/20261013_1349"),
        "Full report/graphs after terminal decision: " + str(root / "analysis"),
        "Final decision/frozen selection: " + str(root / "final_decision.json") + " / frozen_selection.json",
        "No terminal FT-011 result is claimed before the controller writes its decision."])
    return lines


def render_dashboard(snapshot, view="overview"):
    renderers = {"overview": overview, "config": configuration, "results": results,
                 "hardware": hardware, "files": files}
    lines = header(snapshot, view)
    if view == "all":
        for name, renderer in renderers.items():
            lines.extend(["", "### " + name.upper(), *renderer(snapshot)])
    else:
        lines.extend(renderers.get(view, overview)(snapshot))
    lines.extend(["-" * WIDTH, "Keys 1-5 select live views; A shows all (scroll); Q closes DISPLAY ONLY, never the training."])
    return "\n".join(part for line in lines for part in (textwrap.wrap(str(line), WIDTH, replace_whitespace=False,
                    drop_whitespace=False, break_long_words=False, break_on_hyphens=False) or [""]))


def save_display_record(snapshot):
    directory = Path(snapshot["root"]) / "dashboard"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"display_updated_at": datetime.now(MANILA).isoformat(timespec="seconds"), **snapshot}
    temp = directory / (f"latest.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, directory / "latest.json")
    run, reading = snapshot.get("run", {}), snapshot.get("telemetry", {}).get("reading", {})
    entry = {"timestamp": payload["display_updated_at"], "run": run,
        "status": display_status(run, snapshot.get("telemetry", {}), snapshot.get("queue", {})),
        "hardware_sample": reading.get("sample_completed_utc"), "gpu_c": reading.get("gpu_temperature_c"),
        "ram_available": reading.get("ram_available_bytes"), "best_metrics": snapshot.get("metrics", {})}
    with (directory / "display_journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--view", choices=(*VIEWS, "all"), default="overview")
    parser.add_argument("--once", action="store_true", help="Print one snapshot without starting a display loop")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if os.name == "nt" and not args.once:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("FT-011 Live Monitor")
    view, next_refresh = args.view, 0.0
    while True:
        if time.monotonic() >= next_refresh:
            snapshot = load_snapshot(args.log_dir, args.progress)
            if not args.once:
                os.system("cls" if os.name == "nt" else "clear")
            print(render_dashboard(snapshot, view), flush=True)
            if args.once:
                return
            try:
                save_display_record(snapshot)
            except (OSError, ValueError) as exc:
                print("DISPLAY JOURNAL WARNING (watchdog unaffected): " + str(exc))
            next_refresh = time.monotonic() + max(0.2, args.interval)
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "q":
                    return
                if key in "12345":
                    view, next_refresh = VIEWS[int(key) - 1], 0.0
                elif key == "a":
                    view, next_refresh = "all", 0.0
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
