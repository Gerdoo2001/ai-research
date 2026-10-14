# Current Status

FT-011 completed on 13 October 2026. Its fixed seed-42 screening failed four of five gates, so the terminal decision is **retain FT-004**. This tuning phase is closed. There is no active training; FT-011 seeds 123 and 2026 were not run because the protocol required stopping after any failed seed-42 gate.

## Selected frozen system

- **Chosen system:** FT-004, with exactly three DenseNet-201 members using the original predeclared seed-42 checkpoints.
- **Member preprocessing:** VIRIDIS -> Global Histogram Equalization (GE) -> Contrast Limited Adaptive Histogram Equalization (CLAHE); VIRIDIS -> GE; GE.
- **Prediction rule:** Equal arithmetic mean of the three probability vectors, followed by argmax in class order Normal, Viral, Bacterial.
- **Historical robustness confirmation:** Completed at seeds 42, 123, and 2026. These are three separate evaluations; the deployed package contains three seed-42 members, not nine networks.
- **Three-seed mean validation metrics:** Macro F1 `0.8530516857750923`; accuracy `0.8627002288329519`; balanced accuracy `0.8492846583463107`; Viral recall `0.6651718983557549`.
- **Frozen selection:** `model/results/final_ft004_ensemble/frozen_selection.json`, SHA-256 `bbb65ee1202b2c34490cdb1047339795e5751f2c71edf6626d1b1f235acea57b`.
- **Executable inference:** `model/src/multiclass/inference.py::FT004EnsemblePredictor`.

## Final FT-011 result

FT-011 used a fresh ImageNet DenseNet-201 at seed 42 with all layers trainable. The intended change from FT-010 was the initial backbone learning rate `0.00001`, with the classifier remaining at `0.0001`. The matched full-image VIRIDIS -> GE -> CLAHE pipeline, 224 x 224 input, balanced class-weighted cross-entropy with label smoothing 0.05, AdamW weight decay 0.0001, batch size 8, locked 4,076 training/874 validation assignments, augmentation, mixed precision, scheduler and checkpoint rule were retained.

Training ended at epoch 9 of a maximum 12 through early stopping; the selected within-run checkpoint is epoch 5. Its validation accuracy is `0.834096109839817`. All screening comparisons used full-precision values:

| Seed-42 screening metric | FT-011 observed | Required threshold | Result |
| --- | ---: | ---: | --- |
| Macro F1 | 0.8202782987500647 | >= 0.8613107953930739 | Fail |
| Balanced accuracy | 0.8173146418282663 | >= 0.8507499938072187 | Fail |
| Viral recall | 0.5829596412556054 | >= 0.6681614349775785 | Fail |
| Normal recall | 0.9873417721518988 | >= 0.9831223628691983 | Pass |
| Bacterial recall | 0.8816425120772947 | >= 0.9009661835748792 | Fail |

No FT-011 three-seed mean, sample standard deviation or replacement-gate decision is claimed. The best candidate checkpoint is retained for reproducibility, but it is not the selected deployment package.

## Protection and interruptions

The first smoke attempt paused below the 1 GiB available-RAM budget and later disappeared without a recorded termination cause. Its checkpoint and logs were preserved under `model/results/ft011/interrupted_checks/20261013_1349`. The subsequent protected smoke completed at 13:52:09 Manila after a labelled simulated pause and a verified continuous cooling/resume cycle.

The full run paused safely at 14:16:39 Manila when a transient Windows access denial prevented atomic telemetry-file replacement. The watchdog treated that required-monitoring failure as unsafe, saved complete recovery state, and resumed at 14:18:45 after 122.547 continuous healthy seconds. This interruption is explicitly recorded even though the aggregate `monitoring_gaps` counter is zero. Recorded actual overheating and thermal-throttling counts are zero; peak GPU temperature is 51 C. The shared completed-smoke/full-run stream records one simulated smoke pause and two total cooling pauses. CPU temperature remains explicitly unavailable from the installed sensors.

## Evidence and next step

- **Terminal decision and exact metrics:** `model/results/ft011/final_decision.json`.
- **Protocol and controls:** `model/results/ft011/protocol.json`; semantic protocol hash `7a667b9dc65b94ecb89f38e87ef85490c44ab329d1a517b994f966ee4056cee4`.
- **Candidate artifacts:** `model/results/ft011/runs/ft011_densenet201_differential_lr_s42`, including both learning-rate histories, full checkpoints, validation probabilities and confusion matrix.
- **Final report and graphs:** `model/results/ft011/analysis/20261013T070414529276Z/FT011_Final_Report.md`.
- **Monitoring evidence:** `model/results/ft011/monitoring/final_telemetry_summary.json` and timestamped events.

The 10,000-resample paired comparisons use the observed matched seed-42 validation predictions and intact derived split groups. Those groups are filename/hash proxies, not verified patient identifiers, so patient independence is not established. These results are validation findings; earlier locked-test scores from DenseNet-201 + GE do not transfer to FT-004. No test images or test metrics were used for FT-011 tuning, and the live website was unchanged.

The next step is independent evaluation of the frozen FT-004 system. Intended use remains research comparison and prototype demonstration.

**Last terminal status:** 2026-10-13T15:04:15+08:00.
