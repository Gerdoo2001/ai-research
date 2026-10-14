# FT 011 execution readiness, restart and live training record

Recorded 13 October 2026 in Asia Manila time.

**Terminal update:** FT-011 completed seed 42 on 13 October 2026 at 15:04:07 Manila. It selected epoch 5, stopped after epoch 9, and failed four of five frozen screening gates. FT-004 is retained; seeds 123 and 2026 were not run; this tuning phase is closed. Validation macro F1 was 0.8202782987500647, accuracy 0.834096109839817, and balanced accuracy 0.8173146418282663. The [full final report](FT011_Final_Report_20261013.md) supersedes the progress snapshot below for current status.

The remaining sections preserve the timestamped execution-readiness and live-progress history. References to pending work or a running process describe those earlier checks, not the current terminal state. The repeated protected smoke passed at 13:52:09; full training started fresh at 13:52:14. The first interrupted smoke remains archived.

## Records and control recovery

The supplied [FT-011 protocol][1], [completed fine-tuning report][2], and [dataset preparation record][3] were read through the connected Google Drive tools. Their extracted contents are preserved in model/results/ft011/source_records. The protocol document's statement that training was already in progress was not supported by the local repository: no FT-011 files, results, or training process were present at inspection.

The original project is C:\\Users\\JOHN MARK\\Code\\ai-research-main. Historical FT-007 and FT-010 configurations, checkpoints, full-precision metrics, and validation probabilities were recovered. All six FT-010 and FT-004 validation prediction files align with the same 874 assigned validation cases and their labels. Thirty-one control artifacts are hashed in the frozen protocol.

Git core.autocrlf had converted four locked manifest CSV files to CRLF. A line-ending-only restoration to LF recovered every original frozen SHA-256 exactly. No assignments or labels changed. A .gitattributes rule now preserves LF for the locked CSVs. Train, validation, and sealed-test assignments remain 4,076, 874, and 874. No test images were opened for this experiment.

## Fixed candidate

The candidate uses one fully trainable ImageNet DenseNet-201 with Normal, Viral, Bacterial outputs. It retains full-image VIRIDIS then global histogram equalization then CLAHE, 224 by 224 input, the original RGB/grayscale conversions and resize/crop implementation, ImageNet normalization, balanced class-weighted cross-entropy with label smoothing 0.05, AdamW, weight decay 0.0001, batch size 8, at most 12 epochs, and early-stopping patience 4. Training augmentation remains horizontal flip probability 0.5 and rotation within plus or minus 5 degrees.

The only intended training change is the initial learning rate: 0.00001 for the backbone and 0.0001 for the classifier. Optimizer groups are disjoint and cover all 18,098,691 trainable parameters. Both rates are logged before and after scheduler updates. The scheduler remains ReduceLROnPlateau, mode max, factor 0.5, patience 2, relative threshold 0.0001, cooldown 0, minimum rates [0, 0], and epsilon 0.00000001. It monitors validation macro F1. The best checkpoint changes only when macro F1 is greater than the previous best plus the control minimum delta of 0.0001.

Python 3.12.6, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA runtime 12.1 and cuDNN 90100 match the recorded environment. Relevant image and metric package versions also match. DenseNet201_Weights.DEFAULT resolves to IMAGENET1K_V1. The cached weight SHA-256 is c1103571aad4f45d0b5b15556d92944db94c33bee68cd58babba3d76225a6401.

Historical runs did not retain initial model-state hashes or a per-run source commit. Their saved configuration, checkpoint, model metadata, environment record and predictions are available. FT-011 records current implementation hashes and fresh initialization hashes; absent historical initial-state hashes are a provenance limitation, not evidence that those initializations were independently compared byte for byte.

## Frozen gates

No candidate outcome has been used to set these thresholds. Protocol SHA-256: 7a667b9dc65b94ecb89f38e87ef85490c44ab329d1a517b994f966ee4056cee4.

| Seed 42 screening metric | Exact threshold | Current status |
| --- | ---: | --- |
| Macro F1 | >= 0.8613107953930739 | FAIL: 0.8202782987500647 |
| Balanced accuracy | >= 0.8507499938072187 | FAIL: 0.8173146418282663 |
| Viral recall | >= 0.6681614349775785 | FAIL: 0.5829596412556054 |
| Normal recall | >= 0.9831223628691983 | PASS: 0.9873417721518988 |
| Bacterial recall | >= 0.9009661835748792 | FAIL: 0.8816425120772947 |

The macro-F1 threshold is the matched FT-010 seed-42 value 0.8563107953930739 plus exactly 0.005, or 0.500 percentage points. Every screening gate must pass before seeds 123 and 2026 may run.

| Three-seed replacement metric | Exact threshold | Current status |
| --- | ---: | --- |
| Mean macro F1 | > 0.8530516857750923 | Not evaluated |
| Mean balanced accuracy | >= 0.8492846583463107 | Not evaluated |
| Mean Viral recall | >= 0.6651718983557549 | Not evaluated |
| Mean Normal recall | >= 0.9789029535864979 | Not evaluated |
| Mean Bacterial recall | >= 0.8969404186795491 | Not evaluated |

FT-004 is retained if a screening gate fails, a replacement gate fails, or mean macro F1 ties. At most three successful full candidate runs are permitted. The deployment seed remains 42.

Replacement gates were not evaluated because seed-42 screening failed. A single seed does not supply a three-seed mean or sample standard deviation.

## Protection and live display (timestamped execution history)

The separate visible Windows terminal was opened and its title verified as FT-011 Live Monitor. It refreshes approximately every two seconds and reads timestamped watchdog logs. The original laptop is an MSI Thin 15 B13UCX with Intel Core i7-13620H, RTX 2050 4 GiB and approximately 15.71 GiB physical RAM. AC power was confirmed through GetSystemPowerStatus. CPU utilization is available. CPU package temperature is unavailable from the installed sensors and is explicitly labelled unavailable.

The GPU reports target 87 degrees C, slowdown 97 degrees C and shutdown 100 degrees C. Recorded project margins give a warning at 80 degrees C, pause at 84 degrees C, critical stop at 92 degrees C, and resume below 75 degrees C for at least 120 continuous healthy seconds. These values are hardware-anchored project protection choices, not universal manufacturer guarantees. Hardware protection and fan control are preserved.

The watchdog also pauses for missing required telemetry, stale heartbeat, AC loss, less than 1 GiB available RAM, at least 98 percent VRAM use, or insufficient checkpoint disk reserve. Resume requires at least 2 GiB available RAM, VRAM below 92 percent, healthy readings, AC and adequate disk space continuously for two minutes. Checkpoint disk reserve is 1.5 GiB, using two 256 MiB estimates plus 1 GiB. Repeated actual overheating stops the experiment. Simulated pauses are labelled separately.

Thirty focused tests passed. They cover optimizer groups, original sample-order equivalence, complete recovery after a pending augmented batch, exact model/optimizer/scheduler/scaler/RNG restoration on synthetic CPU data, alignment, exact decision margins, strict ties, per-seed metric averaging, whole proxy-cluster resampling and watchdog conditions.

The first GPU smoke attempt used a maximum of 16 training and 16 validation cases. Available RAM fell below the pause budget. A complete atomic checkpoint was saved and restored, including the pending transformed batch and sampler position. It subsequently ended without a recorded termination reason. Its simulated-temperature schedule was applied while the actual RAM pause was active, so that attempt did not verify an isolated simulated protection and successful two-minute resume cycle. These outputs are archived, not overwritten.

At the last recorded check, GPU temperature was 37 degrees C and utilization 0 percent, with approximately 1.19 GiB system RAM available. Recorded peaks were 42 degrees C GPU temperature, 1,119,879,168 bytes VRAM, 15,858,860,032 bytes system RAM used, and 34.05 percent CPU utilization. No actual overheating or required-monitoring gap was recorded. These are smoke/preflight readings, not full-training resource measurements.

## Artifacts and remaining work

- Frozen protocol and exact controls: model/results/ft011/protocol.json
- Candidate configurations: model/results/ft011/configs/seed42.json, seed123.json, seed2026.json
- Complete smoke recovery checkpoint: model/results/ft011/smoke/last_checkpoint.pt
- Smoke status: model/results/ft011/smoke/status.json
- Timestamped telemetry: model/results/ft011/monitoring
- Protected runner: model/scripts/run_ft011.py
- Batch recovery trainer: model/src/multiclass/ft011_training.py
- Watchdog: model/src/multiclass/ft011_safety.py
- Analysis, decision gates and 10,000-resample paired proxy-cluster bootstrap: model/scripts/analyze_ft011.py

The preceding resource measurements describe the first execution-check attempt, not a currently running process. At the October 13, 13:48 Manila audit, neither the smoke process nor its queue helper was running, and telemetry had stopped at 13:43:25. No termination cause was recorded. No smoke verification or full candidate result existed. A completed protected smoke run is required before full seed-42 training.

The interrupted smoke checkpoint, monitoring logs and old progress record were preserved under model/results/ft011/interrupted_checks/20261013_1349 before repeating the identical limited check. This is a documented technical repeat, not an extra successful full candidate run. The frozen protocol and learning semantics are unchanged. A one-time queue helper may start full training only when the repeated protected smoke has verified completion. The full runner rechecks the protocol, starts a fresh seed-42 model and enforces every advancement gate before seeds 123 and 2026.

The user clarified that the Google protocol's "ongoing" label was a facade/status label rather than evidence that full FT-011 training had already occurred. Accordingly, no earlier candidate training is claimed. "Fresh" means the protocol's ImageNet-pretrained DenseNet-201 with a new, seed-matched classifier, not continued training from an FT-010 checkpoint and not random initialization of the entire backbone. The laptop must remain on AC on a hard, ventilated surface with clear vents. No unrelated applications were closed.

The repeated smoke check passed at 13:52:09. Its simulated protection event was counted separately from actual overheating. The watchdog saved and restored the complete state, verified 120 continuous healthy seconds, resumed and completed the limited check. Actual overheating and required-monitoring gaps were zero. Smoke scores are execution checks only and must not appear as full experimental performance results.

Full seed-42 training was verified running at 13:52:57, epoch 1, batch 54 of 510. Its initialization hashes match the fresh seed-42 ImageNet construction used by the smoke check: model state 71c02f26964db716df7078e5fea68f0172265017a7e80527649adb8f98143ad9; new classifier bcdd9379b57b695784bea230bdc71119ed6865a0dd9b4c7aef70f151573b45b5. No smoke-trained or FT-010 checkpoint was used to initialize full training. The candidate directory is model/results/ft011/runs/ft011_densenet201_differential_lr_s42. The latest live reading was GPU 41 degrees C, GPU utilization 56 percent, VRAM 1,289,748,480 bytes, available RAM 2,686,246,912 bytes, and AC connected. These are point-in-time readings, not final peaks.

Training runs in a detached process with the automatic safety watchdog. The visible monitor is still open. The sequential controller evaluates the five seed-42 screening gates before permitting seeds 123 and 2026, and generates the terminal report automatically when the applicable runs finish. The operational queue helper's process check was corrected to accept a quoted script path; this did not change any frozen training source or experiment setting. There is no completed FT-011 candidate score yet.

Actual loss curves, validation curves, confusion matrices, candidate metrics, bootstrap differences and a terminal retain FT-004 or select FT-011 report will be generated only after the corresponding full runs complete. The bootstrap uses saved validation probabilities and intact derived split groups, shared across systems and seeds; each seed's metrics are calculated separately before averaging. The derived groups are proxies and do not establish verified patient independence.

After a terminal selection, the next research phase is independent evaluation of the frozen system. Current validation evidence must not be presented as external performance. The live website remains unchanged.

## Numbered sources

### Live-display update, 13 October 2026

The monitor was expanded without changing any frozen training code, configuration, optimizer state or sample ordering. It reads authoritative run status separately from queue status and hardware telemetry; the queued "training" label no longer hides a protective pause or completed run. It refreshes approximately every two seconds and shows Manila timestamps, phase-specific completed-batch bars, both current learning rates, latest completed epoch, best selected checkpoint so far, conditional seeds 123/2026, and explicitly provisional comparison scores until full training completes.

Keyboard views are 1 Progress, 2 Full configuration (data preparation, preprocessing, augmentation, loss, optimizer and scheduler), 3 Results and exact gates (including per-class scores and confusion matrix), 4 Hardware and safety, 5 Files and hashes, and A All. Q closes only the display, not training. The monitor does not load images, checkpoint tensors or PyTorch. It saves a current dashboard snapshot and timestamped journal under model/results/ft011/dashboard. Seventeen focused standard-library display tests passed, including repeated-score checkpoint attribution and completed-run labels. Existing frozen training-source hashes remain unaffected.

[1]: https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit
[2]: https://docs.google.com/document/d/1mdLbxZMtv2icHKNs-VSjpZiWb5_2lG-S/edit
[3]: https://docs.google.com/document/d/1I9-egb-SZRzbGQ-PXxM8UjvGmc_CCcsZ/edit

[1] [FT-011 protocol and rationale][1]

[2] [Completed fine-tuning report][2]

[3] [Dataset preparation and original experimental journal][3]

The original protocol provides the methodological rationale in [Yosinski et al. 2014](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html), [Long et al. 2015](https://proceedings.mlr.press/v37/long15.html), [Cawley and Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [TRIPOD plus AI 2024](https://www.bmj.com/content/385/bmj-2023-078378), [Lakshminarayanan et al. 2017](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) and [Riley et al. 2024](https://www.bmj.com/content/384/bmj-2023-074820). The exact learning rates, numerical gates and protection margins remain project choices.

