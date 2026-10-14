# Development Log

## 2026-10-14 — documentation and selected-model publication

- Updated the original native FT-011 Google Doc in place using revision-guarded edits; preserved its existing citation table and source chip. Added the actual completed results, failed screening gates, skipped confirmation seeds, recovery disclosures and retained FT-004 decision.
- Produced updated full-journal, complete fine-tuning and standalone FT-011 Word reports, rendered and visually reviewed every final page. Historical dated reports were retained. Native Google Docs versions were imported into the existing research Drive folder, then read back for outcomes, black text, table counts and image counts; direct links are in `model/docs/README.md`.
- Packaged the three original FT-004 seed-42 checkpoints byte for byte with relative inference paths, original configuration provenance and SHA-256 verification. Git LFS carries the 657,342,824 checkpoint bytes. The website and legacy binary inference were not changed.
- Added portable inference/release tests; the full 109-test suite passed. A real three-member CPU forward pass on an artificial image passed; this is an execution check, not a clinical or dataset evaluation.
- Added scoped Git attributes to preserve frozen evidence bytes, locked manifest LF, legacy frozen-source CRLF and new FT-011 source LF across checkouts. No data assignments or frozen training semantics changed.

Append-only. Corrections are added as new entries; prior entries are not removed.

## 2026-07-16T16:55:00+08:00 - Phase 0 initiated

- **Objective:** Start with the required non-destructive repository and environment snapshot.
- **Files inspected:** Attached research specification; repository root; Git metadata; `model` directory structure.
- **Files created or modified:** None.
- **What was changed:** No repository content was changed.
- **Why it was changed:** Inspection is required before editing or training.
- **Commands executed:** `git status --short --branch`; `git rev-parse --show-toplevel`; recursive file inventories; disk and GPU queries.
- **Validation performed:** Confirmed repository root `C:\Users\JOHN MARK\Code\ai-research-main`; confirmed branch `main`; confirmed only pre-existing untracked files were present.
- **Outcome:** Safe audit scope established. Pre-existing untracked `requirements.working.txt` and backend `__pycache__` files will be preserved.
- **Next step:** Capture model assets, environments, dependencies, GPU runtime, and dataset topology.

## 2026-07-16T16:59:25+08:00 - Phase 0 environment and asset snapshot

- **Objective:** Record hardware, software, storage, dependency, output, and dataset facts before implementation.
- **Files inspected:** `model/requirements.txt`; `requirements.working.txt`; `PneumoNet.bat`; all 15 Python files under `model/src`; FastAPI inference files; existing DenseNet classification reports; model output inventory; dataset directory topology.
- **Files created or modified:** Documentation and machine-readable tracking files initialized in this change.
- **What was changed:** Created the continuous documentation structure and empty experiment ledgers after completing the non-mutating initial snapshot.
- **Why it was changed:** Continuous documentation is a required source of truth before major implementation.
- **Commands executed:** recursive PowerShell inventories; `nvidia-smi`; Python version and package queries; TensorFlow CUDA visibility/build-info checks; `pip check`; repository-wide `rg` search for requested terms.
- **Validation performed:** TensorFlow 2.10.1 reports one physical GPU and CUDA build support; NVIDIA-SMI reports RTX 2050, 4096 MiB VRAM, driver 576.28; approximately 108.24 GB was free on C:; 20,023 files were found under `model` at snapshot time.
- **Outcome:** GPU is available through the existing TensorFlow environment. Dependency gaps and the duplicated dataset topology are recorded for formal audit. No training was started.
- **Next step:** Finish the complete code/model-output audit, derive exact binary behavior, and resolve or mark ambiguous research terms.

## 2026-07-16T17:05:59+08:00 - Phase 1 legacy binary pipeline audit completed

- **Objective:** Explain the existing binary workflow, artifacts, compatibility, defects, leakage risks, and reusable elements from actual repository evidence.
- **Files inspected:** All 15 files under `model/src`; `model/requirements.txt`; `requirements.working.txt`; `PneumoNet.bat`; FastAPI backend code; all 27 legacy output artifacts; all six loadable complete HDF5 models; the weights-only HDF5 structure; all three text reports; all 17 saved PNG figures.
- **Files created or modified:** `model/scripts/snapshot_environment.py`; `model/scripts/inventory_repository.py`; `model/results/environment.json`; `model/docs/00_code_inventory.json`; `model/docs/00_existing_binary_pipeline_audit.md`; live status, decision, issue, and development logs.
- **What was changed:** Added non-mutating, reproducible audit utilities; generated the environment/code inventories; replaced the audit placeholder with evidence-backed findings.
- **Why it was changed:** Phase 1 must be complete before broad multiclass implementation, and legacy results must not be mistaken for three-class evidence.
- **Commands executed:** environment and code inventory scripts; HDF5 structure inspection; TensorFlow `load_model(..., compile=False)` checks; SHA-256 comparisons; repository-wide term search; visual inspection of all saved result figures.
- **Validation performed:** Both JSON inventories parse successfully; all complete model files load; DenseNet shapes/parameter counts match their HDF5 configs; text reports agree with plotted confusion matrices; ambiguous-term search is zero-hit outside the new documentation.
- **Outcome:** Existing binary behavior is preserved and fully documented. CLAHE is verified. FILM, SAS Tuning, automated grading, PCA, histogram equalization, pseudo-coloring, ResNet, and YOLO are absent from the legacy implementation.
- **Next step:** Generate and visually verify the repository-audit progress DOCX, then perform the immutable dataset and leakage audit.

## 2026-07-16T17:14:29+08:00 - Phase 1 progress report generated and structurally verified

- **Objective:** Produce the required live DOCX report from the current Markdown evidence.
- **Files inspected:** Document-generation skill instructions; all live documentation sources; six representative legacy result figures.
- **Files created or modified:** `model/scripts/generate_progress_report.py`; `model/docs/Pneumonia_Research_Progress_Report.docx`.
- **What was changed:** Added a deterministic Word report generator using the standard business brief design system and an editorial cover. Corrected the generated heading mapping and cover/contents page-break behavior after structural inspection.
- **Why it was changed:** The report must remain regenerated from the live sources of truth rather than becoming a disconnected manual artifact.
- **Commands executed:** DOCX generation; ZIP/package inspection; paragraph/style/heading/media/section audits; packaged render attempt; Microsoft Word COM export attempt.
- **Validation performed:** DOCX ZIP integrity, six media relationships, section settings, styles, and heading hierarchy passed. LibreOffice was absent and unattended Word export hung, so visual page rendering was not completed.
- **Outcome:** A structurally valid Phase 1 report was produced. The visual-QA limitation is recorded as ISSUE-003.
- **Next step:** Audit the immutable source dataset and construct leakage-controlled locked manifests.

## 2026-07-16T17:26:40+08:00 - Phase 2 immutable dataset audit and manifest lock completed

- **Objective:** Establish the unique canonical three-class image set, quantify integrity risks, and create deterministic group-aware train/validation/test manifests without modifying source files.
- **Files inspected:** All 17,591 physical source files; top-level and nested dataset trees; metadata/resource forks; canonical image headers and content; filenames, dimensions, modes, SHA-256 hashes, and DCT perceptual hashes.
- **Files created or modified:** `model/scripts/audit_dataset.py`; `model/scripts/build_manifests.py`; `model/scripts/plot_dataset_audit.py`; `model/scripts/verify_source_immutable.py`; audit CSV/JSON files and locked manifests under `model/data_manifests`; figures under `model/results/dataset_audit`; Phase 2 documentation.
- **What was changed:** Selected 5,856 canonical image candidates, excluded 32 redundant exact canonical copies, derived three labels from explicit filename/folder evidence, and assigned 5,824 retained images to deterministic 70/15/15 group-aware splits at seed 42.
- **Why it was changed:** The physical source contains a complete nested duplicate tree and the supplied split has a 16-image validation set with no Viral example plus 264 conservative cross-folder group matches.
- **Commands executed:** Dataset audit and SHA-256/pHash pass; deterministic manifest construction; Matplotlib audit plotting using system Python 3.12; source immutability verification; JSON/CSV count and hash checks; visual review of audit figures and sampled unusual/near-match images.
- **Validation performed:** All retained images decode and have label evidence; all three classes appear in every split; assigned group leakage = 0; assigned exact-hash leakage = 0; retained coverage = 5,824 exactly once; post-generation source rehash found zero mismatches across 17,591 files.
- **Outcome:** Locked manifest `three_class_seed42_3b815cf769c8` is ready for Phase 3. The test manifest is sealed for final selected-configuration evaluation only.
- **Next step:** Regenerate and structurally verify the Phase 2 progress report, then implement the configuration-driven multiclass pipeline and tests.

## 2026-07-16T17:54:43+08:00 - Phase 3 core pipeline and isolated CUDA environment validated

- **Objective:** Implement a parallel, resumable, configuration-driven classifier without changing the legacy TensorFlow pipeline.
- **Files inspected:** Phase 3 specification; locked manifest schemas/hashes; existing environments; torchvision and Ultralytics APIs; official pinned YOLOv5 source.
- **Files created or modified:** `.gitignore`; `model/requirements.multiclass-cu121.txt`; default config; `model/src/multiclass` modules; training/evaluation/smoke/asset scripts; unit tests; environment and tracking artifacts.
- **What was changed:** Added locked-manifest loaders, four preprocessing paths, eight supported model heads, AMP training, best/last checkpoints, resume, early stopping, scheduler/loss choices, metrics/plots, ledger/journal updates, and inference.
- **Why it was changed:** The research requires a fair common three-class contract, GPU execution, crash recovery, and actual artifacts while preserving binary behavior.
- **Commands executed:** Isolated Python 3.12 environment creation; pinned dependency install; `pip check`; CUDA tensor backward; unittest/compile checks; eight model-head CPU contracts; six torchvision and two YOLO CUDA backward smokes.
- **Validation performed:** `pip check` passes; CUDA/cuDNN visible; eight unit tests pass; all supported models emit three logits; YOLOv10 detection heads are rejected; manifest test lock is enforced.
- **Outcome:** Core implementation is ready for controlled experiment planning. The legacy TensorFlow environment was not modified.
- **Next step:** Validate end-to-end training, controlled resume, preprocessing families, inference, and progress reporting.

## 2026-07-16T18:05:59+08:00 - Phase 3 end-to-end and recovery smokes completed

- **Objective:** Prove the pipeline's real-data training, persistence, failure recovery, third-party adapters, evaluation artifacts, and inference contracts before baselines.
- **Files inspected:** Per-run configs/status/history/metrics/checkpoints/plots; experiment and failure ledgers; training journal; model support audit.
- **Files created or modified:** Ten completed smoke/contract run directories; one preserved failed YOLOv8 run; training journal/ledgers; implementation/testing/decision/issue logs.
- **What was changed:** Fixed probability normalization warnings, added controlled interruption testing, pinned YOLO assets, added missing Seaborn, normalized Ultralytics eval tuples to logits, and stabilized scientific configuration hashes against diagnostic-only fields.
- **Why it was changed:** Real execution surfaced integration details that static shape checks could not detect.
- **Commands executed:** DenseNet/ResNet/YOLO and preprocessing end-to-end CUDA smokes; epoch-1 controlled interruption plus epoch-2 resume; strict checkpoint inference for DenseNet, YOLOv5, and YOLOv8; full unittest reruns.
- **Validation performed:** Best/last checkpoints and all required smoke artifacts exist; history and training journal match; resume continuity passed; corrected inference returns normalized probabilities in Normal/Viral/Bacterial order; no locked test data was loaded.
- **Outcome:** Phase 3 implementation and smoke gate is complete. Smoke metrics are explicitly excluded from model selection.
- **Next step:** Regenerate the Phase 3 progress report, save preprocessing examples, finalize throughput/storage estimates and the controlled run matrix, then launch baselines.

## 2026-07-16T18:18:30+08:00 - Phase 4/5 preprocessing evidence and run matrix locked

- **Objective:** Validate preprocessing visually, prove pretrained initialization, and estimate baseline time/memory/storage before expensive work.
- **Files inspected:** Locked validation manifest; implemented transform metadata; pretrained model weight sources; CUDA resource measurements; one full-manifest pilot history/metrics.
- **Files created or modified:** `generate_preprocessing_examples.py`; preprocessing example PNG/JSON artifacts; `benchmark_training_resources.py`; `training_resource_estimate.json`; `build_run_matrix.py`; `run_matrix.csv`; `run_matrix_summary.json`; `02_experiment_plan.md`; live logs.
- **What was changed:** Saved validation-only transform comparisons, completed four pretrained family smokes, measured candidate batches for all eight supported models, ran a one-epoch full-manifest DenseNet121 pilot, and generated a 39-row staged matrix.
- **Why it was changed:** The specification requires evidence-backed throughput, disk, GPU-memory, duration, support, and experiment-count estimates before the benchmark.
- **Commands executed:** Preprocessing example generation/visual inspection; four pretrained end-to-end CUDA smokes; eight-model synthetic resource calibration; full 4,076-train/874-validation pilot; matrix generation and status audit.
- **Validation performed:** All candidate batches passed with peak allocation below 1.43 GB; pretrained heads produced three logits and full artifacts; the pilot completed in 253.996 seconds per epoch and never accessed the locked test; approximately 97.2 GB remained free.
- **Outcome:** Eight baselines are planned, YOLOv10 is explicitly unsupported, estimated baseline duration is 26,227 seconds before early stopping, and estimated storage is 4,122,949,400 bytes.
- **Next step:** Reverify source/GPU/tests/disk, regenerate the progress report, then launch Stage B sequentially with resume support.

## 2026-07-16T19:37:04+08:00 - Stage B launched and first controlled baseline completed

- **Objective:** Execute the supported architecture benchmark sequentially on the locked training and validation manifests while preserving crash recovery and continuous evidence.
- **Files inspected:** Launch-gate results; DenseNet121 config, status, history, metrics, checkpoints, predictions, plots, and run log; live experiment ledger and current status; active DenseNet169 status.
- **Files created or modified:** Completed `baseline_densenet121_s42` artifacts; active `baseline_densenet169_s42` artifacts; consolidated summary/ranking files; `05_results_interpretation.md`; progressive DOCX generator/report; live status, ledger, and training journal.
- **What was changed:** Launched the eight-run Stage B sequence after passing source immutability, dependency, test, CUDA, and disk gates. Materialized readable run logs, consolidated the matrix with zero consistency errors, and added dynamically discovered completed-baseline figures to the report.
- **Why it was changed:** Controlled results must be visible and auditable as they complete without treating partial rankings as final or accessing the locked test set.
- **Commands executed:** Background baseline launcher; periodic process/GPU/status/history checks; run-log materialization; result consolidation; DOCX generation and Open XML package inspection.
- **Validation performed:** DenseNet121 completed 12 epochs and selected epoch 10 at validation macro F1 0.837745. Its metrics/config/manifest identifiers agree with the ledger; one completed baseline is present in the ranking; consolidation reports zero consistency errors. The report ZIP package is valid and contains 14 media assets.
- **Outcome:** DenseNet121 is the provisional best by default because it is the only completed controlled baseline. DenseNet169 is running. The locked test manifest remains sealed.
- **Next step:** Complete the remaining seven supported baselines, reconsolidate the ranking, and select Stage C/D candidates from validation evidence only.

## 2026-07-16T19:47:55+08:00 - Later-stage selection, tuning, robustness, and test gates implemented

- **Objective:** Make Stages C-G executable and resumable from actual predecessor evidence while the independent Stage B GPU run continues.
- **Files inspected:** Training/config/tracking/evaluation interfaces; run matrix; consolidation logic; reproduction guide; current DenseNet169 status/history.
- **Files created or modified:** `run_tuning.py`; `resolve_run_matrix.py`; `evaluate_multiclass.py`; `consolidate_results.py`; `build_run_matrix.py`; pinned requirements; run matrix/summary; implementation, testing, decision, development, and reproduction documentation.
- **What was changed:** Installed and pinned Optuna 4.9.0; added SQLite-backed TPE studies with normal per-trial checkpoints and CSV/JSON exports; added validation-only matrix-resolution gates; added an independent seed-42 robustness row; and strengthened locked-test evaluation to require a matching pre-test selection lock.
- **Why it was changed:** Placeholder rows must be resolved from real validation evidence, failed trials must remain visible, three-seed comparisons need equal training budgets, and the single test evaluation must fail closed until selection is frozen.
- **Commands executed:** Optuna version inventory/install; `pip check`; Python compile/help checks for all new orchestration entry points; consolidation; active DenseNet169 status/history check.
- **Validation performed:** Optuna 4.9.0 imports; `pip check` reports no broken requirements; new scripts compile and expose expected CLI contracts; consolidation covers 40 matrix rows with zero consistency errors; DenseNet169 epoch 1 persisted normally on CUDA.
- **Outcome:** The complete post-baseline path now has exact commands and artifact contracts. No placeholder was resolved early, no tuning trial was launched, and the test manifest remains sealed.
- **Next step:** Finish Stage B, run Stage C, resolve Stage D/E from validation evidence, then perform independent three-seed robustness and the frozen single locked-test evaluation.

## 2026-07-16T19:57:00+08:00 - Completion-gated final reporting and Word table rendering implemented

- **Objective:** Prepare deterministic final Markdown/DOCX generation without allowing incomplete experiments to appear final, and improve progressive Word table readability.
- **Files inspected:** Document-generation skill; progressive generator; continuous sources; result schemas; split/environment artifacts; final-report requirements.
- **Files created or modified:** `generate_final_report.py`; progressive generator; final-report reproduction commands; orchestration tests; implementation/testing/development documentation; regenerated progress DOCX.
- **What was changed:** Added real Word-table rendering for Markdown tables, a final-report gate covering all experiment/test/consistency prerequisites, artifact-driven final narrative/table/error-analysis construction, evidence-figure assembly, and DOCX ZIP/heading/media audits.
- **Why it was changed:** The final report must be generated only from actual complete artifacts, show readable comparison tables, preserve unsupported/failed work, and fail rather than fill missing results.
- **Commands executed:** Bundled document-runtime compile checks; final gate check; progress DOCX regeneration; Open XML table/heading/media/section inspection; Optuna export contract test; orchestration unit tests.
- **Validation performed:** The final gate correctly reports every incomplete stage and writes no final artifact; the progress DOCX is a valid 2,379,232-byte package with 18 Heading 1 paragraphs, 207 Heading 2 paragraphs, eight real tables, 14 media assets, and one section; both new orchestration tests pass.
- **Outcome:** Final reporting is ready to run only after real completion. The progressive report includes the current DenseNet121 controlled evidence and readable tables. Visual page rendering remains unavailable and is not claimed.
- **Next step:** Continue Stage B and refresh the progressive artifacts after each additional architecture completes.

## 2026-07-16T20:02:00+08:00 - Guarded full-workflow continuation launched

- **Objective:** Continue automatically from the active baseline sequence through every authorized staged experiment, verification gate, and final report without duplicate training or premature test access.
- **Files inspected:** Stage entry points, run matrix statuses, ledger state, final-selection contracts, document runtime, active baseline process, and continuation failure/restart cases.
- **Files created or modified:** `run_research_pipeline.py`; `update_results_interpretation.py`; generic stage-runner guards; idempotent matrix-resolution logic; orchestrator log/status; implementation/reproduction/development documentation.
- **What was changed:** Added a stop-on-failure orchestrator that waits on ledger state, resumes incomplete stages, consolidates and documents each major stage, persists its own state, and invokes the final test only through the frozen-selection evaluator. Regenerated results interpretation directly from consolidated actual artifacts.
- **Why it was changed:** The multi-hour workflow must continue safely beyond the current interactive session while remaining observable, resumable, and leakage-controlled.
- **Commands executed:** Static compile/help checks; dry ledger-state query; wrong-entry-point failure-gate checks; hidden `Start-Process` launch with polling enabled.
- **Validation performed:** The orchestrator started as PID 19088, wrote status `waiting`, observed exactly one completed baseline and active `baseline_densenet169_s42`, and issued no duplicate training command. The stored resume command is exact.
- **Outcome:** DenseNet169 and the existing Stage B launcher remain the only GPU training activity. The continuation is waiting and will advance only after Stage B exits or perform one guarded resume pass if necessary.
- **Next step:** Monitor Stage B completion and the orchestrator state; inspect `pipeline_orchestrator.log` if any later gate stops.

## 2026-07-16T20:04:50+08:00 - Idle continuation reloaded after restart-safety improvements

- **Objective:** Ensure the waiting continuation uses the final idempotent resume logic and refreshed environment/report verification path.
- **Files inspected:** Orchestrator parent/child command lines, active training PIDs, persisted waiter state, and post-launch source differences.
- **Files created or modified:** Orchestrator process/state/log and this append-only correction entry.
- **What was changed:** Stopped only idle orchestrator PIDs 19088/19012 after verifying their command lines, then relaunched the same waiting command from the updated source as launcher PID 1676 with child PID 21200.
- **Why it was changed:** A process already running does not reload later source edits; the final waiter needed restart-safe tuning/test behavior and final environment refresh without changing the active training process.
- **Commands executed:** Targeted process inventory; targeted `Stop-Process`; hidden `Start-Process`; ledger/process/state verification; ten-test suite and updated environment snapshot.
- **Validation performed:** Baseline training PIDs 26196/10548 remained alive; the replacement waiter reports `waiting`, one of eight baselines complete, and DenseNet169 active. Ten unit tests pass, `pip check` passes, CUDA remains available, and Optuna 4.9.0 is recorded.
- **Outcome:** The continuation is running the latest guarded code and has issued no training command while Stage B is active.
- **Next step:** Allow the active benchmark to complete; the waiter will then advance through the documented stage gates.


## 2026-08-03T05:06:47+08:00 - Stage C preprocessing ablation completed

- **Objective:** Complete and verify the independent preprocessing comparison from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for the independent preprocessing comparison; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded the independent preprocessing comparison workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_benchmark.py --stage preprocessing_ablation --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/analyze_preprocessing_ablation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage C preprocessing ablation complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** CLAHE, global histogram equalization, and exploratory VIRIDIS runs completed against the existing DenseNet169, DenseNet201, and ResNet50 controls.
- **Next step:** Resolve the best completed architecture/preprocessing evidence without redundant retraining.


## 2026-08-03T05:06:58+08:00 - Stage D focused comparison completed

- **Objective:** Complete and verify the validation-resolved architecture/preprocessing comparison from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for the validation-resolved architecture/preprocessing comparison; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded the validation-resolved architecture/preprocessing comparison workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/analyze_preprocessing_ablation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage C preprocessing ablation complete"; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/resolve_run_matrix.py --stage focused; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_benchmark.py --stage focused_comparison --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage D focused comparison complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** Focused rows were explicitly skipped because all representative preprocessing combinations already had completed artifacts.
- **Next step:** Resolve the top three distinct architecture configurations for general Optuna tuning.


## 2026-08-03T08:18:08+08:00 - Stage E bounded Optuna tuning completed

- **Objective:** Complete and verify three validation-only general tuning studies from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for three validation-only general tuning studies; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded three validation-only general tuning studies workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/resolve_run_matrix.py --stage tuning; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top1 --parent-run-id ablation_densenet201_histogram_equalization_s42 --trials 4 --epochs 6; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top2 --parent-run-id baseline_densenet169_s42 --trials 4 --epochs 6; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top3 --parent-run-id ablation_resnet50_pseudo_color_s42 --trials 4 --epochs 6; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage E Optuna general tuning complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** All three studies persisted their trial states and selected one best six-epoch validation trial each; no test data was accessed.
- **Next step:** Freeze the selected hyperparameters and independently retrain seeds 42, 123, and 2026.


## 2026-08-03T13:03:52+08:00 - Stage E bounded Optuna tuning completed

- **Objective:** Complete and verify three validation-only general tuning studies from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for three validation-only general tuning studies; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded three validation-only general tuning studies workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top1 --parent-run-id ablation_densenet201_histogram_equalization_s42 --trials 4 --epochs 6 --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top2 --parent-run-id baseline_densenet169_s42 --trials 4 --epochs 6 --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_tuning.py --matrix-run-id tuning_top3 --parent-run-id ablation_resnet50_pseudo_color_s42 --trials 4 --epochs 6 --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage E Optuna general tuning complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** All three studies persisted their trial states and selected one best six-epoch validation trial each; no test data was accessed.
- **Next step:** Freeze the selected hyperparameters and independently retrain seeds 42, 123, and 2026.


## 2026-08-03T13:04:09+08:00 - Stage F multi-seed robustness completed

- **Objective:** Complete and verify independent full-budget seeds 42, 123, and 2026 from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for independent full-budget seeds 42, 123, and 2026; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded independent full-budget seeds 42, 123, and 2026 workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage E Optuna general tuning complete"; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/resolve_run_matrix.py --stage robustness; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/run_benchmark.py --stage robustness --config model/configs/final_selected.json --resume; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 6 - Stage F multi-seed robustness complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** The frozen selected configuration completed at all three seeds and its validation distribution was consolidated.
- **Next step:** Freeze the representative seed-42 checkpoint before the single locked-test evaluation.


## 2026-08-03T13:05:25+08:00 - Stage G frozen locked-test evaluation completed

- **Objective:** Complete and verify the single predeclared representative checkpoint evaluation from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for the single predeclared representative checkpoint evaluation; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded the single predeclared representative checkpoint evaluation workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/evaluate_multiclass.py --run-dir "C:\Users\JOHN MARK\Code\ai-research-main\model\results\runs\final_selected_s42" --split test --checkpoint best --allow-locked-test --final-selection "C:\Users\JOHN MARK\Code\ai-research-main\model\results\final_selection.json"; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/generate_gradcam_review.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/calibrate_final_model.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/measure_final_efficiency.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 7 - frozen single locked-test evaluation complete"
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** The locked test was evaluated once after selection was frozen; test metrics were consolidated separately from validation selection.
- **Next step:** Run final source, test, consistency, inference-artifact, and report verification.


## 2026-08-03T13:06:13+08:00 - Phase 10 final verification gates completed

- **Objective:** Complete and verify the complete pre-report verification workflow from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for the complete pre-report verification workflow; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded the complete pre-report verification workflow workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/measure_final_efficiency.py; "C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model/scripts/generate_progress_report.py --phase-label "Phase 7 - frozen single locked-test evaluation complete"; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/verify_source_immutable.py --workers 4; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" -m unittest discover -s model/tests -v; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/snapshot_multiclass_environment.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** Source immutability, tests, and final consolidation passed; completion-gated final report generation is authorized from actual artifacts.
- **Next step:** Generate and structurally verify the final Markdown, final DOCX, and final progressive DOCX.


## 2026-08-03T13:54:54+08:00 - Phase 10 final verification gates completed

- **Objective:** Complete and verify the complete pre-report verification workflow from persisted predecessor evidence.
- **Files inspected:** Run matrix, experiment ledger, per-run status/history/metrics, consolidated artifacts, and current documentation.
- **Files created or modified:** Per-run artifacts for the complete pre-report verification workflow; consolidated result tables; live interpretation; progressive report; orchestrator state/log.
- **What was changed:** Executed the guarded the complete pre-report verification workflow workflow and regenerated its machine-readable and human-readable evidence.
- **Why it was changed:** The staged research plan requires actual results, continuous documentation, crash recovery, and no test feedback before selection freezes.
- **Commands executed:** "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/verify_source_immutable.py --workers 4; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" -m unittest discover -s model/tests -v; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/snapshot_multiclass_environment.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/materialize_run_logs.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/consolidate_results.py; "C:\Users\JOHN MARK\Code\ai-research-main\.venv\Scripts\python.exe" model/scripts/update_results_interpretation.py
- **Validation performed:** Stage gates, subprocess return codes, per-run status, and consolidation consistency checks passed.
- **Outcome:** Source immutability, tests, and final consolidation passed; completion-gated final report generation is authorized from actual artifacts.
- **Next step:** Generate and structurally verify the final Markdown, final DOCX, and final progressive DOCX.

## 2026-10-13T15:04:15+08:00 - FT-011 terminal evidence and documentation update

- **Objective:** Execute the final fixed FT-011 learning-rate comparison and document its terminal decision against the recovered full-precision controls.
- **Evidence inspected:** FT-011 protocol, final decision, seed-42 metrics/history/status and initialization, frozen FT-004 selection, timestamped protection events, final telemetry summary, and preserved interrupted smoke records.
- **Implementation:** Added the bounded FT-011 runner, full batch-state recovery trainer, device-derived safety watchdog, visible live monitor, exact-gate analysis and 10,000-resample paired proxy-cluster bootstrap. Backbone/classifier parameter groups are disjoint and cover all trainable parameters; both rates are logged. The recovered ImageNet initialization, pipeline and runtime configuration were preserved except for the intended initial backbone learning rate.
- **Source/data repair:** Restored locked manifest CSV line endings to their original LF bytes and original hashes; assignments and labels did not change. Preserved prior smoke outputs before repeating the identical execution check.
- **Protection findings:** Initial smoke RAM pause and subsequent unexplained process disappearance are archived under `model/results/ft011/interrupted_checks/20261013_1349`. The fresh protected smoke passed at 13:52:09. The full run's required telemetry-file replacement failed with Windows access denial at 14:16:39; safe-boundary checkpoint recovery and a 122.547-second continuous healthy interval permitted resume at 14:18:45. This recorded monitoring interruption must not be hidden by the aggregate `monitoring_gaps` counter of zero. Actual overheating and thermal throttling were zero, with peak GPU temperature 51 C. CPU temperature was unavailable; automatic fan control and hardware protections were preserved.
- **Outcome:** Seed 42 completed through epoch 9 with epoch 5 selected by the unchanged checkpoint rule. Macro F1 0.8202782987500647, accuracy 0.834096109839817, balanced accuracy 0.8173146418282663, Viral recall 0.5829596412556054. Four screening gates failed; seeds 123 and 2026 were not run. Final decision: retain FT-004 and close this tuning phase.
- **Documentation changes:** Replaced stale `CURRENT_STATUS.md` facts and appended terminal records to `TRAINING_JOURNAL.md`, `DEVELOPMENT_LOG.md` and `DECISION_LOG.md`. Historical entries remain intact.
- **Selected system:** The unchanged frozen FT-004 package contains three original seed-42 DenseNet-201 checkpoints using VIRIDIS -> GE -> CLAHE, VIRIDIS -> GE, and GE, with equal probability averaging and Normal/Viral/Bacterial argmax.
- **Interpretation boundary:** The matched bootstrap comparisons describe validation uncertainty for the observed seed 42 and derived filename/hash clusters. They are not independent external validation or verified patient-level independence. No test data was used for FT-011 tuning and the live website was unchanged.
- **Next step:** Independently evaluate the frozen FT-004 system; no further parameter search is authorized by this terminal experiment.
