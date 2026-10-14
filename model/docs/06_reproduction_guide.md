# Reproduction Guide

All commands below run from:

```text
C:\Users\JOHN MARK\Code\ai-research-main
```

Use Command Prompt or PowerShell as noted. Do not run commands from inside the immutable dataset directory.

## Create and Install the Isolated Environment

Command Prompt:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r model\requirements.multiclass-cu121.txt
.venv\Scripts\python.exe -m pip check
```

This environment is separate from the legacy `venv`. Do not replace or upgrade the legacy TensorFlow packages as part of this workflow.

## Check GPU and Capture Environments

```text
nvidia-smi
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
venv\Scripts\python.exe model\scripts\snapshot_environment.py
.venv\Scripts\python.exe model\scripts\snapshot_multiclass_environment.py
```

Long training stops when CUDA is unavailable unless a configuration explicitly enables CPU fallback.

## Audit the Repository and Dataset

```text
venv\Scripts\python.exe model\scripts\inventory_repository.py
venv\Scripts\python.exe model\scripts\audit_dataset.py --source model\datasets\chest_xray
venv\Scripts\python.exe model\scripts\build_manifests.py
py -3.12 model\scripts\plot_dataset_audit.py
venv\Scripts\python.exe model\scripts\verify_source_immutable.py
```

Before using regenerated manifests as the same experiment, verify manifest version `three_class_seed42_3b815cf769c8` and the hashes in `model/data_manifests/split_summary.json`.

## Prepare Pinned YOLO Classifiers

```text
.venv\Scripts\python.exe model\scripts\prepare_yolo_assets.py
```

This creates git-ignored `.research_cache` content and writes a tracked research audit at `model/results/yolo_support_audit.json`. YOLOv10 classification remains unsupported.

## Run Tests and GPU Smokes

```text
.venv\Scripts\python.exe -m compileall -q model\src\multiclass model\scripts model\tests
.venv\Scripts\python.exe -m unittest discover -s model\tests -v
.venv\Scripts\python.exe model\scripts\gpu_smoke_models.py --batch-size 1 --input-size 224
.venv\Scripts\python.exe model\scripts\train_multiclass.py --smoke-test --architecture densenet121 --preprocessing baseline --device cuda --run-id reproduction_smoke
```

## Generate Preprocessing Evidence and Resource Plan

```text
.venv\Scripts\python.exe model\scripts\generate_preprocessing_examples.py
.venv\Scripts\python.exe model\scripts\benchmark_training_resources.py
.venv\Scripts\python.exe model\scripts\build_run_matrix.py
```

Rebuilding the run matrix reads the pilot history; reproduce the pilot first if its run directory is absent:

```text
.venv\Scripts\python.exe model\scripts\train_multiclass.py --architecture densenet121 --preprocessing baseline --device cuda --batch-size 16 --epochs 1 --weights imagenet --run-id pilot_densenet121_full_epoch
```

## Train One Selected Baseline

```text
.venv\Scripts\python.exe model\scripts\train_multiclass.py --architecture densenet121 --preprocessing baseline --device cuda --batch-size 16 --epochs 12 --weights imagenet --seed 42 --run-id baseline_densenet121_s42
```

## Run or Resume the Entire Baseline Benchmark

```text
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage baseline
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage baseline --resume
```

Completed identical runs are skipped. A failed independent row is preserved and the runner continues.

For a specific interrupted run, use the `resume_command` in its `status.json`, for example:

```text
.venv\Scripts\python.exe model\scripts\train_multiclass.py --config model\results\runs\baseline_densenet121_s42\config.json --resume
```

## Evaluate Validation

```text
.venv\Scripts\python.exe model\scripts\evaluate_multiclass.py --run-dir model\results\runs\baseline_densenet121_s42 --split val --checkpoint best
```

## Evaluate the Locked Test

Do not run this until the final configuration is frozen in the decision log:

```text
.venv\Scripts\python.exe model\scripts\evaluate_multiclass.py --run-dir model\results\runs\FINAL_SELECTED_RUN --split test --checkpoint best --allow-locked-test
```

The explicit flag is mandatory. A test evaluation must never trigger retraining or selection changes.

## Consolidate Current Results

```text
.venv\Scripts\python.exe model\scripts\consolidate_results.py
```

The command writes summary, ranking, preprocessing, tuning, robustness, test, and consistency artifacts. Planned/unsupported/failed rows remain represented.

## Generate and Verify the Progress Word Report

```text
"C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model\scripts\generate_progress_report.py
```

The DOCX is saved to `model/docs/Pneumonia_Research_Progress_Report.docx`. LibreOffice is unavailable on the audited machine, so use the document-package heading/image/section/style audits or open it manually for visual page QA.

## Generate the Final Markdown and Word Reports

The final generator fails closed until all supported baselines, supported preprocessing ablations, three tuning studies, three robustness seeds, the frozen single locked-test evaluation, and the zero-error consistency audit are complete.

```text
"C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model\scripts\generate_final_report.py --check-only
"C:\Users\JOHN MARK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" model\scripts\generate_final_report.py
```

Successful generation writes `model/docs/Pneumonia_Three_Class_Classification_Final_Report.md`, the matching `.docx`, and `model/docs/final_report_docx_audit.json`. The DOCX structural audit checks ZIP integrity, headings, and embedded media; manual visual page QA remains required because unattended rendering is unavailable on the audited machine.

## Resolve and Run Later Stages

After every stage, consolidate first. Placeholder resolution refuses to proceed until its required predecessor results are complete.

```text
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage preprocessing_ablation --resume
.venv\Scripts\python.exe model\scripts\consolidate_results.py
.venv\Scripts\python.exe model\scripts\resolve_run_matrix.py --stage focused
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage focused_comparison --resume
.venv\Scripts\python.exe model\scripts\consolidate_results.py
.venv\Scripts\python.exe model\scripts\resolve_run_matrix.py --stage tuning
```

The tuning resolver writes exact parent-specific commands to `model/results/tuning_plan.json`. Each command follows this form:

```text
.venv\Scripts\python.exe model\scripts\run_tuning.py --matrix-run-id tuning_top1 --parent-run-id <resolved_run_id> --trials 4 --epochs 6
```

Continue an interrupted study by adding `--resume`. Optuna TPE uses validation macro F1 only, saves its SQLite database under `model/results/tuning/<study_id>/study.db`, and exports all completed, pruned, failed, interrupted, or pending trial states. It is general tuning and is not called SAS Tuning.

After all three studies:

```text
.venv\Scripts\python.exe model\scripts\consolidate_results.py
.venv\Scripts\python.exe model\scripts\resolve_run_matrix.py --stage robustness
.venv\Scripts\python.exe model\scripts\run_benchmark.py --stage robustness --config model\configs\final_selected.json --resume
.venv\Scripts\python.exe model\scripts\consolidate_results.py
.venv\Scripts\python.exe model\scripts\resolve_run_matrix.py --stage test
```

The last resolver freezes the selection from validation evidence before any test access. Use the exact command written to `model/results/selection_resolution.json`; its form is:

```text
.venv\Scripts\python.exe model\scripts\evaluate_multiclass.py --run-dir "<frozen_seed42_run_dir>" --split test --checkpoint best --allow-locked-test --final-selection model\results\final_selection.json
```

Test evaluation without both the explicit unlock and matching frozen selection file fails closed.

## Resume the Complete Guarded Workflow

To resume the full staged workflow from the currently active baseline process, use:

```text
.venv\Scripts\python.exe model\scripts\run_research_pipeline.py --from-stage post_baseline --wait-for-baselines --poll-seconds 60
```

The orchestrator writes `model/results/pipeline_orchestrator_status.json` and `pipeline_orchestrator.log`, stops at the first failed gate, and records the exact stage-specific resume command. It does not bypass the validation-only placeholder resolvers or the frozen-selection test gate. To resume a later failed stage, replace `post_baseline` with the `stage` value recorded in the status JSON.

## Current frozen FT-004 publication — 13 October 2026

The commands above describe earlier experimental stages, not authorization to reopen the now-closed tuning phase. FT-011 finished seed 42, failed four screening gates, and retained FT-004. No further confirmation seeds or locked-test evaluations were run for FT-011.

For a fresh Windows research checkout, use Python 3.12, Git LFS, and the multiclass requirements:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r model/requirements.multiclass-cu121.txt
git lfs install
git lfs pull --include="model/artifacts/ft004/*.pt"
.venv\Scripts\python.exe model/scripts/package_ft004_release.py --verify-only
.venv\Scripts\python.exe model/scripts/predict_ft004.py --synthetic --device cpu
```

The published selection resolves member files relative to `model/artifacts/ft004/selection.json`; it does not read the dataset. The synthetic check is not a validation result. For your own research image, replace `--synthetic` with `--image "C:\path\to\your_xray.jpg"`. The three original seed-42 members are averaged, and original website inference remains separate.

Current FT-011 report generation uses the bundled document Python with `model/scripts/update_ft011_documents.py`. `export_research_docs_word.ps1 -RepositoryRoot "C:\path\to\ai-research-main"` exports a private read-only Microsoft Word instance to PDF on this Windows host; `render_word_export_pages.py` uses the canonical document renderer with that PDF and bundled PDFium for page QA. Earlier statements that unattended rendering was unavailable apply to the earlier reporting phase, not this October workflow. Rendering needs desktop Microsoft Word; training and inference do not.

## Safety checks (all phases)

- Never modify `model/datasets/chest_xray`.
- Never load `test_manifest.csv` during selection.
- Never compare smoke/pilot metrics as architecture results.
- Verify the configuration and manifest hashes in every run.
- Keep both best and last checkpoints.
- Treat every output as research-only, not clinical evidence.
