# Pneumonia Three-Class Research Documentation

This directory is the human-readable source of truth for the reproducible research pipeline that distinguishes:

0. Normal
1. Viral Pneumonia
2. Bacterial Pneumonia

The existing binary and segmentation code is preserved. New multiclass work is being added as a parallel, manifest-driven research pipeline. Metrics in these documents must come from persisted artifacts; unsupported, failed, interrupted, and unresolved work remains visible.

## Live documents

- `CURRENT_STATUS.md`: concise operational status and resume command.
- `DEVELOPMENT_LOG.md`: append-only chronological engineering log.
- `DECISION_LOG.md`: stable research and engineering decisions.
- `TRAINING_JOURNAL.md`: queued, running, epoch, checkpoint, resume, failure, and completion events.
- `ISSUES_AND_FIXES.md`: stable issue records and verified fixes.
- `00_existing_binary_pipeline_audit.md`: existing implementation audit.
- `01_dataset_audit.md`: dataset, label, duplicate, and leakage audit.
- `02_experiment_plan.md`: controlled staged experiment plan.
- `03_implementation_notes.md`: multiclass implementation notes.
- `04_testing_and_validation.md`: verification evidence.
- `05_results_interpretation.md`: interpretation of actual results only.
- `06_reproduction_guide.md`: exact Windows reproduction commands.

Machine-readable artifacts live under `model/results` and locked data manifests under `model/data_manifests`.

## Safety scope

This is a research classifier, not a clinically deployable medical device. External clinical validation, broader datasets, bias assessment, calibration review, and regulatory/medical review are required before any clinical use.
