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

## Current reports and frozen model — 13 October 2026

FT-011 completed seed 42 and failed four of five prespecified screening gates. Seeds 123 and 2026 were not run. FT-004 remains selected and this tuning phase is closed. The original website remains unchanged.

- [Complete updated fine-tuning Word report](fine_tuning/DenseNet201_VIRIDIS_GE_CLAHE_Fine_Tuning_Report.docx): FT-001 through FT-011, segmentation feasibility, classification changes, comparisons, explanations, and references.
- [Full research journal Word report](Pneumonia_Three_Class_Classification_Full_Journal_20261013.docx): original data/baseline/preprocessing work, explicitly historical earlier conclusions, and the current FT-004 decision.
- [Standalone FT-011 Word report](fine_tuning/FT011_Final_Report_20261013.docx) and [matching Markdown](fine_tuning/FT011_Final_Report_20261013.md): complete fixed configuration, gates, results, recovery, runtime, uncertainty, and references.
- [Fine-tuning Markdown journal](fine_tuning/DenseNet201_VIRIDIS_GE_CLAHE_Fine_Tuning_Journal.md).
- [FT-004 model package](../artifacts/ft004/README.md): the three original selected checkpoints, integrity hashes, and portable inference commands; large weights use Git LFS.
- [FT-011 protocol and outcome on Google Docs](https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit): the original native document updated in place.
- [Full research journal on Google Docs](https://docs.google.com/document/d/1zIHFrBz1PpBkKLieQvkj_6c4JlztsO23IC6fnQj-aUE/edit).
- [Complete FT-001 to FT-011 fine-tuning report on Google Docs](https://docs.google.com/document/d/1diTu8Vhr506Lsa6UsdjB_lYPT-R_6OiBHhx_FTacHn0/edit).
- [Standalone completed FT-011 report on Google Docs](https://docs.google.com/document/d/1t0aNgHgDE_dxFhKHI1rb_X25z9bH5VO3ko7-GvxljNY/edit).

The three new native Google Docs were published in the existing research Drive folder and their contents read back. The original native FT-011 record was edited in place with its reference table and source chip preserved. The previous Word uploads remain untouched as historical files. Native text is black; scientific example images retain their meaningful original colours.

The dated August reports are preserved as historical records. Earlier DenseNet-201 + GE test metrics must not be attributed to the later FT-004 ensemble. Current FT-004 and FT-011 comparisons use validation evidence, not an independent external cohort.

## Safety scope

This is a research classifier, not a clinically deployable medical device. External clinical validation, broader datasets, bias assessment, calibration review, and regulatory/medical review are required before any clinical use.
