# Pneumonia chest X-ray classification research

This repository contains the original application and a separate, manifest-driven three-class research pipeline for **Normal, Viral Pneumonia, and Bacterial Pneumonia**.

## Current selection — 13 October 2026

**FT-004 remains the frozen validation-selected research system.** It averages the class probabilities from three original seed-42 DenseNet-201 models:

1. VIRIDIS → Global Histogram Equalization (GE) → Contrast Limited Adaptive Histogram Equalization (CLAHE).
2. VIRIDIS → GE.
3. GE.

The FT-004 design was confirmed at seeds 42, 123, and 2026. Its three-seed mean validation macro F1 is **85.305%**, accuracy **86.270%**, and balanced accuracy **84.928%**. Inference uses three networks, not nine.

The final FT-011 differential-learning-rate experiment completed seed 42, selected epoch 5, stopped after epoch 9, and failed four of five prespecified screening gates. Its validation macro F1 was 82.028%. Seeds 123 and 2026 were not run, FT-004 was retained, and this tuning phase is closed.

## Documentation and selected model

- [Documentation index](model/docs/README.md): current reports, Google Docs links, historical evidence, and reproduction instructions.
- [FT-011 complete report](model/docs/fine_tuning/FT011_Final_Report_20261013.md): preparation, transformations, configuration, reasons, exact results, protection/recovery, and references.
- [Selected FT-004 package](model/artifacts/ft004/README.md): original checkpoint hashes, inference instructions, and provenance.
- [Current status](model/docs/CURRENT_STATUS.md).

The three selected checkpoints are stored with **Git LFS**. A checkout containing small LFS pointer files is not a downloaded model. From the repository root:

```powershell
git lfs install
git lfs pull --include="model/artifacts/ft004/*.pt"
```

Use Python 3.12 and the separate research dependencies in `model/requirements.multiclass-cu121.txt`. The original website and its binary predictor have not been replaced by this package. The dataset and nonselected recovery checkpoints are not included in this publication.

This is a research prototype, not a clinical diagnostic device. Validation was reused during development, and derived split groups are not verified patient identifiers. Earlier test scores for another configuration do not apply to FT-004. Independent external evaluation remains required.
