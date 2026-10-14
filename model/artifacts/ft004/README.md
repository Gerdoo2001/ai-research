# Frozen FT-004 research model

This package publishes the **three original seed-42 DenseNet-201 checkpoints** selected by the existing FT-004 freeze. No checkpoint was retrained or replaced. FT-011 did not satisfy its screening gates and did not replace FT-004.

| Member | Preprocessing order | Checkpoint SHA-256 |
| --- | --- | --- |
| 1 | VIRIDIS → GE → CLAHE | `266c2d66841a8013a92628fd364b9b9d9ef614100f44999711e5993624385e40` |
| 2 | VIRIDIS → GE | `b1e1c9445ff799fb634a2e8cd795ddf05f5fef1c56aae6779b63efff3e1551c9` |
| 3 | GE | `b13383f573bea5ede2ae226161753f764cd8cba148a02d2f9665b7d1a079d74a` |

GE means Global Histogram Equalization. CLAHE means Contrast Limited Adaptive Histogram Equalization. Each model produces Normal, Viral, and Bacterial probabilities, in that order. Their equal arithmetic mean is the final vector, and the largest probability determines the predicted class. There are no learned ensemble weights or binary routing stages in FT-004.

## Download and verify

The checkpoints total 657,342,824 bytes and use Git LFS. Run these commands from the repository root:

```powershell
git lfs install
git lfs pull --include="model/artifacts/ft004/*.pt"
.venv\Scripts\python.exe model/scripts/package_ft004_release.py --verify-only
```

`package_manifest.json` records sizes and hashes. `source_frozen_selection.json` preserves the original freeze byte for byte; its SHA-256 is `bbb65ee1202b2c34490cdb1047339795e5751f2c71edf6626d1b1f235acea57b`. `selection.json` is a portable derivative with relative member/config paths and original provenance. Original absolute training paths are retained only to verify historical configuration hashes. Inference does not open training manifests or require those original directories.

The verifier rejects missing, corrupt, or pointer-only checkpoints. To rebuild this publication from the original research checkout, `package_ft004_release.py` requires the original frozen selection and source files; it does not generate new weights.

## Run research inference

Use Python 3.12 and `model/requirements.multiclass-cu121.txt`, not the legacy TensorFlow requirements. Existing research environment:

```powershell
.venv\Scripts\python.exe model/scripts/predict_ft004.py --synthetic --device cpu
.venv\Scripts\python.exe model/scripts/predict_ft004.py --image "C:\path\to\your_xray.jpg" --device cpu
```

`--synthetic` is an execution check on an artificial image, not a diagnostic result or performance evaluation. CUDA inference is available with `--device cuda` when the matching runtime and hardware are available. All trained tensors come from the checkpoints; inference does not need a separate ImageNet download. Only load these verified, trusted PyTorch checkpoints: the historical checkpoint format uses Python serialization and must not be treated as safe for arbitrary untrusted files.

The final images are aspect-ratio-preservingly fitted and center cropped to 224 × 224, converted to float32, and ImageNet-normalized. Inference applies each member's saved deterministic preprocessing, without training flip or rotation. GE and CLAHE return replicated grayscale RGB channels; the three-operation pipeline does not retain a colourful VIRIDIS palette at its final input.

## Evidence and limitations

FT-004's three-seed mean validation results are macro F1 85.305%, accuracy 86.270%, balanced accuracy 84.928%, and Viral recall 66.517%. Those are design-repeatability summaries; the packaged deployment uses three original seed-42 networks. They are not external-test accuracy. Current FT-011 evidence and all comparison definitions are in `model/docs/fine_tuning/FT011_Final_Report_20261013.md`.

This publication does not alter the website, include the raw dataset, publish nonselected FT-011 weights, or establish clinical readiness. Independent evaluation of the frozen system is the next research step.
