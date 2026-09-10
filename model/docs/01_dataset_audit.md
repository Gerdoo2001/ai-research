# Dataset Audit and Locked Split Construction

**Audit completed:** 2026-07-16
**Source:** `model/datasets/chest_xray`
**Source policy:** read-only
**Locked manifest version:** `three_class_seed42_3b815cf769c8`

## Executive Summary

The physical source contains 17,591 files occupying 2,483,915,296 bytes. It is not a single clean dataset tree: it contains one canonical top-level `train/val/test` tree, a second nested copy, macOS resource-fork/metadata records, and four other metadata files. The audit identified 5,856 readable images in the canonical tree and excluded 32 redundant exact duplicates within that tree, leaving 5,824 unique validated images for modeling.

Three labels are derived only from verified folder and filename evidence: Normal, Viral, and Bacterial. The retained class counts are 1,579 Normal, 1,485 Viral, and 2,760 Bacterial. No label was inferred from pixels.

The supplied folder split was not retained. A conservative filename-derived grouping rule found 264 group identifiers spanning the original folders. These matches are a leakage risk, not proof of patient identity: the source does not provide authoritative patient metadata and some `personN` identifiers appear under both pneumonia etiologies. All retained images were therefore reassigned, without consulting image content or test metrics, using deterministic multi-label group balancing with seed 42. The locked manifests contain 4,076 training, 874 validation, and 874 test images. Exact image hashes and assigned group identifiers are disjoint across all three splits.

## Source Topology and Canonical Selection

| Physical role | Files | Treatment |
| --- | ---: | --- |
| Top-level canonical `train/val/test` | 5,856 | Validated and deduplicated |
| Nested `chest_xray/train/val/test` | 5,860 | 5,856 verified exact image copies excluded; 4 `.DS_Store` records excluded |
| `__MACOSX` metadata/resource forks | 5,871 | Excluded |
| Other unrecognized metadata | 4 | Excluded |
| **Total physical source** | **17,591** | Source left unchanged |

Canonical selection is path-based and then verified by content hashes. Files under the top-level split directories are the modeling candidates. Every corresponding nested image has the same SHA-256 digest as its top-level counterpart. Nothing was deleted, renamed, or moved in the source tree.

The complete physical-file audit is stored in `model/data_manifests/dataset_audit.csv`. Exclusion records are stored in `model/data_manifests/excluded_images.csv`.

## Label Derivation and Class Order

The class mapping is locked in `model/data_manifests/class_mapping.json`:

| Index | Class | Verified evidence |
| ---: | --- | --- |
| 0 | Normal | Canonical parent folder is `NORMAL` |
| 1 | Viral | Canonical parent folder is `PNEUMONIA` and filename contains `_virus_` |
| 2 | Bacterial | Canonical parent folder is `PNEUMONIA` and filename contains `_bacteria_` |

All 5,856 canonical image candidates had readable image content and valid label evidence. Before canonical deduplication, the class counts were 1,583 Normal, 1,493 Viral, and 2,780 Bacterial. After excluding 32 redundant exact canonical copies, the modeling set contains:

| Class | Images | Share |
| --- | ---: | ---: |
| Normal | 1,579 | 27.11% |
| Viral | 1,485 | 25.50% |
| Bacterial | 2,760 | 47.39% |
| **Total** | **5,824** | **100.00%** |

The bacterial class is the largest, but the imbalance is moderate enough to preserve all available unique examples and handle imbalance in the training configuration and evaluation metrics.

## Image Integrity, Modes, and Dimensions

All retained files were decoded and inspected for width, height, color mode, file size, SHA-256, and a 64-bit DCT perceptual hash.

- Image modes: 5,541 grayscale (`L`) and 283 RGB.
- Width: minimum 384, median 1,280, maximum 2,916 pixels.
- Height: minimum 127, median 886, maximum 2,713 pixels.
- Aspect ratio: minimum 0.83539, median 1.41608, maximum 3.37879.
- File size: minimum 5,441, median 95,839, maximum 2,414,342 bytes.

The few very short images were visually checked and are readable, tightly cropped chest radiographs rather than decode failures. They remain included. The pipeline must standardize channel count and size without altering the immutable source.

## Exact and Near-Duplicate Review

The audit reports 6,172 findings in `model/data_manifests/duplicate_report.csv`:

| Finding type | Count | Action |
| --- | ---: | --- |
| Verified nested exact copies | 5,856 | Excluded from modeling |
| Redundant exact duplicates within the canonical tree | 32 | Deterministic first path retained; other copies excluded |
| Original cross-split conservative group matches | 264 | Original split rejected; groups kept intact during reassignment |
| Nested metadata mismatch/missing counterpart | 4 | Excluded metadata records |
| Potential perceptual near-duplicate pairs | 16 | Flagged for review; not automatically excluded or unioned |

The perceptual check uses a 64-bit DCT pHash from a 32 by 32 grayscale representation. Candidates were reported when Hamming distance was at most 2, when a candidate shared the conservative group identifier, or when equal dimensions accompanied distance at most 3. The 16 retained candidates have different cryptographic hashes. Visual sampling showed distinct radiographs whose shared anatomy produced close pHashes. Because perceptual similarity alone cannot establish duplicate identity, these candidates remain in the dataset and are explicitly flagged rather than silently removed.

## Group Derivation and Leakage Limitations

No authoritative patient identifier file is present. The following conservative filename rules are used:

- Pneumonia names beginning with `personN` map to `pneumonia_person_N`.
- Normal names beginning with `IM-NNNN-` or `NORMAL2-IM-NNNN-` map to `normal_im_N`.
- Exact-hash-connected records are forced into the same split group.

This yields 2,789 assigned groups. Of the filename-derived groups, 979 contain both viral and bacterial filenames. This may reflect repeated subject imaging, identifier reuse between etiologies, or both. Accordingly, the audit does not claim that every matching identifier is a proven patient match. Keeping each conservative group intact is the safer leakage-control choice, at the cost of possibly grouping unrelated images and reducing split flexibility.

The original folder distribution was highly unsuitable for three-class selection: its validation directory contained only 8 Normal and 8 Bacterial images and no Viral image. In addition, 264 conservative group identifiers crossed original folders. The original test directory is therefore not treated as a leakage-free locked test set.

## Deterministic Split Construction

All 5,824 retained images were reassigned using deterministic greedy multi-label group allocation followed by eight local-improvement passes. The seed is 42 and target ratios are 70% training, 15% validation, and 15% test. Assignment used labels and group membership only; it did not use image pixels, predictions, losses, or test performance.

| Split | Normal | Viral | Bacterial | Total | Groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,105 | 1,039 | 1,932 | 4,076 | 2,326 |
| Validation | 237 | 223 | 414 | 874 | 228 |
| Test | 237 | 223 | 414 | 874 | 235 |

Post-construction checks found:

- 0 assigned group identifiers shared across splits.
- 0 SHA-256 image hashes shared across splits.
- All three classes present in every split.
- Exact total coverage: 5,824 assigned images with no retained image omitted or multiply assigned.

The test manifest is locked and must not be used for architecture, preprocessing, threshold, or hyperparameter selection. Only the final selected configuration may be evaluated on it.

## Locked Artifacts and Hashes

The manifest version hash is `3b815cf769c85658a04748da5eb2e15b981b89d9c9b77a404e90fe23473d2774`. Individual locked artifacts have these SHA-256 digests:

| Artifact | SHA-256 |
| --- | --- |
| `all_images_manifest.csv` | `07cba9e6b9c3e9d0152fd13663c7467ab3f6e94f4752d350e50edd7122646b51` |
| `class_mapping.json` | `4378d8f47704c3401c0c12d470b2b74a4004ec9049ed7fab8d80cc4777575e02` |
| `train_manifest.csv` | `96bf458d0a214df7a61c2449a0624a0790e7f10fce067ebd073c01c253dd9c7e` |
| `val_manifest.csv` | `bd23b46f554752459b588728722fb606a980a8415a510f21a2ab671fd334049e` |
| `test_manifest.csv` | `a91a1af8306f72e01b2c95b8003458dc578d3369966230ebcdb960f98f5268a7` |

The full physical source inventory hash is `4ee6dd784207a910043f62cff542a5d805c312fcddafa90ce1a058d03402f734`. A post-generation integrity pass rehashed all 17,591 source files and found zero missing, size-mismatched, or content-mismatched files.

## Quality and Reproducibility Limitations

- Labels are filename-derived dataset annotations, not independently adjudicated diagnoses.
- The source provides no authoritative patient metadata; group identity is conservative and uncertain.
- pHash is a screening heuristic and can collide on visually similar chest anatomy.
- Image acquisition, projection, equipment, institution, demographics, and clinical context are unavailable.
- Tight crops and heterogeneous grayscale/RGB encodings may introduce nuisance variation.
- These data and resulting models are for research only, not clinical diagnosis or deployment.

## Reproduction

From the repository root, use the repository TensorFlow environment for hashing/audit and the available plotting runtime for figures:

```text
venv\Scripts\python.exe model\scripts\audit_dataset.py --source model\datasets\chest_xray
venv\Scripts\python.exe model\scripts\build_manifests.py
py -3.12 model\scripts\plot_dataset_audit.py
venv\Scripts\python.exe model\scripts\verify_source_immutable.py
```

Rebuilding is an audit operation, not permission to overwrite or mutate the source dataset. Any regenerated manifest must match the locked version and hashes above before it can be used as the same experimental dataset.
