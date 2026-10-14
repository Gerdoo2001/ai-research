# DenseNet-201 + VIRIDIS -> GE -> CLAHE Fine-Tuning Journal

## Document status

- Document type: separate fine-tuning and ROI experiment record
- Created: 2026-08-10; historical confirmation finalized: 2026-08-13; updated: 2026-10-13
- Current status: FT-011 completed seed 42 and failed four of five prespecified screening gates. Its seeds 123 and 2026 were not run, FT-004 remains frozen, and this tuning phase is closed. FT-010 remains the simpler confirmed alternative and FT-009B the higher-Viral-recall alternative. No newly selected configuration has been evaluated on a genuinely independent external cohort.
- Model-development split: locked training and validation manifests
- Locked test status: not accessed
- Relationship to prior documentation: extension after completion of the controlled baseline and preprocessing-order experiments; it does not replace or rewrite those results
- Current artifact: `DenseNet201_VIRIDIS_GE_CLAHE_Fine_Tuning_Report.docx`; the dated August report remains preserved.

## Purpose

This journal records experiments intended to improve the selected DenseNet-201 configuration after preprocessing selection. The selected preprocessing order is:

1. VIRIDIS pseudo-coloring.
2. Global Histogram Equalization (GE).
3. Contrast Limited Adaptive Histogram Equalization (CLAHE).

The primary objective is to improve validation macro F1 and Viral/Bacterial discrimination. Validation accuracy is secondary. A configuration must not be selected solely because it increases accuracy while reducing Viral recall or balanced accuracy.

### How references are used in this journal

Every FT section now states both the reason the experiment was conducted and the reference basis that motivated the idea. A publication can justify testing a method, but it cannot prove that the method works on this pediatric three-class dataset. Exact settings such as 320 pixels, translation +/-3%, PSPNet threshold 0.20, crop margin 18%, focal gamma 2.0, and label smoothing 0.05 are project-defined experimental choices unless the section explicitly says otherwise. The final decision for each FT is based on the locked validation protocol and saved artifacts, never on a paper's reported accuracy. Sources from general image classification, adult chest X-rays, segmentation, or object detection are marked as partial or background evidence and are not presented as direct evidence for the final classifier.

## Separation from the original experiment documentation

The original experiment documentation covers:

- dataset auditing and manifest locking;
- duplicate control and patient/group separation;
- eight-model controlled baselines;
- preprocessing ablations and operation-order comparisons;
- the seed-42 selection of DenseNet-201 + VIRIDIS -> GE -> CLAHE;
- the three-seed robustness results for the selected pipeline.

This fine-tuning journal covers only post-selection experiments, including:

- alternate input resolution;
- optimizer and regularization tuning;
- staged freezing or differential learning rates, if implemented;
- loss-function tuning;
- lung segmentation, ROI cropping, or ROI masking;
- multi-seed confirmation of any improved candidate.

Results from these two experimental phases must remain distinguishable in tables and conclusions.

## Reference configuration: FT-REF-224-S42

| Field | Value |
|---|---|
| Architecture | DenseNet-201 |
| Pretrained weights | ImageNet |
| Preprocessing | VIRIDIS -> GE -> CLAHE |
| Input resolution | 224 x 224 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss | Balanced class-weighted cross-entropy |
| Augmentation | Horizontal flip p=0.5; random rotation up to +/-5 degrees, training only |
| Maximum epochs | 12 |
| Early-stopping patience | 4 |
| Seed | 42 |
| Validation macro F1 | 85.294% |
| Validation accuracy | 86.613% |
| Validation balanced accuracy | 84.546% |
| Validation Viral recall | 62.332% |
| Run ID | `staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42` |

The reference values above are completed validation results. They are the control against which the first fine-tuning experiment will be compared.

### Reference basis and why FT-REF exists

DenseNet-201 was retained because the completed eight-model and preprocessing-order experiments selected it empirically under validation macro F1. Huang et al. introduced DenseNet and describe its dense feature reuse and gradient-flow properties; this is architecture background, not evidence for the exact pneumonia configuration: [Huang et al., Densely Connected Convolutional Networks, CVPR 2017](https://arxiv.org/abs/1608.06993). No cited publication established the exact VIRIDIS -> GE -> CLAHE order, batch size 8, or the complete FT-REF hyperparameter set. Those values came from this project's controlled validation experiments, so FT-REF exists as the matched experimental control against which later one-variable changes can be interpreted.

## Experiment FT-001: 320 x 320 resolution

### Research question

Does increasing the DenseNet-201 input from 224 x 224 to 320 x 320 preserve additional radiographic detail and improve Viral/Bacterial discrimination when the selected VIRIDIS -> GE -> CLAHE preprocessing order is retained?

### Hypothesis

The 224 x 224 center crop may remove or compress subtle texture patterns. A 320 x 320 input contains approximately 2.04 times as many pixels and may improve classification, particularly Viral recall. Higher resolution may also increase overfitting and training cost, so the outcome must be determined experimentally.

### Reference basis and why FT-001 was conducted

Touvron et al. show that train/test image resolution and the scale of image content can materially affect classification, and they evaluate fine-tuning at a target resolution: [Touvron et al., Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423). This is background evidence from general image classification, not direct pediatric chest-X-ray evidence and not a prescription to use 320 x 320. FT-001 was conducted because 224 x 224 might compress subtle radiographic texture, while 320 x 320 was still feasible on the available RTX 2050. The matched ablation was necessary to learn whether the extra pixels helped this dataset; they did not.

### Controlled configuration

All reference settings remain fixed except input resolution and, only if required by measured GPU safety, physical batch size.

| Field | FT-001 value |
|---|---|
| Architecture | DenseNet-201 |
| Preprocessing | VIRIDIS -> GE -> CLAHE |
| Input resolution | 320 x 320 |
| Candidate batch sizes | 8, then 6, then 4 if required |
| Seed | 42 |
| Maximum epochs | 12 |
| Primary metric | Validation macro F1 |
| Secondary metrics | Accuracy, balanced accuracy, Viral recall, validation loss |
| Test access | Prohibited during tuning |

### Resource-safety procedure

After the active CLAHE multi-seed queue releases the GPU, the controller performs one real CUDA optimizer-step benchmark at batch sizes 8, 6, and 4 in descending order. The largest successful batch is recorded and used. No simultaneous full training runs are permitted on the 4 GB RTX 2050.

### Acceptance criteria

FT-001 will be considered promising when:

- validation macro F1 improves by approximately 0.5 percentage points or more over 85.294%;
- Viral recall improves or remains acceptably stable;
- balanced accuracy does not materially decline;
- the learning curves do not show substantially worse overfitting;
- the result is reproducible through seeds 123 and 2026 before final selection.

An isolated seed-42 improvement is a screening result, not final proof of superiority.

### Execution record

| Item | Location or status |
|---|---|
| Configuration | `model/configs/fine_tuning/densenet201_viridis_ge_clahe_320_s42.json` |
| Queue controller | `model/scripts/run_densenet201_320_queue.py` |
| Live progress | `model/results/densenet201_320_tuning/progress.json` |
| Final controller result | `model/results/densenet201_320_tuning/result.json` |
| Current state | Completed on 2026-08-10; early stopping ended training after epoch 10 of 12 |
| Selected batch size | 8; measured CUDA batch benchmark passed |
| Best epoch | 6 |
| Final validation macro F1 | 84.215% |
| Final validation balanced accuracy | 83.939% |
| Final validation accuracy | 85.469% |
| Final validation Viral recall | 63.229% |
| Training duration | 6,793.36 seconds (approximately 1 hour 53 minutes) |
| Locked test accessed | No |

### FT-001 conclusion

Increasing the resolution from 224 x 224 to 320 x 320 did not improve the selected pipeline. Relative to FT-REF-224-S42, validation macro F1 decreased by 1.079 percentage points and accuracy decreased by 1.144 percentage points. Viral recall increased by 0.897 percentage points, but that isolated gain did not compensate for the lower macro F1, balanced accuracy, and accuracy. The 224 x 224 reference remains preferred.

## Experiment FT-002: pretrained lung ROI segmentation audit

### Candidate model

The first implementation candidate is TorchXRayVision's ChestX-Det PSPNet anatomical segmentation model. It produces separate Left Lung and Right Lung output channels as part of a 14-structure 512 x 512 output.

### Reference basis and why FT-002 was conducted

Cohen et al. describe TorchXRayVision as an open chest-X-ray library with common preprocessing and pretrained models: [Cohen et al., TorchXRayVision: A library of chest X-ray datasets and models](https://arxiv.org/abs/2111.00595). The [official TorchXRayVision repository](https://github.com/mlmed/torchxrayvision) was used to verify the callable implementation and output contract. This is partial evidence: the available model is useful for research prototyping, but adult/general CXR performance does not guarantee reliable pediatric lung boundaries. FT-002 was therefore an audit before classification, not an assumption that cropping would improve accuracy. Its purpose was to determine whether a frozen lung ROI could reduce irrelevant borders and anatomy without removing pneumonia evidence.

### Proposed processing order

1. Open the original unenhanced grayscale X-ray.
2. Apply the segmentation model using its required normalization and resolution.
3. Combine the Left Lung and Right Lung masks.
4. Validate mask quality and add a conservative boundary margin.
5. Apply either an ROI bounding-box crop or an expanded lung mask to the original image.
6. Apply VIRIDIS -> GE -> CLAHE.
7. Resize/crop to the selected classification resolution.
8. Fine-tune and evaluate DenseNet-201 using the locked training and validation manifests.

Segmentation must precede VIRIDIS, GE, and CLAHE because the pretrained segmenter expects its native grayscale chest-X-ray input distribution.

### Required validation before classification training

- Review at least 50 to 100 representative pediatric images across Normal, Viral, and Bacterial classes.
- Record empty, fragmented, truncated, or anatomically implausible masks.
- Verify that peripheral disease patterns are not removed.
- Compare bounding-box cropping and expanded soft/binary masks separately.
- Do not assume adult-trained segmentation performance transfers to the pediatric dataset.

### Status

The official TorchXRayVision implementation and PSPNet output contract were verified on 2026-08-10. TorchXRayVision 1.5.2, ImageIO, and a NumPy-1.26.4-compatible scikit-image stack are installed in the project virtual environment; `pip check` reports no broken requirements. The official PSPNet weights were downloaded and recorded with SHA-256 `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc`. A real RTX 2050 inference produced the expected `[1, 14, 512, 512]` output, took approximately 1.10 seconds, and peaked at approximately 605 MiB of allocated GPU memory. The full audit evaluated 90 validation images, 30 per class. Seventy-nine passed the geometric checks and 11 were flagged, for an 87.78% automated pass rate. The user then reviewed all 90 original/overlay pairs. Sixty-nine were accepted and 21 were rejected, for a 76.67% manual acceptance rate. Manual class acceptance was Normal 100.00%, Viral 70.00%, and Bacterial 60.00%. Ten of 11 automated flags were manually rejected, but 11 automated passes were also manually rejected. PSPNet threshold 0.5 with an 8% crop margin is therefore rejected for ROI classification training as-is.

### FT-002B: threshold and crop-margin recalibration

### Reference basis and why FT-002B was conducted

FT-002B inherits the TorchXRayVision sources cited in FT-002. Its specific thresholds and margins were not copied from a publication. It was conducted because the initial threshold 0.50 and margin 8% failed manual review on 21 of 90 audited images. Reusing the already observed failures to choose a broader fixed policy was a tuning step only; that is why FT-002B was followed by the independent FT-002C audit rather than being reported as unbiased validation.

The next ROI step is a controlled visual recalibration using only the 21 validation-audit images rejected by the user. It does not train DenseNet-201, access the locked test, or authorize ROI cropping by itself. PSPNet probabilities are computed once per image, then three broader mask/crop policies are rendered beside the original image:

| Choice | Policy | Mask probability threshold | Crop margin | Purpose and tradeoff |
|---:|---|---:|---:|---|
| 1 | Mild expansion | 0.40 | 12% | Recover some weak lung pixels while adding modest protection around the mask. |
| 2 | Moderate expansion | 0.30 | 15% | Retain more lung boundary and peripheral opacity, with a larger amount of surrounding anatomy. |
| 3 | Aggressive expansion | 0.20 | 18% | Maximize anatomical retention, with the greatest risk of including background or non-lung structures. |
| N | None acceptable | N/A | N/A | Record that none of the tested policies creates a trustworthy mask/crop. |

All 21 side-by-side comparison images were generated on 2026-08-11. The source review record is cryptographically identified by SHA-256 `eabb934368270ae551d1b4d5da22cecfdf0db6f3fddc3190d724bf524cf3432d`. The persistent recalibration record is `model/results/roi_pspnet_recalibration/progress.json`. Each keyboard choice is saved immediately and the review resumes at the first pending image.

This 21-image set is a tuning set because the candidate thresholds were chosen after observing failures on these images. It cannot provide an unbiased pass rate for the selected policy. After the user completes the comparisons, the preferred single global policy must be frozen and evaluated on a new independent 90-image validation-only audit, stratified as 30 Normal, 30 Viral, and 30 Bacterial images that were not used in this recalibration. ROI classifier training remains prohibited unless that new audit passes the predefined automated and manual quality gates without class-dependent fallback behavior.

The completed FT-002B manual comparison selected the aggressive option for 14 of the 21 prior failures and selected `None acceptable` for seven. Mild and Moderate were never selected as the best option. The global policy was therefore frozen at threshold 0.20 and an 18% crop margin for independent testing. This preference count is not an acceptance-rate estimate because the interface recorded the single best option rather than a separate yes/no judgment for every candidate.

### FT-002C: independent frozen-policy audit

### Reference basis and why FT-002C was conducted

FT-002C also inherits the TorchXRayVision model source, while its independence rule comes from the project's bias-control protocol rather than a claimed clinical threshold. It was necessary because evaluating the selected threshold/margin on the same 21 images used to choose it would produce an optimistically biased estimate. FT-002C froze the policy first, excluded all earlier audit images, and then measured it on 90 new validation images. The project-defined 95% automated gate was an engineering guardrail, not a published standard.

FT-002C used selection seed 2026 to draw 30 new Normal, 30 new Viral, and 30 new Bacterial images from the validation manifest. All 90 paths from FT-002 were explicitly excluded before sampling. Post-run verification found zero relative-path overlap and zero cryptographic-image-hash overlap with the original audit. The locked test was not accessed.

The frozen PSPNet policy was applied without per-image adjustment: probability threshold 0.20, largest-component cleanup for each lung, and an 18% bounding-box margin. The automatic geometric audit produced:

| Class | Images | Automatic passes | Automatic flags | Automatic pass rate |
|---|---:|---:|---:|---:|
| Normal | 30 | 28 | 2 | 93.33% |
| Viral | 30 | 22 | 8 | 73.33% |
| Bacterial | 30 | 23 | 7 | 76.67% |
| Overall | 90 | 73 | 17 | 81.11% |

Before FT-002C, the project code defined a conservative engineering guardrail of at least a 95% automatic geometric pass rate plus non-empty bilateral masks. This 95% value was introduced by the project during audit design; it was not taken from a cited publication, clinical guideline, or user requirement. FT-002C did not satisfy that project-defined guardrail. Recorded flags included 12 union-area violations, six right-lung-area violations, three left-lung-area violations, three bounding-box-width violations, three bounding-box-height violations, and six fragmented-raw-mask warnings; one image can contribute more than one flag.

All 90 original/overlay pairs were then reviewed manually. The result was 82 accepted and eight rejected, for 91.11% overall manual acceptance:

| Class | Manually accepted | Manually rejected | Manual acceptance rate |
|---|---:|---:|---:|
| Normal | 30 | 0 | 100.00% |
| Viral | 28 | 2 | 93.33% |
| Bacterial | 24 | 6 | 80.00% |
| Overall | 82 | 8 | 91.11% |

Sixty-nine images were both automatic passes and manual acceptances; four automatic passes were manually rejected; 13 automatic flags were manually accepted; and four automatic flags were manually rejected. Thus the geometric rules were conservative for some images but also missed four failures. More importantly, eight genuine failures remained and six occurred in the Bacterial class, showing class-dependent reliability. Mixing cropped inputs for successful cases with full-image fallback for failures would create a preprocessing pattern correlated with the target class and is prohibited.

The manual review did not have a separately established 95% requirement. Its 91.11% overall acceptance and 80.00% Bacterial acceptance are reported as empirical limitations, not as failure against a literature-derived cutoff. The evidence does not establish clinical reliability, but it permits a carefully bounded research question: does uniformly applying the frozen crop improve or degrade downstream three-class validation performance? On 2026-08-11 the user authorized one controlled seed-42 classifier ablation. Every training and validation image must receive the same deterministic PSPNet policy; manually accepted/rejected status cannot select inputs, label-dependent fallback is prohibited, and the locked test remains unopened.

## Experiment FT-003: conservative spatial augmentation

### Research question

Can mild position and scale variation reduce overfitting and improve Viral/Bacterial discrimination while preserving the selected VIRIDIS -> GE -> CLAHE pipeline?

### Proposed controlled change

Retain the 224 x 224 input, DenseNet-201, seed 42, optimizer, loss, and preprocessing. Add only medically conservative affine variation during training: translation up to +/-3% of image width and height and scale jitter from 0.95 to 1.05, in addition to the existing horizontal flip and +/-5 degree rotation. Vertical flips, large rotations, aggressive crops, and transformations that can remove lung boundaries remain excluded.

### Reference basis, rationale, and gate

Ganesan et al. identify rotation and translation among traditional augmentation operations and study augmentation for chest-radiograph abnormality classification: [Ganesan et al., Assessment of Data Augmentation Strategies Toward Performance Improvement of Abnormality Classification in Chest Radiographs](https://lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9938.pdf). This is partial CXR evidence: it motivates testing spatial augmentation, but it does not establish the exact safe ranges for this pediatric dataset. FT-003 used only +/-3% translation and 0.95-1.05 scale because chest anatomy has a fixed orientation and aggressive transformations can remove or distort clinically relevant boundaries. Vertical flipping, large rotation, and aggressive cropping remained excluded.

The project-specific reason for FT-003 was that the reference run achieved its best result at epoch 6 and then stopped improving, which is consistent with overfitting. Mild spatial variation was tested as a way to expose the model to plausible acquisition-position changes. The ranges above were pre-registered in `model/configs/fine_tuning/densenet201_viridis_ge_clahe_mild_affine_s42.json`. All 14 pipeline contract tests passed, the historical winner's configuration hash remained unchanged, and the one-epoch limited-data GPU smoke run `smoke_densenet201_viridis_ge_clahe_mild_affine_s42` completed successfully on 2026-08-10. Its smoke metric is not an effectiveness result. The full seed-42 run `tuning_densenet201_viridis_ge_clahe_mild_affine_s42` ran from approximately 15:34 to 17:09 Asia/Manila and completed through early stopping after epoch 8. Its best checkpoint was epoch 4: macro F1 83.605%, balanced accuracy 83.178%, accuracy 84.325%, and Viral recall 65.471%. Macro F1 was 1.688 percentage points below the 85.294% reference, so FT-003 does not advance to multi-seed confirmation.

## Screening Experiment FT-004: equal-probability ensemble

### Reference basis and why FT-004 was conducted

Lakshminarayanan et al. present deep ensembles as a simple method that combines independently trained neural networks and can improve predictive performance and uncertainty behavior: [Lakshminarayanan, Pritzel, and Blundell, Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). This is general-method background and does not validate the exact three-member pneumonia ensemble. FT-004 was conducted because the project already had completed DenseNet-201 models whose preprocessing variants made different validation errors. Averaging their aligned probabilities tested whether those complementary errors could improve class balance without retraining another architecture. Because several combinations were screened, the selected membership had to be frozen and checked across seeds 42, 123, and 2026.

### Method

The saved validation probabilities from completed seed-42 runs were aligned by sample ID and averaged with equal weights. No model was retrained and the locked test was not accessed. The highest validation macro-F1 screening combination used three DenseNet-201 members with different preprocessing:

1. VIRIDIS -> GE -> CLAHE.
2. VIRIDIS -> GE.
3. GE alone.

The class with the highest mean probability was selected for each image.

### Screening result

| Metric | FT-004 ensemble | FT-REF single model | Difference |
|---|---:|---:|---:|
| Validation macro F1 | 86.953% | 85.294% | +1.659 points |
| Validation accuracy | 87.757% | 86.613% | +1.144 points |
| Validation balanced accuracy | 86.593% | 84.546% | +2.047 points |
| Validation Viral recall | 70.404% | 62.332% | +8.072 points |

### Interpretation boundary

This is the strongest improvement observed during the post-selection screening work, but it is not yet a final confirmed configuration. Multiple pair and three-member combinations were compared on the same validation set, so selecting the best combination can overfit the validation data. Before final selection, the ensemble membership and equal-weight rule must be frozen and confirmed using independent seeds or another pre-registered confirmation procedure. The tradeoff is approximately three DenseNet-201 inference passes per image and increased model storage.

### FT-004 confirmation queue

The frozen equal-probability ensemble is being confirmed using seeds 42, 123, and 2026. Seed 42 reuses completed predictions. Four missing member runs are queued sequentially on the single RTX 2050:

1. DenseNet-201 + VIRIDIS -> GE, seed 123.
2. DenseNet-201 + GE, seed 123.
3. DenseNet-201 + VIRIDIS -> GE, seed 2026.
4. DenseNet-201 + GE, seed 2026.

After these runs, `model/scripts/evaluate_validation_ensemble.py` aligns predictions by sample ID, verifies label alignment, calculates each seed's metrics, and reports the mean plus sample standard deviation. The persistent controller is `model/scripts/run_improvement_queue.py`; its state is `model/results/improvement_queue/progress.json`. The all-details terminal is `model/scripts/monitor_improvement_queue.ps1`. FT-002 automatically follows FT-004 and stops at the manual ROI approval gate.

### FT-004 confirmation result

| Seed | Macro F1 | Accuracy | Balanced accuracy | Viral recall |
|---:|---:|---:|---:|---:|
| 42 | 86.953% | 87.757% | 86.593% | 70.404% |
| 123 | 84.233% | 85.240% | 83.925% | 64.574% |
| 2026 | 84.730% | 85.812% | 84.267% | 64.574% |
| Mean | 85.305% | 86.270% | 84.928% | 66.517% |
| Sample standard deviation | 1.448 points | 1.320 points | 1.452 points | 3.366 points |

The corresponding single DenseNet-201 + VIRIDIS -> GE -> CLAHE pipeline averaged 84.168% macro F1, 85.202% accuracy, 83.932% balanced accuracy, and 64.275% Viral recall across the same three seeds. The ensemble therefore improved the fair three-seed averages by 1.137, 1.068, 0.996, and 2.242 percentage points respectively. However, the large seed-42 screening gain did not reproduce at the same magnitude in seeds 123 and 2026. The ensemble is a real average improvement, not a 90% solution, and it costs three inference passes per image.

## Experiment FT-005: frozen PSPNet ROI classification ablation

### Research question

When all other controls are inherited from the best confirmed single classifier, does a deterministic PSPNet lung bounding-box crop improve validation macro F1 without causing a severe class-level regression?

### Reference basis and why FT-005 was conducted

FT-005 uses the TorchXRayVision paper and official repository cited under FT-002 as the source for the pretrained PSPNet implementation. It was conducted only after FT-002B froze one threshold/margin policy and FT-002C measured its limitations on new images. The scientific question was whether uniformly focusing every development image on the predicted lung box would help the classifier despite imperfect masks. Before training, the full 4,950-image feasibility pass tested whether the same policy could actually produce bilateral crops for every training and validation image. The 116 class-dependent failures made a controlled classifier comparison unsafe, so the pre-registered fail-closed rule stopped the run. No publication was used to override that observed feasibility failure.

### Frozen one-variable comparison

| Component | Full-image comparator | FT-005 ROI candidate |
|---|---|---|
| Architecture | DenseNet-201 | DenseNet-201 |
| Classifier preprocessing | VIRIDIS -> GE -> CLAHE | VIRIDIS -> GE -> CLAHE |
| Input resolution | 224 x 224 | 224 x 224 |
| Seed | 42 | 42 |
| Batch size | 8 | 8 |
| Optimizer | AdamW | AdamW |
| Learning rate / weight decay | 0.0001 / 0.0001 | 0.0001 / 0.0001 |
| Augmentation | Horizontal flip 0.5; rotation +/-5 degrees | Same |
| Epoch / early-stop budget | 12 / patience 4 | Same |
| Image field | Full center-fitted image | PSPNet bbox from grayscale original; threshold 0.20; largest component per lung; 18% margin |

The segmentation is computed before VIRIDIS, GE, CLAHE, augmentation, resizing, tensor conversion, and ImageNet normalization. Bounding boxes are precomputed only for rows whose authoritative `assigned_split` is train or validation. No row assigned to the locked test split and no locked-test image is opened. The standard bundle-integrity check may hash the test-manifest CSV itself, but it does not load its image rows for this experiment. A source `relative_path` can begin with a legacy folder name such as `test/`; that historical folder prefix is not the experimental split. The crop coordinates, model-weight digest, package versions, source-image hashes, per-image mask statistics, and any failure are persisted. Precomputation must fail closed if either lung is empty; it must not silently substitute a full image.

### FT-005 execution record

| Item | Frozen value or artifact |
|---|---|
| Authorization | One controlled research-only seed-42 ablation; no clinical-reliability claim |
| Classifier configuration | `model/configs/fine_tuning/densenet201_viridis_ge_clahe_pspnet_roi_s42.json` |
| ROI coordinate builder | `model/scripts/build_pspnet_roi_bbox_manifest.py` |
| Restartable controller | `model/scripts/run_ft005_pspnet_roi.ps1` |
| Live monitor | `model/scripts/monitor_ft005_pspnet_roi.ps1` |
| ROI progress record | `model/results/roi_pspnet_bbox_precompute/progress.json` |
| Final ROI coordinate manifest | `model/data_manifests/roi_pspnet_t020_m018_train_val.json` |
| Full run ID | `fine_tuning_densenet201_viridis_ge_clahe_pspnet_roi_s42` |
| ROI rows expected | 4,076 training + 874 validation = 4,950 |
| Crop persistence | Coordinates only; no duplicate cropped-image dataset |
| Source integrity | Each image SHA-256 must match its locked-manifest digest |
| Implementation contracts | 14 configuration, manifest, transform, and ROI tests passed on 2026-08-11 |
| Historical comparability | The pre-ROI winner retained configuration hash `ef029d89d066687af73d2973be80caf5b16bb3d648000f617b77a73306ab2691` |
| Launch status | Completed 2026-08-11 Asia/Manila; feasibility gate failed closed before smoke/full classifier training |
| Locked test | No locked-test image loading or evaluation; no test metric used |

The first full-manifest failure occurred after 387 usable coordinate records: `test/PNEUMONIA/person69_virus_129.jpeg`, whose authoritative assignment is training, had a non-empty left mask but an empty right-lung mask at the frozen 0.20 threshold. The legacy `test/` folder prefix does not mean it belongs to the locked test split. The builder stopped as originally designed, persisted the records, and launched no classifier. It was then changed only to finish measuring the same frozen policy over all 4,950 development images while retaining the fail-closed decision at the end. It records empty masks as unusable and will not produce a training coordinate manifest or advance to classifier training if any such record exists. This continuation changes logging completeness, not the segmentation or crop policy.

### FT-005 feasibility result

The complete audit evaluated all 4,076 locked training assignments and all 874 locked validation assignments, for 4,950 development images. A laptop shutdown occurred after 4,162 records; the atomic progress record preserved those results, and execution resumed at the next unfinished image. No prior image inference had to be repeated beyond the normal checkpoint interval.

| Feasibility measure | Result |
|---|---:|
| Development images evaluated | 4,950 / 4,950 |
| Usable complete bilateral crops | 4,834 (97.657%) |
| Unusable under the frozen policy | 116 (2.343%) |
| Images with an empty left-lung mask | 56 |
| Images with an empty right-lung mask | 99 |
| Images with an empty union/no bounding box | 39 |
| Training-assignment failures | 93 / 4,076 (2.282%) |
| Validation-assignment failures | 23 / 874 (2.632%) |
| Classifier smoke test started | No |
| Full DenseNet-201 training started | No |
| Locked-test images accessed | No |

The empty-mask categories overlap: an image with both lungs empty contributes to the left-empty, right-empty, and union-empty counts. The class distribution was:

| Class | Development images | Unusable crops | Failure rate |
|---|---:|---:|---:|
| Normal | 1,342 | 0 | 0.000% |
| Viral | 1,262 | 48 | 3.803% |
| Bacterial | 2,346 | 68 | 2.899% |
| Overall | 4,950 | 116 | 2.343% |

Within training, 34 Viral and 59 Bacterial images failed. Within validation, 14 Viral and nine Bacterial images failed. No Normal image failed. This is a strong class-dependent preprocessing pattern: using a full-image fallback only for the 116 failures would reveal a crop-style cue that occurs only in pneumonia classes. Removing the 116 images would instead change the class distribution and discard difficult disease cases. Both actions would undermine the controlled comparison.

Therefore FT-005 failed its preprocessing-feasibility gate. The coordinate builder did not create `model/data_manifests/roi_pspnet_t020_m018_train_val.json`; the smoke-test and full-run directories were not created; and there is intentionally no FT-005 accuracy, macro F1, balanced accuracy, Viral recall, best epoch, or classifier training time. Writing zeros or reusing reference metrics would be false reporting. The outcome is a completed negative feasibility result, not an incomplete classification result.

The compact result artifact is `model/results/roi_pspnet_bbox_precompute/feasibility_summary.json`, SHA-256 `24d021ba1ca6d22fe8840b8c1e188c74785e71016c14be7eb822f83de4853b46`. It includes the frozen policy, PSPNet weight digest, package/runtime details, locked-manifest digests, every failure record, class/split counts, and the explicit no-training decision. The larger restartable record remains `model/results/roi_pspnet_bbox_precompute/progress.json`.

### Advancement rule

FT-005 begins with implementation tests and a limited smoke run, neither of which is an effectiveness result. The full seed-42 result is compared with the matched full-image reference: macro F1 85.294%, balanced accuracy 84.546%, accuracy 86.613%, and Viral recall 62.332%. Macro F1 is primary. The ROI candidate advances to seeds 123 and 2026 only if macro F1 improves and the per-class report shows no severe regression. Otherwise the PSPNet crop is stopped and retained as a documented negative or inconclusive ablation.

## Experiment FT-006: clean balanced focal-loss ablation

### Research question

When the selected DenseNet-201 architecture and VIRIDIS -> GE -> CLAHE image pipeline remain fixed, does focal loss improve the difficult Viral/Bacterial decisions relative to balanced cross-entropy?

### Why this is a new controlled experiment

Earlier automated tuning trials included focal loss, but those trials simultaneously changed the optimizer to SGD, the learning rate, weight decay, and sometimes the scheduler or label smoothing. Their lower scores cannot isolate the effect of focal loss. FT-006 changes exactly one conceptual factor: the training loss. Lin et al. introduced focal loss to down-weight well-classified examples and focus optimization on hard examples: [Lin et al., Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). This is background evidence from object detection, not direct proof for three-class chest-X-ray classification. Gamma 2.0 was a project-defined test setting, and balanced class weights were retained so the only conceptual change from FT-REF was the loss formulation.

### Frozen one-variable configuration

| Field | FT-REF-224-S42 | FT-006-FOCAL-S42 |
|---|---|---|
| Architecture | DenseNet-201 | Same |
| ImageNet initialization | Yes | Same |
| Preprocessing | VIRIDIS -> GE -> CLAHE | Same |
| Input / batch | 224 x 224 / 8 | Same |
| Augmentation | Horizontal flip p=0.5; rotation +/-5 degrees | Same |
| Optimizer | AdamW | Same |
| Learning rate | 0.0001 | Same |
| Weight decay | 0.0001 | Same |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 | Same |
| Maximum epochs / early stop | 12 / patience 4 | Same |
| Class weighting | Balanced | Same |
| Label smoothing | 0.0 | Same |
| Seed | 42 | Same |
| Loss | Balanced cross-entropy | Balanced focal loss, gamma 2.0 |
| ROI | None | None |

### Execution and advancement rule

The frozen configuration is `model/configs/fine_tuning/densenet201_viridis_ge_clahe_focal_s42.json`; the full run ID is `fine_tuning_densenet201_viridis_ge_clahe_focal_s42`. Contract tests and a 12-training/12-validation smoke run must pass before full training. The smoke result tests execution only and cannot be interpreted as effectiveness.

FT-006 is compared against the matched seed-42 reference: macro F1 85.294%, balanced accuracy 84.546%, accuracy 86.613%, and Viral recall 62.332%. It advances to seeds 123 and 2026 only if macro F1 improves by approximately 0.5 percentage points or more, Viral recall improves or remains acceptable, and no class has a severe recall regression. The locked test remains unopened throughout selection.

### FT-006 execution record

| Item | Value |
|---|---|
| Configuration hash | `456f3b807e07db42447b85e68487e28014d25e7631d206add82e0c876ecfb95a` |
| Contract tests | 17 passed on 2026-08-11 |
| Smoke run ID | `smoke_ft006_densenet201_focal_s42` |
| Smoke result | Completed one limited-data epoch; execution check only |
| Full run ID | `fine_tuning_densenet201_viridis_ge_clahe_focal_s42` |
| Full-run execution | Completed all 12 epochs from 2026-08-11 13:00 to 14:49 Asia/Manila |
| Best epoch | 11 |
| Training duration | 6,515.28 seconds (approximately 1 hour 48 minutes 35 seconds) |
| Live monitor | `model/scripts/monitor_ft006_focal_loss.ps1` |
| Full artifacts | `model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_focal_s42` |
| Locked test | Not accessed |

### FT-006 result and matched comparison

The final metrics were recomputed from the saved best epoch-11 checkpoint on all 874 locked validation assignments.

| Metric | Cross-entropy reference | FT-006 focal loss | Difference |
|---|---:|---:|---:|
| Macro F1 | 85.294% | 83.917% | -1.377 points |
| Balanced accuracy | 84.546% | 83.844% | -0.702 points |
| Accuracy | 86.613% | 84.439% | -2.174 points |
| Viral recall | 62.332% | 67.713% | +5.381 points |
| Viral precision | 82.738% | 70.892% | -11.846 points |
| Viral F1 | 71.100% | 69.266% | -1.834 points |
| Bacterial recall | 92.995% | 85.507% | -7.488 points |
| Normal recall | 98.312% | 98.312% | 0.000 points |

Focal loss found 151 of 223 Viral cases, compared with 139 for the reference. However, it also mislabeled 59 Bacterial images as Viral, compared with 26 under cross-entropy. That increased Viral recall but sharply reduced Viral precision and Bacterial recall. The net effect was lower macro F1, balanced accuracy, and overall accuracy. The training loss continued falling through epoch 12 while validation performance fluctuated, indicating that additional epochs alone would not justify overriding the saved best checkpoint.

FT-006 therefore does not advance to seeds 123 and 2026. This is a completed negative one-variable ablation: balanced focal loss gamma 2.0 changed the error tradeoff in favor of Viral sensitivity, but it was inferior under the pre-registered primary metric and overall class balance.

## Experiment FT-007: clean label-smoothing ablation

### Research question

When DenseNet-201, VIRIDIS -> GE -> CLAHE, balanced cross-entropy, and every training control remain fixed, does mild label smoothing improve validation macro F1 by preventing excessively confident class predictions?

### Reference basis and why FT-007 was conducted

Szegedy et al. describe label smoothing as a regularization component for image classification: [Szegedy et al., Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). This is general image-classification evidence, not direct evidence for pediatric pneumonia or for the value 0.05. FT-007 was selected as the next experiment because FT-006's focal loss shifted too many Bacterial cases into Viral: Viral recall increased, but Viral precision, Bacterial recall, macro F1, balanced accuracy, and accuracy declined. Label smoothing is a gentler test: it retains cross-entropy and only softens the target distribution slightly. Earlier automated trials that included label smoothing also changed other fields, so they could not isolate its effect on the current winner.

### Frozen one-variable configuration

| Field | FT-REF-224-S42 | FT-007-LS005-S42 |
|---|---|---|
| Architecture | DenseNet-201 | Same |
| ImageNet initialization | Yes | Same |
| Preprocessing | VIRIDIS -> GE -> CLAHE | Same |
| Input / batch | 224 x 224 / 8 | Same |
| Augmentation | Horizontal flip p=0.5; rotation +/-5 degrees | Same |
| Optimizer | AdamW | Same |
| Learning rate / weight decay | 0.0001 / 0.0001 | Same |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 | Same |
| Maximum epochs / early stop | 12 / patience 4 | Same |
| Loss and class weighting | Balanced cross-entropy | Same |
| Label smoothing | 0.00 | 0.05 |
| Seed | 42 | Same |
| ROI | None | None |

### Execution plan, artifacts, and advancement rule

The frozen configuration is `model/configs/fine_tuning/densenet201_viridis_ge_clahe_label_smoothing005_s42.json`, configuration hash `d39fa041fab58fa14fbd1b3f96c53def36b1c4d04bbf5c03af22533dca4f2245`, and full run ID `fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42`. Eighteen configuration, manifest, transform, and metric contracts passed on 2026-08-11, including an explicit requirement that label smoothing remain within [0,1) and preservation of the historical winner hash. The 12-training/12-validation ImageNet smoke run `smoke_ft007_densenet201_label_smoothing005_s42` completed successfully at 16:10 Asia/Manila; its metric verifies execution only and is not an effectiveness result. Full seed-42 training then completed all 12 epochs in 7,714.52 seconds (2 h 8 min 34.52 s). The saved best checkpoint is epoch 10. Full artifacts are in `model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42`.

FT-007 is compared with the matched seed-42 FT-REF metrics: macro F1 85.294%, balanced accuracy 84.546%, accuracy 86.613%, and Viral recall 62.332%. Its best checkpoint reached macro F1 85.631%, balanced accuracy 85.075%, accuracy 86.384%, and Viral recall 66.816%. Relative to FT-REF, macro F1 improved by 0.337 percentage points, balanced accuracy improved by 0.529 points, Viral recall improved by 4.484 points, and accuracy decreased by 0.229 points. Because the primary-metric gain was below the pre-registered approximate +0.5-point advancement threshold, FT-007 does not advance to seeds 123 and 2026. This is a useful positive-but-insufficient seed-42 ablation, not a new confirmed winner. The locked test remains unopened.

## Experiment FT-008: two-model U-Net lung-ROI screening

### Research question

Can either existing local U-Net artifact produce more reliable, class-balanced pediatric lung masks and proposed ROI crops than the previously audited PSPNet policy, without opening the locked test or starting another classifier prematurely?

### Reference basis and why FT-008 was queued

Ronneberger et al. introduced the U-Net encoder-decoder architecture and its skip connections for biomedical segmentation: [Ronneberger, Fischer, and Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Liu et al. evaluated an EfficientNet-B4-based U-Net family on difficult chest X-rays and reported that severe opacity, occlusion, and lung deformation can reduce segmentation performance: [Liu et al., Automatic lung segmentation in chest X-ray images using improved U-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC9127108/). A pediatric multi-center study reported a TransResUNet method for children's chest X-rays, supporting the need to consider pediatric anatomy rather than assuming adult-model transfer: [Zhang et al., Development of lung segmentation method in x-ray images of children based on TransResUNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10365102/).

These papers motivate testing U-Net, but none validates the two local HDF5 files or proves that ROI preprocessing will improve the current classifier. FT-008 was queued because PSPNet failed on 116 of 4,950 development images and every failure occurred in a Viral or Bacterial case. A different segmentation family must first demonstrate complete and class-balanced mask behavior before any downstream DenseNet comparison is scientifically defensible.

### Frozen local artifacts and provenance boundary

| Candidate | Artifact | SHA-256 | Verified contract | Provenance limitation |
|---|---|---|---|---|
| Model A | `model/outputs/cxr_reg_weights.best.hdf5` | `7bf4468a2d585b1781c41e4f4588ff2887826c681c3e77125378b7e4bb1dfaaf` | Classic U-Net topology; grayscale 256 x 256 input; one-channel mask output | Weights only; no in-repository training manifest, split, history, or validated metric |
| Model B | `model/outputs/trained_model.hdf5` | `9406564a140aef8ecf6c6acf20a3b9477d416a3da84e9f6e324c6dec1072c021` | Complete legacy Keras model; grayscale 256 x 256 input and output | Training source and validation provenance are not recorded |

Neither artifact is called a validated pediatric model. The audit records exact hashes and treats both as frozen screening candidates. The separate EfficientNet-B4 U-Net source under `model/src/unet/model.py` is not queued for training because its expected final weight file and mask-training dataset are absent, and its legacy training script is not reproducible as written.

### FT-008A supersession and corrected FT-008A2 protocol

The first 90-image FT-008A inference pass completed, but its postprocessing kept only the single largest connected component from the combined binary prediction. A combined lung mask can legitimately contain two disconnected regions—left and right lungs—so this rule could discard an entire lung. Its automatic pass/fail counts are therefore invalid for scientific model selection and are not reported as findings. The original files remain preserved under `model/results/roi_unet_screening`, with the correction recorded in `SUPERSEDED.md`; they are audit history only.

FT-008A2 corrects exactly this implementation defect by retaining up to the two largest connected components before morphological closing, contour filling, and bounding-box construction. A synthetic bilateral-mask contract confirms that two large lung-like regions are preserved while a smaller fragment is removed; a separate regression contract confirms that one-component mode reproduces the superseded behavior. This correction does not make the masks anatomically valid by itself, so the run still stops for human review.

1. Confirm that FT-007 has reached a terminal completed status, then allow a GPU-release cooldown so PyTorch DenseNet and TensorFlow U-Net do not share the 4 GB GPU.
2. Run seven CPU-safe geometry, cohort, artifact, and bilateral-component contracts before launch.
3. Reuse the exact 90 validation-only images from FT-002C: 30 Normal, 30 Viral, and 30 Bacterial. This provides a matched PSPNet comparison, but it is a screening cohort already used in prior work and is not a new independent confirmation cohort.
4. For both U-Nets, apply center-square grayscale preparation, resize to 256 x 256, convert to float32 in [0,1], threshold at 0.50, keep up to the two largest connected components, fill/close the retained mask regions, and propose a bounding box with an 18% margin.
5. Save the binary masks plus one three-panel image per case: original, Model A overlay/crop, and Model B overlay/crop.
6. Record empty masks, mask area, fragmentation, bounding-box geometry, automatic flags, per-class counts, model/runtime details, and every source-image identity.
7. Stop at `awaiting_manual_review`. Automated geometry is not anatomical accuracy, and no Dice score may be claimed because this screening cohort has no project-held pixel-level ground-truth masks.
8. Do not start ROI classification, remove failed images, apply class-dependent fallback, or access the locked test.

### Queue implementation and next gate

| Item | Location or status |
|---|---|
| Persistent queue controller | `model/scripts/run_ft008_unet_queue.py` |
| Queue launcher | `model/scripts/run_ft008_unet_queue.ps1` |
| Live monitor | `model/scripts/monitor_ft008_unet_queue.ps1` |
| U-Net audit implementation | `model/scripts/audit_unet_lung_roi.py` |
| CPU-safe contracts | `model/tests/test_ft008_unet_audit.py`; seven passed before corrected queue launch |
| Superseded FT-008A audit | `model/results/roi_unet_screening`; preserved for audit history and explicitly excluded from selection |
| Corrected queue progress | `model/results/ft008a2_unet_queue/progress.json` |
| Corrected audit progress | `model/results/roi_unet_screening_two_components/progress.json` |
| Corrected audit summary | `model/results/roi_unet_screening_two_components/summary.json` |
| Current status | FT-008A superseded; corrected FT-008A2 completed 90/90 automatic and manual reviews; both U-Nets received `do_not_advance`; no ROI classifier training authorized |

### Corrected FT-008A2 automatic screening result

Corrected inference ran from 22:34:57 to 22:35:35 Asia/Manila on 2026-08-11. The summary records 90/90 validation images completed, `max_components = 2`, no locked-test access, and no classifier training. The automatic geometry rules produced the following screening counts:

| Candidate | Overall automatic pass | Empty masks | Normal pass | Viral pass | Bacterial pass |
|---|---:|---:|---:|---:|---:|
| Classic U-Net weights | 15/90 (16.67%) | 4 | 14/30 (46.67%) | 1/30 (3.33%) | 0/30 (0.00%) |
| Legacy full U-Net | 2/90 (2.22%) | 28 | 2/30 (6.67%) | 0/30 (0.00%) | 0/30 (0.00%) |

The corrected component rule successfully permits bilateral regions; for example, reviewed Normal comparisons show the Classic U-Net retaining both lungs. However, the extreme class imbalance in automatic passes is a major transfer warning. Spot checks of Viral and Bacterial comparisons show partial-lung masks, unilateral masks, and tiny off-lung regions, especially from the legacy model. The Classic U-Net is clearly the stronger of these two candidates, but neither is approved for ROI classifier training from these automatic counts. Manual review remains required because area and bounding-box rules cannot establish anatomical correctness, and no Dice or intersection-over-union metric can be computed without pixel-level ground-truth masks.

### FT-008A2 persistent manual-review procedure

The manual-review record was initialized on 2026-08-12 at `model/results/roi_unet_screening_two_components/manual_review/progress.json`. The source summary is bound into that record by SHA-256 `cc3f286ba6871f8bc177702d4d7d6c9badc3da1ce8b63e4fff1a72151c355fd0`. Each screen shows the original image, Classic U-Net overlay/crop, and Legacy U-Net overlay/crop. One of four mutually exclusive decisions is saved atomically after each image: Classic only, Legacy only, both, or neither. This preserves separate accept/reject counts for each model while requiring only one review action per image. The interface resumes at the first pending image, supports returning to a prior image, and allows a safe pause without losing completed decisions.

Four manual-review contracts passed before launch: model-specific decision mapping, per-model/per-class summary accounting, verification of the complete corrected 90-image source, and existence of all 90 comparison images. A fifth contract verifying the automatic-versus-manual agreement table passed before finalization. No automatic flag was treated as a human answer. Approval required the predicted mask to follow the visible lungs and the proposed yellow crop to retain both lungs without an anatomically harmful cut. Locked-test access and ROI classifier training remained prohibited throughout this review.

### FT-008A2 completed manual-review result and decision

All 90 images were reviewed on 2026-08-12. The decisions were Classic only for 8 images, Legacy only for 0, both for 2, and neither for 80. The finalized record is `model/results/roi_unet_screening_two_components/manual_review/summary.json`; it is bound to the complete progress record by SHA-256 `754b592f68ac5617330f2a9b439f5231207d432fbd3b7f3573a7bc9b690d02b0`.

| Candidate | Manual accept | Manual reject | Normal accept | Viral accept | Bacterial accept |
|---|---:|---:|---:|---:|---:|
| Classic U-Net weights | 10/90 (11.11%) | 80/90 (88.89%) | 10/30 (33.33%) | 0/30 (0.00%) | 0/30 (0.00%) |
| Legacy full U-Net | 2/90 (2.22%) | 88/90 (97.78%) | 2/30 (6.67%) | 0/30 (0.00%) | 0/30 (0.00%) |

For the Classic U-Net, 10 automatic passes were manually accepted, five automatic passes were manually rejected, and none of the 75 automatically flagged masks was accepted. For the Legacy U-Net, both automatic passes were manually accepted and none of its 88 automatically flagged masks was accepted. Thus, the geometry flags were conservative enough not to hide a manually acceptable flagged case in this cohort, but five Classic automatic passes were false assurances anatomically; automatic rules cannot replace the human review.

Neither model produced a single manually acceptable Viral or Bacterial mask/crop. This is not a small reduction in average quality: it is complete class-dependent failure on the two disease classes that the classifier must distinguish. Both candidates therefore receive `do_not_advance`. No new independent U-Net cohort, complete-development ROI processing, or DenseNet ROI classifier training is authorized. These manual acceptability counts are not Dice or intersection-over-union measurements because project-held pixel-level ground truth is unavailable, and they are not locked-test results.

Because neither U-Net advanced, the planned new independent U-Net cohort, complete 4,950-image feasibility pass, and DenseNet ROI ablation are cancelled for these artifacts. Reusing either mask generator would require a new artifact with documented provenance and a new audit registration; it cannot be resumed from the rejected candidates.

## Experiment FT-009A: matched validation-error analysis

### Research question and authorization boundary

Do FT-REF, FT-007 label smoothing, and the frozen FT-004 seed-42 ensemble fail on the same locked-validation images, and are their remaining errors dominated by the Normal/Pneumonia boundary or by Viral/Bacterial etiology? FT-009A is analysis of saved prediction files only. It does not train a model, change labels, access the locked test, or authorize FT-009B training.

### Reference basis and why FT-009A was conducted

A pediatric CXR study explicitly evaluated two scenarios—Pneumonia versus Normal and Bacterial versus Viral—supporting examination of whether this dataset also separates naturally into those tasks: [Computer-assisted lung diseases detection from pediatric chest radiography using long short-term memory networks](https://doi.org/10.1016/j.compeleceng.2022.108402). A separate pediatric study trained an image model specifically to distinguish Viral from Bacterial pneumonia, providing direct/partial evidence that etiology discrimination is a distinct modeling target: [Image-based deep learning in diagnosing the etiology of pneumonia on pediatric chest X-rays](https://doi.org/10.1002/ppul.25229). A pediatric pneumonia generalizability study motivates retaining strict validation and external-validity boundaries: [Limited generalizability of deep learning algorithm for pediatric pneumonia classification on external data](https://pubmed.ncbi.nlm.nih.gov/34648114/).

These publications justify studying the two decision boundaries, but they do not establish that a hierarchical classifier will improve this project. They also do not justify the exact project-defined design gate below. FT-009A therefore analyzes matched per-image behavior before any new configuration is drafted.

### Frozen source artifacts and cohort contracts

| System | Saved prediction artifact | SHA-256 | Interpretation role |
|---|---|---|---|
| FT-REF DenseNet-201 | `model/results/runs/staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42/predictions.csv` | `10c2cd4466049389ee51f495e8c0c7176616ae7ac03a38b4174edc3307491007` | Matched seed-42 single-model reference |
| FT-007 label smoothing 0.05 | `model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42/predictions.csv` | `b20b0e0097ef3aac0d2626a39ee6fe1dfda8a525d6c298d5a026fab7508cb150` | Matched seed-42 one-variable ablation |
| FT-004 frozen ensemble seed 42 | `model/results/ensemble_confirmation/ensemble_predictions_s42.csv` | `ba1dfb653c0093396677576e15c9e77d4dc22945f37123a0c765a8bd1d1351de` | Matched seed-42 ensemble behavior; not the three-seed mean |

Five contracts passed before the completed analysis: all three hashes matched, every file contained 874 unique rows, sample IDs/relative paths/true labels were identical, probability vectors were valid and agreed with saved predictions, label mismatches fail closed, and duplicate sample IDs fail closed. Relative paths can begin with `train/` or `test/` because those are legacy folders in the downloaded source dataset; the locked project manifest assignment, not that folder prefix, defines all 874 rows as validation data.

### Pre-run FT-009B design gate

Before computing the FT-009A findings, the project registered a design-only gate: at least two of the three systems must have (1) Viral/Bacterial mistakes constituting at least 70% of their three-class errors and (2) Normal-versus-Pneumonia binary balanced accuracy exceeding Viral-versus-Bacterial binary balanced accuracy by at least 10 percentage points. These are project-defined diagnostic thresholds, not published clinical standards. Passing authorizes only drafting and contract-testing FT-009B; it does not authorize training, multi-seed escalation, or locked-test access.

### Matched three-class results

| System | Macro F1 | Balanced accuracy | Accuracy | Normal recall | Viral recall | Bacterial recall |
|---|---:|---:|---:|---:|---:|---:|
| FT-REF | 85.294% | 84.546% | 86.613% | 98.312% | 62.332% | 92.995% |
| FT-007 label smoothing | 85.631% | 85.075% | 86.384% | 98.312% | 66.816% | 90.097% |
| FT-004 ensemble seed 42 | 86.953% | 86.593% | 87.757% | 98.312% | 70.404% | 91.063% |

The ensemble seed-42 row is intentionally not presented as the confirmed three-seed ensemble result. It is used here because per-image error overlap must compare systems on the same seed-42 cohort. The confirmed FT-004 three-seed mean remains reported separately in the main results table.

### Error decomposition

| System | Correct | Total errors | Normal/Pneumonia errors | Viral/Bacterial errors | Viral/Bacterial share of errors |
|---|---:|---:|---:|---:|---:|
| FT-REF | 757 | 117 | 15 | 102 | 87.18% |
| FT-007 label smoothing | 755 | 119 | 8 | 111 | 93.28% |
| FT-004 ensemble seed 42 | 767 | 107 | 13 | 94 | 87.85% |

The dominant failure is not recognizing that pneumonia exists. It is distinguishing its Viral and Bacterial labels. All three systems exceed the pre-registered 70% etiology-error threshold.

### Derived binary-task evidence from saved probabilities

| System | Normal/Pneumonia balanced accuracy | Viral/Bacterial balanced accuracy within 637 true-pneumonia images | Gap |
|---|---:|---:|---:|
| FT-REF | 98.239% | 79.147% | 19.092 points |
| FT-007 label smoothing | 98.842% | 79.353% | 19.489 points |
| FT-004 ensemble seed 42 | 98.528% | 82.303% | 16.225 points |

For Normal/Pneumonia, the Viral and Bacterial probabilities were summed and compared with the Normal probability. For etiology, only true-pneumonia validation images were used and the saved Viral and Bacterial probabilities were compared. These are diagnostic re-analyses of existing outputs, not independently trained binary classifiers. All three exceed the pre-registered 10-point gap threshold.

### Error overlap and model complementarity

Seventy-four images were misclassified by all three systems, while 712 were classified correctly by all three. Relative to FT-REF, label smoothing corrected 36 errors but introduced 38 new errors; the ensemble corrected 27 reference errors and introduced 17. FT-007 therefore changed the error tradeoff without producing a net accuracy gain. The ensemble showed the better matched seed-42 tradeoff, but 74 shared failures demonstrate a persistent common limitation rather than three fully independent solutions.

The saved visual evidence includes confusion matrices, an error-decomposition chart, the highest-confidence errors shared by all three systems, reference errors rescued by the ensemble, and reference errors rescued by label smoothing. Visuals display original validation X-rays for research audit only and do not revise labels.

### Artifacts, result and FT-009B decision

The complete record is `model/results/ft009a_validation_error_analysis/summary.json`, SHA-256 `b0233892de2833c3cd269675e87b653cb8805330572433dd3150be64f31870c9`. The aligned 874-row record is `per_sample_comparison.csv`, SHA-256 `8432a4c42c0af1373ce28118e4c31c89809582ad7024397e7eb85e86d6662b56`. Supporting CSV and image artifacts remain in the same directory.

FT-009A passes the hierarchical design gate for all three systems. The next permitted action is to draft a matched seed-42 FT-009B configuration and contracts with two stages: Normal versus Pneumonia, then Viral versus Bacterial. This is not proof that hierarchy will improve the final three-class metrics. FT-009B training remains unauthorized until the combination rule, stage-specific data, loss, probability mapping, stopping rule, comparison gate, and failure behavior are frozen and tested without locked-test access.

## Experiment FT-009B: hierarchical two-stage classification

### Research question and model structure

Does a matched two-stage DenseNet-201 hierarchy improve the final three-class validation result by separating the comparatively easy Normal/Pneumonia boundary from the harder Viral/Bacterial etiology boundary? FT-009B remains image classification. Stage 1 is a two-output Normal-versus-Pneumonia classifier. Stage 2 is a separate two-output Viral-versus-Bacterial classifier. It is not lung segmentation, detection, or hard-coded diagnosis.

The research basis remains the three verified sources registered under FT-009A/009B in the reference map. They support studying the two boundaries and maintaining strict generalizability limits. They do not prescribe this implementation, its exact hyperparameters, its thresholds, or guarantee improvement.

### Frozen data and label rules

The original manifests remain unchanged and retain SHA-verified version `three_class_seed42_3b815cf769c8`. Stage 1 uses all 4,076 training images and all 874 validation images: Normal is label 0, while Viral and Bacterial are combined as Pneumonia label 1. Its training counts are 1,105 Normal and 2,971 Pneumonia; balanced-loss weights are approximately 1.8443 and 0.6860.

Stage 2 optimization excludes Normal and uses only the 2,971 disease training images and 637 disease validation images: Viral is remapped from original index 1 to binary index 0, and Bacterial from original index 2 to binary index 1. Its training counts are 1,039 Viral and 1,932 Bacterial; balanced-loss weights are approximately 1.4297 and 0.7689. The locked manifests are filtered in memory; no row, split assignment, source path, or original label is rewritten.

During final combined validation, Stage 2 nevertheless infers all 874 validation images, including Normal images. This is necessary to obtain a conditional Viral/Bacterial distribution for every row and avoid a biased hard-routing shortcut. Stage-2 optimization still never uses Normal targets.

### Frozen matched configuration

| Setting | Stage 1 | Stage 2 | Why frozen this way |
|---|---|---|---|
| Task | Normal vs Pneumonia | Viral vs Bacterial | Isolates the two boundaries diagnosed in FT-009A |
| Architecture | DenseNet-201 | DenseNet-201 | Holds model capacity constant |
| Initialization | ImageNet pretrained | ImageNet pretrained | Matches FT-REF and avoids an extra initialization variable |
| Preprocessing | VIRIDIS -> GE -> CLAHE | VIRIDIS -> GE -> CLAHE | Matches the selected single-model reference |
| Resolution | 224 x 224 | 224 x 224 | Matches FT-REF and the validated resource envelope |
| Augmentation | Horizontal flip 0.5; rotation +/-5 degrees | Same | Holds the training-only image variation constant |
| Loss | Balanced cross-entropy; no label smoothing | Same | Tests hierarchy rather than changing the objective |
| Optimizer | AdamW | AdamW | Matches FT-REF |
| Learning rate / weight decay | 0.0001 / 0.0001 | Same | Matches FT-REF |
| Batch / epoch ceiling | 8 / 12 | Same | Matches FT-REF and current GPU memory limits |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 2 | Same | Matches FT-REF |
| Early stopping | Patience 4; minimum delta 0.0001 macro F1 | Same | Prevents indefinite training while using the same selection rule |
| Seed | 42 | 42 | Provides the first matched screening run |

The Stage-1 configuration hash is `cbbcab362fafeeeed21f073c754b809ff2dc92c9ed865320d503306ab5eb4d2c`; the Stage-2 configuration hash is `2c85ad7534b292e699acec7de0e215f196560fe8f2c8aa3b59927916ac79ad07`.

### Frozen probability composition

For each image, let Stage 1 output `P(Normal)` and `P(Pneumonia)`, and Stage 2 output `P(Viral | Pneumonia)` and `P(Bacterial | Pneumonia)`. The final vector is:

1. `P_final(Normal) = P_stage1(Normal)`
2. `P_final(Viral) = P_stage1(Pneumonia) x P_stage2(Viral | Pneumonia)`
3. `P_final(Bacterial) = P_stage1(Pneumonia) x P_stage2(Bacterial | Pneumonia)`

Every binary vector must contain finite probabilities within [0,1] and sum to one within absolute tolerance 0.000001. The composed three-class vector must also sum to one without post-hoc renormalization. Configuration drift, checkpoint/configuration mismatch, validation-path misalignment, missing checkpoints, and invalid probabilities fail closed. The evaluator contains no locked-test option.

### Pre-registered seed-42 advancement gate

These thresholds were frozen before FT-009B full training and are project-defined research gates, not published clinical standards. Seed 42 advances to seeds 123 and 2026 only if all four requirements pass: combined validation macro F1 at least 85.794% (FT-REF plus 0.500 point), balanced accuracy at least 84.546%, Viral recall at least 62.332%, and Normal recall at least 97.312%. Passing would authorize only multi-seed development confirmation; it would not authorize locked-test access.

A future best-confirmed-system claim additionally requires all three seeds and must exceed the FT-004 three-seed mean macro F1 of 85.305%, while not falling below its 84.928% mean balanced accuracy or 66.517% mean Viral recall. FT-004's seed-42 ensemble macro F1 of 86.953% is recorded as a visible single-seed benchmark, not substituted for the multi-seed decision rule.

### Contract tests, smoke test, and authorization

Eleven dedicated FT-009B tests passed, covering task definitions, exact Stage-2 filtering counts, label remapping, historical FT-REF hash preservation, locked-test rejection for both stages, matched-configuration enforcement, probability products, row-sum validation, path-order alignment, float32 softmax precision, and the all-requirements advancement rule. Twenty core pipeline regression tests and five FT-009A artifact tests also passed.

The first combined smoke evaluation correctly stopped because mixed-precision logits were passed directly to softmax, creating half-precision row-sum rounding beyond the frozen tolerance. This was an evaluator defect, not a model finding. Softmax was moved to float32, a regression test was added, and the identical smoke protocol was rerun successfully.

Both one-epoch, 64 x 64, 12-training-image smoke components completed on the GPU, and the 12-image balanced end-to-end composition passed. Its random-initialization metrics are deliberately excluded from research comparison because the smoke test checks execution and artifact contracts rather than performance. The frozen protocol SHA-256 is `cc8eab2d9391dad4901d225363b53778042149a54f3450d7bdbf876cd3fa52dd`; the authorization record SHA-256 is `3bfc41248f975f899f491a49d9535f581bcdf2ba7307675fc6e5e6b4080bdba8`. This authorizes seed-42 development training only. The locked test and additional seeds remain prohibited.

### Full seed-42 execution result

The authorized background orchestrator started Stage 1 at `2026-08-12T08:40:35+08:00` on the NVIDIA GeForce RTX 2050. Both binary models and the combined 874-image validation evaluation completed successfully. Stage 1, Normal versus Pneumonia, selected epoch 9 and achieved 98.699% macro F1, 98.764% balanced accuracy, and 98.970% accuracy. Stage 2, Viral versus Bacterial, selected epoch 5 and achieved 79.850% macro F1, 80.456% balanced accuracy, and 81.319% accuracy on the 637 true-pneumonia validation images.

The composed three-class system achieved 85.270% macro F1, 85.661% balanced accuracy, 85.355% accuracy, 98.312% Normal recall, 75.336% Viral recall, and 83.333% Bacterial recall. Its confusion matrix was `[[233, 4, 0], [5, 168, 50], [0, 69, 345]]` in the fixed order Normal, Viral, Bacterial. The hierarchy improved Viral recall but missed the pre-registered macro-F1 advancement minimum of 85.794%; it therefore stopped at seed 42. The result is retained as the strongest Viral-recall alternative among the focused finalists, not the best overall system.

## Experiment FT-010A: completed label-smoothing multi-seed confirmation

### Question, fixed configuration, and execution

FT-010A tested whether FT-007's seed-42 gain repeated when only the random seed changed. All three runs used DenseNet-201 with ImageNet initialization, VIRIDIS -> GE -> CLAHE preprocessing, 224 x 224 input, batch size 8, AdamW, learning rate 0.0001, weight decay 0.0001, balanced class-weighted cross-entropy, label smoothing 0.05, maximum 12 epochs, early-stopping patience 4, horizontal flip probability 0.5, and random rotation within +/-5 degrees. The seeds were fixed at 42, 123, and 2026. The locked test was not accessed.

The seed-123 run completed at epoch 10 with its best checkpoint at epoch 6. A temporary Windows lock on `summary.json` then stopped the queue controller after training had finished; it did not invalidate or lose the seed-123 checkpoint. The controller was made tolerant of transient Windows file locks and relaunched. It recognized seeds 42 and 123 as complete and trained only the remaining seed 2026. Seed 2026 stopped at epoch 9 with its best checkpoint at epoch 5.

### Per-seed and aggregate findings

| Seed | Best epoch | Macro F1 | Balanced accuracy | Accuracy | Normal recall | Viral recall | Bacterial recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 85.631% | 85.075% | 86.384% | 98.312% | 66.816% | 90.097% |
| 123 | 6 | 84.502% | 84.212% | 85.355% | 98.312% | 65.919% | 88.406% |
| 2026 | 5 | 84.955% | 84.216% | 85.812% | 97.046% | 65.022% | 90.580% |
| **Mean** | N/A | **85.029%** | **84.501%** | **85.850%** | **97.890%** | **65.919%** | **89.694%** |

The sample standard deviation of macro F1 was 0.568 percentage points. Label smoothing therefore repeated reasonably consistently and is the strongest simpler single-network finalist, but its three-seed mean macro F1 remained 0.276 percentage points below FT-004's 85.305% mean.

## Final model selection: frozen FT-004 three-member ensemble

### What "three models" means

The chosen system contains three separately trained DenseNet-201 members for each prediction. It is one ensemble system, not a two-stage hierarchy. Each member receives the same original image but applies a different frozen preprocessing path: (1) VIRIDIS -> GE -> CLAHE, (2) VIRIDIS -> GE, and (3) GE alone. Each network outputs probabilities for Normal, Viral, and Bacterial. The final probability vector is the equal arithmetic mean:

`p_final = (p_member_1 + p_member_2 + p_member_3) / 3`

The class with the largest averaged probability is returned. There are no learned ensemble weights and no tuned decision threshold. The deployable frozen package uses the three original seed-42 checkpoints because seed 42 was the predeclared reference used to define FT-004; it was not retrospectively chosen as the best of the three confirmation seeds. Seeds 42, 123, and 2026 are repeatability trials of the ensemble configuration. They do not mean that deployment requires nine networks.

### Decision rule and statistical confirmation

FT-004 was selected because macro F1 was the predeclared primary metric and FT-004 achieved the highest mean macro F1 among finalists that completed the same three-seed confirmation. Its means were 85.305% macro F1, 86.270% accuracy, 84.928% balanced accuracy, and 66.517% Viral recall. The original VIRIDIS -> GE -> CLAHE single pipeline had 84.168% mean macro F1 and 85.202% mean accuracy. Thus, FT-004 improved macro F1 by 1.137 percentage points and accuracy by 1.068 percentage points.

A 10,000-resample paired, class-stratified validation bootstrap used the same resampled validation images for both systems and averaged each metric across seeds 42, 123, and 2026. FT-004's macro-F1 improvement over the original pipeline was +1.137 points with a 95% percentile interval from +0.155 to +2.126 points; the interval excluded zero. Its accuracy improvement was +1.068 points with a 95% interval from +0.153 to +1.983 points. These are validation-sample uncertainty intervals, not independent-test or clinical-validation claims.

Against FT-010, FT-004's macro-F1 lead was +0.276 points with a 95% interval from -0.833 to +1.369 points. Therefore, FT-004 wins the fixed ranking rule, but the small FT-004-versus-FT-010 difference is statistically uncertain on this validation cohort. FT-010 remains a defensible lower-compute alternative when running three DenseNet-201 models per image is impractical.

### Frozen artifacts, executable inference, and interpretability

The exact final definition, configuration hashes, checkpoint hashes, prediction hashes, and averaging rule are stored in `model/results/final_ft004_ensemble/frozen_selection.json` and `model/configs/final_ft004_ensemble.json`. Paired-bootstrap details are stored in `model/results/final_ft004_ensemble/paired_bootstrap.json` and `.csv`. The executable `FT004EnsemblePredictor` in `model/src/multiclass/inference.py` loads exactly three checkpoints, verifies SHA-256 and configuration hashes, runs each member's own preprocessing, validates probability vectors, and averages them equally. Twenty-four ensemble and regression contracts passed. A real end-to-end image test loaded all three checkpoints and produced a valid three-class prediction.

Six validation-only Grad-CAM panels are stored under `model/results/final_ft004_ensemble/gradcam`: one high-confidence correct case and one high-confidence error for each true class. Every panel shows the original X-ray, the three member maps, and their normalized mean. Attention often overlaps the thoracic region but sometimes extends beyond the lungs. Grad-CAM is therefore reported only as qualitative attention evidence; it is not a lung mask, causal explanation, localization ground truth, or clinical validation.

### Final interpretation boundary

FT-004 is the project's **final validation-selected research model**. It is not described as clinically validated or externally generalizable. The earlier one-time locked-test metrics belong only to the earlier DenseNet-201 + GE configuration and are not transferred to FT-004. A genuinely independent external cohort remains required for an unbiased final performance estimate.

## Reference map and applicability boundaries

| Experiment | Verified source | Evidence applicability | What the source supports | What remains project-specific |
|---|---|---|---|---|
| FT-REF | [Huang et al., DenseNet](https://arxiv.org/abs/1608.06993) | Background | DenseNet architecture and dense feature reuse | Selection of DenseNet-201; all pneumonia metrics; VIRIDIS -> GE -> CLAHE; complete hyperparameters |
| FT-001 | [Touvron et al., resolution discrepancy](https://arxiv.org/abs/1906.06423) | Background | Resolution can affect image-classification behavior and can be fine-tuned experimentally | Choice of 320 x 320 and whether it helps pediatric pneumonia |
| FT-002/2B/2C/005 | [Cohen et al., TorchXRayVision](https://arxiv.org/abs/2111.00595); [official repository](https://github.com/mlmed/torchxrayvision) | Partial | Chest-X-ray library, pretrained-model availability, implementation basis | Pediatric mask quality, thresholds, margins, gates, and downstream benefit |
| FT-003 | [Ganesan et al., CXR augmentation](https://lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9938.pdf) | Partial | Evaluating rotation/translation-style augmentation in CXR classification | Exact conservative ranges and the observed result |
| FT-004 | [Lakshminarayanan et al., deep ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Background | Combining independent networks as an ensemble method | The exact three members, equal weights, and pneumonia performance |
| FT-006 | [Lin et al., focal loss](https://arxiv.org/abs/1708.02002) | Background | Down-weighting easy examples to focus on harder examples | Gamma 2.0 for this task and its observed class tradeoff |
| FT-007 | [Szegedy et al., label smoothing](https://arxiv.org/abs/1512.00567) | Background | Label smoothing as classification regularization | Smoothing 0.05 and whether it improves this model |
| FT-008 | [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597); [Liu et al., improved CXR U-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC9127108/); [Zhang et al., pediatric TransResUNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10365102/) | Partial | U-Net segmentation design, severe-CXR limitations, and pediatric-specific motivation | Validity of the two local artifacts, exact threshold/margin, mask quality, and downstream classifier benefit |
| FT-009A/009B | [Pediatric two-scenario CNN-LSTM study](https://doi.org/10.1016/j.compeleceng.2022.108402); [Pediatric Viral/Bacterial etiology study](https://doi.org/10.1002/ppul.25229); [external-generalizability study](https://pubmed.ncbi.nlm.nih.gov/34648114/) | Direct/Partial plus Background | Separate Normal/Pneumonia and Viral/Bacterial tasks; strict external-validity interpretation | The two-stage DenseNet-201 design, exact evidence thresholds, probability combination, and project performance |

## Reporting rules

1. Every fine-tuning run receives a unique experiment ID, configuration hash, run ID, seed, timestamps, and artifact paths.
2. Screening results use validation data only.
3. A selected candidate requires the same three-seed protocol: 42, 123, and 2026.
4. Mean, sample standard deviation, per-class recall, macro F1, balanced accuracy, and accuracy must be reported.
5. No newly selected configuration inherits or reuses the earlier locked-test result; final external evaluation requires a genuinely independent cohort.
6. Failed, interrupted, and superseded runs remain in the experiment ledger and are not silently removed.
7. Pending results are labeled Pending and are never written as completed findings.

## Results table

| Experiment | Resolution | ROI policy | Seed | Batch | Best epoch | Macro F1 | Balanced accuracy | Accuracy | Viral recall | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| FT-REF-224-S42 | 224 | Full image | 42 | 8 | 6 | 85.294% | 84.546% | 86.613% | 62.332% | Completed reference |
| FT-001-320-S42 | 320 | Full image | 42 | 8 | 6 | 84.215% | 83.939% | 85.469% | 63.229% | Completed; did not improve |
| FT-002-ROI | 512 segmentation; classifier rejected | Lung crop/mask audit | 42 audit selection | 1 segmentation | N/A | N/A | N/A | N/A | N/A | Manual review completed: 69/90 accepted; PSPNet settings rejected as-is |
| FT-002B-ROI-RECAL | 512 segmentation; no classifier training | Three threshold/margin candidates on the 21 prior rejections | Validation tuning subset | 1 segmentation | N/A | N/A | N/A | N/A | N/A | Completed: Aggressive selected 14 times; None selected 7 times |
| FT-002C-ROI-INDEPENDENT | 512 segmentation; no classifier training | Frozen threshold 0.20 and 18% margin on 90 new images | 2026 audit selection | 1 segmentation | N/A | N/A | N/A | N/A | N/A | Completed limitation audit: automatic 73/90 (81.11%); manual 82/90 (91.11%); Bacterial 80.00% |
| FT-003-AUG | 224 | Full image | 42 | 8 | 4 | 83.605% | 83.178% | 84.325% | 65.471% | Completed; did not improve |
| FT-004-ENS | 224 | Full image; three preprocessing members | 42, 123, 2026 | 8 per member | N/A | 85.305% mean | 84.928% mean | 86.270% mean | 66.517% mean | Multi-seed confirmation completed |
| FT-005-ROI-S42 | 224 | Frozen PSPNet threshold 0.20 + 18% bbox margin, then VIRIDIS -> GE -> CLAHE | 42 | 8 planned | N/A | N/A | N/A | N/A | N/A | Completed negative feasibility result: 116/4,950 unusable; stopped before smoke/full classifier training |
| FT-006-FOCAL-S42 | 224 | Full image; VIRIDIS -> GE -> CLAHE; balanced focal loss gamma 2.0 | 42 | 8 | 11 | 83.917% | 83.844% | 84.439% | 67.713% | Completed; Viral recall improved, primary metrics declined; no multi-seed advance |
| FT-007-LS005-S42 | 224 | Full image; VIRIDIS -> GE -> CLAHE; balanced cross-entropy; label smoothing 0.05 | 42 | 8 | 10 | 85.631% | 85.075% | 86.384% | 66.816% | Completed; +0.337-point macro-F1 gain was below the multi-seed advancement threshold |
| FT-008A-UNET-ROI | 256 segmentation; no classifier training | Two frozen U-Nets; threshold 0.50; single largest component; 18% bbox margin | Matched FT-002C screening cohort | 1-image inference | N/A | N/A | N/A | N/A | N/A | Superseded: single-component cleanup could discard one lung; counts excluded from findings |
| FT-008A2-UNET-ROI | 256 segmentation; no classifier training | Two frozen U-Nets; threshold 0.50; up to two largest components; 18% bbox margin | Matched FT-002C screening cohort | 1-image inference | N/A | N/A | N/A | N/A | N/A | Completed no-advance: Classic 10/90 and Legacy 2/90 manual accepts; neither accepted any Viral or Bacterial image |
| FT-009A-ERROR-AUDIT | No training; saved outputs only | Full-image FT-REF, FT-007 and matched FT-004 seed-42 ensemble predictions | Matched seed 42 | N/A | N/A | N/A | N/A | N/A | N/A | Completed: 87.18%-93.28% of errors were Viral/Bacterial; design-only gate passed and FT-009B proceeded to separate contracts |
| FT-009B-HIER-S42 | 224 per stage | Full-image two-stage classification; Stage 1 Normal/Pneumonia, Stage 2 Viral/Bacterial; VIRIDIS -> GE -> CLAHE | 42 | 8 per stage | Stage 1: 9; Stage 2: 5 | 85.270% combined | 85.661% | 85.355% | 75.336% | Completed; Viral recall improved but seed-42 advancement gate failed; no multi-seed advance |
| FT-010A-LS005-MULTISEED | 224 | Full image; VIRIDIS -> GE -> CLAHE; label smoothing 0.05 | 42, 123, 2026 | 8 | 10, 6, 5 | 85.029% mean | 84.501% mean | 85.850% mean | 65.919% mean | Completed three-seed confirmation; strongest simpler single-network finalist |
| FT-011-DIFFERENTIAL-LR | 224 | Full image; VIRIDIS -> GE -> CLAHE; backbone LR 0.00001; classifier LR 0.0001 | 42 only | 8 | 5 of 9 completed | 82.028% | 81.731% | 83.410% | 58.296% | Four screening gates failed; retain FT-004; seeds 123/2026 not run; tuning closed |
| FINAL-FT004-ENS-S42 | 224 | Three DenseNet-201 members: VIRIDIS -> GE -> CLAHE; VIRIDIS -> GE; GE; equal probability mean | Frozen deployment seed 42; design confirmed at 42, 123, 2026 | 8 per member | Member-specific best checkpoints | 85.305% three-seed mean | 84.928% mean | 86.270% mean | 66.517% mean | **Chosen and frozen final validation-selected research model** |

## Interpretation status

### Final October update — FT-011

The final controlled candidate retained the FT-010 single-network configuration except for a ten-times-smaller backbone learning rate: 0.00001 for features and 0.0001 for the classifier, with every layer trainable. It reused the locked manifests, full-image VIRIDIS → GE → CLAHE, 224 × 224 input, flip/rotation augmentation, balanced cross-entropy with label smoothing 0.05, batch size 8, AdamW, and the matched scheduler/early-stopping policy. Fresh training meant a new ImageNet-initialized run, not continuation from a smoke or FT-010 checkpoint.

Seed 42 completed nine epochs and selected epoch 5. Macro F1 was 0.8202782987500647, balanced accuracy 0.8173146418282663, and accuracy 0.834096109839817. Normal recall passed its gate, while macro F1, balanced accuracy, Viral recall, and Bacterial recall failed. Consequently seeds 123 and 2026 were not run, FT-004 remains selected, and this tuning phase is closed. A lower backbone rate was a hypothesis motivated by feature-transfer research, not an automatically superior policy or a paper-prescribed optimum.

The full configuration, every exact gate, class metrics, actual history and figures, matched seed comparisons, 10,000-resample paired proxy-group bootstrap, hardware protection, interrupted-smoke history, full-run telemetry-write pause and recovery, and numbered research references are in [the complete FT-011 report](FT011_Final_Report_20261013.md). The original frozen FT-004 selection hash remains unchanged. Its [portable package](../../artifacts/ft004/README.md) copies only the three selected seed-42 checkpoints, not a new model or nine-network deployment. The website is unchanged; independent external evaluation is the next research phase.

FT-001 and FT-003 completed without exceeding the 224 x 224 reference. FT-002 through FT-005 showed that the available PSPNet and U-Net ROI artifacts had class-dependent feasibility failures and were correctly stopped before they could bias classifier training. FT-006 increased Viral recall but reduced overall balance. FT-007 produced the highest label-smoothed seed-42 single-network result, and FT-010 confirmed that configuration across all three seeds at 85.029% mean macro F1. FT-009A established that most remaining mistakes were Viral/Bacterial; FT-009B then increased Viral recall to 75.336% but did not meet its overall macro-F1 advancement gate. FT-004 remains the highest fully three-seed-confirmed system at 85.305% mean macro F1 and is now frozen as the final validation-selected research model. Its paired-bootstrap improvement over the original pipeline excludes zero, while its small advantage over FT-010 does not. FT-010 is therefore the simpler statistically competitive alternative, and FT-009B is the higher-Viral-recall alternative. No new selected model inherits the earlier locked-test result; independent external evaluation remains required.
