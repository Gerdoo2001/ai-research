# FT 011 Final Controlled Experiment and Selected Model

Updated 13 October 2026 Asia Manila. Decision: **retain FT-004**. FT-011 completed seed 42, failed four of five screening gates, and closed this tuning phase. Seeds 123 and 2026 were not run. We retain the existing frozen three-member DenseNet-201 ensemble rather than replace it with the new single-network candidate.

## Purpose and comparison

We tested whether a smaller backbone learning rate would improve the established VIRIDIS -> Global Histogram Equalization (GE) -> Contrast Limited Adaptive Histogram Equalization (CLAHE) classifier. The backbone is the feature-extracting part; the classifier converts those features into three class scores. FT-010 used 0.0001 throughout. FT-011 used 0.00001 for the backbone and 0.0001 for the classifier, with every layer trainable. This was a fixed comparison, not a new search.

Transferred features can vary in task specificity, which motivates testing how aggressively they should adapt [1]. Domain-adaptation research provides related context [2]. Neither publication establishes that these exact rates are optimal for this dataset. The rates and numerical decision gates are project choices. Prespecifying the gates reduces post-result selection flexibility; repeated validation tuning can still introduce selection bias [3].

## Data preparation and preprocessing

The experiment reused 5,824 canonical unique images and their locked group-controlled assignments: 4,076 training, 874 validation and 874 sealed-test cases. The training classes contain 1,105 Normal, 1,039 Viral and 1,932 Bacterial images. Validation contains 237 Normal, 223 Viral and 414 Bacterial images. Assigned split comes from the manifest, not the legacy directory prefix. FT-011 opened only training and validation images.

No new split, relabelling, duplicate removal or ROI was introduced. Git had converted the four manifest CSVs to CRLF; restoring LF recovered the original exact hashes without changing a row or label. Derived filename and exact-hash groups control known overlap but do not establish verified patient independence.

The pipeline opens the original image and converts it to RGB. VIRIDIS applies the OpenCV colour map to grayscale intensity. GE uses PIL global grayscale equalization. CLAHE then uses OpenCV grayscale enhancement with clip limit 2.0 and an 8 by 8 tile grid. Each grayscale enhancement returns replicated RGB channels. Therefore VIRIDIS affects the intermediate intensity transformation, but the final input is not a retained colourful palette.

Training only then uses a random horizontal flip with probability 0.5 and a uniformly sampled rotation from -5 to +5 degrees, using bilinear interpolation and black fill. Both are retained from the control to keep the comparison matched. Validation has neither random operation. The image is aspect-ratio-preservingly fitted and center cropped to 224 by 224, converted to float32 pixels divided by 255, transposed from height-width-channel to channel-height-width, and normalized with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. Output is 3 by 224 by 224.

## Training configuration and reasons

Every candidate starts from fresh ImageNet DenseNet-201 weights with a seed-matched new three-output classifier. Fresh means no continued FT-010 or smoke-trained checkpoint; it does not mean a random backbone. Seed 42 fixes the random-number sequence used for initialization, order and augmentation. The number 42 is an arbitrary reproducibility label, not 42 models or 42 training attempts.

AdamW uses weight decay 0.0001. The backbone initial rate 0.00001 is ten times smaller than the classifier rate 0.0001. Learning rate controls update size; weight decay discourages large weights. The lower backbone rate was intended to preserve transferable features while the new head adapted. Its weaker result does not prove a mechanism such as underfitting; this experiment only establishes that this fixed policy did not meet the gates.

Balanced weighted cross-entropy uses the training-count formula N divided by K times each class count. The Normal, Viral and Bacterial weights are 1.2295626401901245, 1.30766761302948 and 0.703243613243103. Label smoothing is 0.05, reducing the hard certainty of training targets. Validation loss is ordinary unweighted cross-entropy, so its magnitude is not directly interchangeable with the weighted smoothed training loss.

Batch size remains 8, which defines examples per optimizer update. The maximum is 12 epochs, where an epoch is one pass over the assigned training cases. Early stopping waits four epochs without macro-F1 improvement exceeding 0.0001. A new best checkpoint requires strictly greater macro F1 than the prior best plus 0.0001. These unchanged rules selected epoch 5 and stopped after epoch 9.

ReduceLROnPlateau monitors validation macro F1 in max mode, with factor 0.5 and patience 2. Its relative threshold is 0.0001, cooldown 0, minimum rates [0,0] and epsilon 1e-8. Epoch 8 triggered the rate reduction for the next epoch: backbone 0.000005 and classifier 0.00005. CUDA mixed precision, all BatchNorm layers, zero loader workers, original seed handling and data ordering were retained. The parameter groups are disjoint and cover all 18,098,691 trainable parameters.

The runtime matched Python 3.12.6, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, CUDA 12.1 and cuDNN 90100. ImageNet weights resolve to IMAGENET1K_V1 with SHA256 c1103571aad4f45d0b5b15556d92944db94c33bee68cd58babba3d76225a6401. Historical runs did not save initial-state hashes or per-run source commits; FT-011 records them for the current implementation and fresh initialization.

## Exact screening gates

Every gate had to pass before the two confirmation seeds were authorized. Full-precision unrounded metrics determine the decision; displayed percentages are for reading only. The macro-F1 margin is 0.005 on the zero-to-one scale, which is 0.500 percentage points, not a 0.5 percent relative increase.

| Metric | Exact requirement | Exact observed | Result |
| --- | --- | --- | --- |
| Macro F1 | >= 0.8613107953930739 | 0.8202782987500647 | FAIL |
| Balanced accuracy | >= 0.8507499938072187 | 0.8173146418282663 | FAIL |
| Viral recall | >= 0.6681614349775785 | 0.5829596412556054 | FAIL |
| Normal recall | >= 0.9831223628691983 | 0.9873417721518988 | PASS |
| Bacterial recall | >= 0.9009661835748792 | 0.8816425120772947 | FAIL |

Only Normal recall passed. The protocol therefore requires retaining FT-004, stopping FT-011 and not running seeds 123 or 2026. Three-seed replacement gates are not evaluated, and a one-seed result has no three-seed sample standard deviation.

## Actual results and training history

The selected epoch-5 checkpoint achieved accuracy 83.410%, balanced accuracy 81.731% and macro F1 82.028%. Macro F1 gives each class equal importance; accuracy counts the overall fraction of correct images. Balanced accuracy is the mean of class recalls. Recall asks how many actual examples of a class were found, precision asks how many predicted examples were correct, and F1 combines precision and recall.

| Class | Precision | Recall | F1 | Images |
| --- | --- | --- | --- | --- |
| Normal | 94.355% | 98.734% | 96.495% | 237 |
| Viral | 73.446% | 58.296% | 65.000% | 223 |
| Bacterial | 81.292% | 88.164% | 84.589% | 414 |

| Epoch | Train loss | Val loss | Macro F1 | Duration seconds |
| --- | --- | --- | --- | --- |
| 1 | 0.730425 | 0.485431 | 76.628% | 545.682 |
| 2 | 0.599886 | 0.444436 | 78.386% | 435.069 |
| 3 | 0.543696 | 0.419926 | 80.881% | 399.942 |
| 4 | 0.514506 | 0.414561 | 81.636% | 413.751 |
| 5 | 0.457313 | 0.406582 | 82.028% | 417.882 |
| 6 | 0.415765 | 0.425266 | 81.140% | 478.423 |
| 7 | 0.383387 | 0.428391 | 81.682% | 463.752 |
| 8 | 0.349619 | 0.457961 | 81.430% | 514.985 |
| 9 | 0.325597 | 0.444617 | 81.841% | 422.848 |

![Actual loss history](figures/ft011_loss.png)
Figure 1 Actual weighted training and unweighted validation loss histories

![Actual validation history](figures/ft011_validation.png)
Figure 2 Actual validation scores with the fixed macro F1 screening target

![Selected confusion matrix](figures/ft011_confusion_matrix.png)
Figure 3 True classes are rows and predicted classes are columns

## Matched comparisons and uncertainty

| System at seed 42 | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | --- | --- |
| FT010 | 86.384% | 85.075% | 85.631% |
| FT004 | 87.757% | 86.593% | 86.953% |
| FT011 | 83.410% | 81.731% | 82.028% |

![Matched seed comparison](figures/ft011_matched_comparison.png)
Figure 4 Matched seed 42 comparisons do not mix a single seed with a three seed mean

The paired bootstrap used 10,000 resamples with identical sampled cases across systems. It sampled intact derived groups within class strata, calculated each seed's metric separately and then averaged metrics when multiple seeds existed. Only seed 42 exists for FT-011. This uses 228 proxy groups, including 181 repeated-image groups. They are derived filename and hash clusters rather than verified patient identifiers.

| Comparison | Macro F1 difference points | 95 percent interval points |
| --- | --- | --- |
| FT011 minus FT010 | -3.603250 | -5.733804 to -1.498926 |
| FT011 minus FT004 | -4.924931 | -7.475332 to -2.341838 |

Both observed macro-F1 differences and their validation bootstrap intervals are negative. This supports retaining the incumbent for this matched comparison. It does not establish population performance across arbitrary random seeds, independent external accuracy or verified patient independence [3,4,6].

## Laptop protection and technical recovery

The protected smoke used only 16 training and 16 validation cases; its scores are execution checks, not experimental results. Its simulated unsafe reading exercised complete atomic checkpointing, state restoration and at least 120 continuous healthy seconds before resuming. Full seed-42 training started fresh afterward.

The first smoke attempt paused for RAM pressure, then disappeared without a recorded termination cause. Its checkpoint and logs were preserved under interrupted_checks/20261013_1349 before the identical smoke was repeated. No prior full FT-011 run is claimed from the protocol's earlier ongoing status label.

During full epoch 4 at 14:16:39 Manila, a Windows access-denied error prevented atomic telemetry-file replacement. Required monitoring was treated as unavailable, GPU work paused at a safe batch boundary, and the full model, optimizer, scheduler, scaler, random states, early-stopping state, pending transformed batch and sampler position were checkpointed and restored. The run resumed at 14:18:45 after 122.547 continuous healthy seconds. This interruption must be disclosed even though the aggregate monitoring-gap counter is zero.

Recorded active runtime was 4185.139 seconds and wall runtime 4310.720 seconds. The GPU peaked at 51.0 C. Device VRAM peaked at 1,356,857,344 bytes; peak PyTorch allocated memory was 1,112,218,112 bytes. System RAM used peaked at 15,360,393,216 bytes. Actual overheating and thermal-throttle samples were zero. Shared monitoring history contains one simulated smoke pause and two cooling pauses in total. CPU temperature was unavailable, not zero.

The RTX 2050 reports target 87 C, slowdown 97 C and shutdown 100 C. Project margins warn at 80 C, pause at 84 C, stop critically at 92 C, and resume below 75 C. RAM pauses below 1 GiB available and resumes at least 2 GiB; VRAM pauses at 98 percent and resumes below 92 percent. Required heartbeat age is at most six seconds. Checkpoint disk reserve is 1.5 GiB. These are hardware-anchored project choices rather than universal safe-temperature claims.

Selected-checkpoint inference measured 21.899 milliseconds per image and 45.665 images per second, using batch one, five warm-ups and twenty timed full-float passes. Preprocessing is excluded; this is not total application latency or the three-network FT-004 cost.

## Retained best model and publication

The selected system remains FT-004: three seed-42 DenseNet-201 checkpoints with member pipelines VIRIDIS -> GE -> CLAHE, VIRIDIS -> GE, and GE. Each produces Normal, Viral and Bacterial probabilities. We take their equal arithmetic mean and choose the largest probability. The three confirmation seeds describe repeatability of this design; deployment uses three networks, not nine. Ensemble research motivates this aggregation but does not prove its benefit for every population [5].

The previously completed FT-004 three-seed means are macro F1 85.30516857750923 percent, accuracy 86.27002288329519 percent, balanced accuracy 84.92846583463107 percent and Viral recall 66.51718983557549 percent. These means are different summaries from the matched seed-42 table above. FT-004 remains a validation-selected research prototype, not a clinically validated device.

Its original frozen selection SHA256 remains bbb65ee1202b2c34490cdb1047339795e5751f2c71edf6626d1b1f235acea57b. The portable repository publication copies its three original checkpoints without changing their bytes, records each hash and original configuration provenance, and resolves member files relative to the published selection. Git LFS carries the large weights. No FT-011 weights replace the selected model and the live website remains unchanged.

Canonical evidence is model/results/ft011/protocol.json, final_decision.json, frozen_selection.json, the seed-42 run directory, and the timestamped analysis directory. The documentation index identifies the current Word reports and model package. Historical records and the interrupted smoke remain preserved.

## Conclusion and next evaluation

FT-011 is a completed negative result for this fixed learning-rate policy. It did not improve the matched control enough to satisfy the predeclared gates, and the tuning phase is closed. The next step is genuinely independent evaluation of the frozen FT-004 system on a separately collected, appropriately labelled cohort. Do not transfer the earlier DenseNet-201 plus GE locked-test score to FT-004 or describe validation reuse as external performance [4,6].

## Numbered references

[1] [Yosinski et al 2014 How transferable are features in deep neural networks](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html)
[2] [Long et al 2015 Learning Transferable Features with Deep Adaptation Networks](https://proceedings.mlr.press/v37/long15.html)
[3] [Cawley and Talbot 2010 On Over fitting in Model Selection and Subsequent Selection Bias](https://www.jmlr.org/papers/v11/cawley10a.html)
[4] [Collins et al 2024 TRIPOD AI statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/)
[5] [Lakshminarayanan et al 2017 Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
[6] [Riley et al 2024 How to undertake an external validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10788734/)
