# FT-011 final controlled learning-rate experiment

Decision: **retain FT-004**.

The single DenseNet-201 candidate changes the initial backbone learning rate to 0.00001 while the classifier remains at 0.0001. Transferability research provides a rationale for testing different learning rates; it does not establish that this particular rate will improve pneumonia classification. [1] [2] The fixed screening and replacement rules guard against selecting a favorable result after inspecting it. [3] [7]

## Exact decision gates

Seed-42 screening:

| Metric | Rule | Full-precision observed | Pass/fail |
|---|---|---:|---|
| macro_f1 | >= 0.8613107953930739 | 0.8202782987500647 | FAIL |
| balanced_accuracy | >= 0.8507499938072187 | 0.8173146418282663 | FAIL |
| viral_recall | >= 0.6681614349775785 | 0.5829596412556054 | FAIL |
| normal_recall | >= 0.9831223628691983 | 0.9873417721518988 | PASS |
| bacterial_recall | >= 0.9009661835748792 | 0.8816425120772947 | FAIL |

Three-seed replacement gates were not evaluated. A single completed seed cannot establish the three-seed final selection.

## Actual validation results

| Seed | Accuracy | Balanced accuracy | Macro F1 | Normal recall | Viral recall | Bacterial recall |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.834096109839817 | 0.8173146418282663 | 0.8202782987500647 | 0.9873417721518988 | 0.5829596412556054 | 0.8816425120772947 |

Seed 42, selected epoch 5:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9435483870967742 | 0.9873417721518988 | 0.9649484536082474 | 237 |
| Viral | 0.7344632768361582 | 0.5829596412556054 | 0.65 | 223 |
| Bacterial | 0.8129175946547884 | 0.8816425120772947 | 0.8458864426419467 | 414 |

Recorded active training time: 4185.139188299887 seconds; peak allocated GPU memory: 1112218112 bytes. Inference latency: 21.89868506975472 ms per image, throughput: 45.66484228686159 images/second. The latency benchmark excludes preprocessing and uses the recorded measurement settings.


Each seed's metric is calculated separately. Means and sample standard deviations are in the machine-readable analysis.json; no probability vectors are averaged across confirmation seeds.

## Uncertainty and interpretation

Paired bootstrap uses 10,000 resamples. Each draw is shared by FT-011, FT-010 and FT-004 and every included seed. Entire derived split_group_id groups are sampled with replacement within class strata. These are proxy filename/hash clusters, not verified patient identifiers; they cannot establish patient independence. The bootstrap measures uncertainty on the already used validation cases. It does not produce independent external performance. [3] [4] [6]

The incumbent FT-004 uses a previously frozen equal-probability ensemble, with separate seed-matched ensemble predictions for comparison. Ensemble research explains its rationale without proving superiority for this dataset. [5] [8]

## Artifacts and next step

The original locked train/validation assignments and preparation record were preserved. [9] No test images or test metrics were accessed. Candidate artifacts, exact gate evidence, hashes, bootstrap intervals and actual graphs are recorded alongside this report. The live website is unchanged. Independent evaluation of the frozen system is the next step. [4] [6]

![Actual result graph](seed_42_training_curves.png)
![Actual result graph](seed_42_learning_rates.png)
![Actual result graph](seed_42_confusion_matrix.png)
![Actual result graph](matched_seed_model_comparison.png)

## Numbered references

[1] [Yosinski et al. (2014), transferability of neural-network features](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html)
[2] [Long et al. (2015), transferable features](https://proceedings.mlr.press/v37/long15.html)
[3] [Cawley and Talbot (2010), model-selection bias](https://www.jmlr.org/papers/v11/cawley10a.html)
[4] [TRIPOD+AI (2024), transparent prediction-model reporting](https://www.bmj.com/content/385/bmj-2023-078378)
[5] [Lakshminarayanan et al. (2017), deep ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
[6] [Riley et al. (2024), external validation](https://www.bmj.com/content/384/bmj-2023-074820)
[7] [FT-011 protocol](https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit)
[8] [Completed fine-tuning report](https://docs.google.com/document/d/1mdLbxZMtv2icHKNs-VSjpZiWb5_2lG-S/edit)
[9] [Dataset preparation record](https://docs.google.com/document/d/1I9-egb-SZRzbGQ-PXxM8UjvGmc_CCcsZ/edit)

[1]: https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html
[2]: https://proceedings.mlr.press/v37/long15.html
[3]: https://www.jmlr.org/papers/v11/cawley10a.html
[4]: https://www.bmj.com/content/385/bmj-2023-078378
[5]: https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
[6]: https://www.bmj.com/content/384/bmj-2023-074820
[7]: https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit
[8]: https://docs.google.com/document/d/1mdLbxZMtv2icHKNs-VSjpZiWb5_2lG-S/edit
[9]: https://docs.google.com/document/d/1I9-egb-SZRzbGQ-PXxM8UjvGmc_CCcsZ/edit

Protocol references provide methodological rationale and reporting guidance; their use is not a claim of verified efficacy for the chosen FT-011 configuration.
