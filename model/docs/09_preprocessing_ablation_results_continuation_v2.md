# Preprocessing Ablation Results

Generated: 2026-08-03T05:06:34+08:00

This report is generated only when all twelve validation artifacts exist: three completed no-enhancement controls and nine new enhanced runs. The locked test set is not loaded.

## Results

| Architecture | Condition | Validation macro F1 | Delta vs same-model control | Paired 95% bootstrap interval | Viral recall | Best epoch | Descriptive interpretation |
|---|---|---:|---:|---|---:|---:|---|
| densenet169 | No enhancement | 0.840293 | +0.000000 | [+0.000000, +0.000000] | 0.619 | 6 | control |
| densenet169 | CLAHE | 0.833828 | -0.006464 | [-0.029280, +0.017440] | 0.668 | 5 | bootstrap interval includes no difference |
| densenet169 | Global HE | 0.829458 | -0.010834 | [-0.033207, +0.012754] | 0.659 | 5 | bootstrap interval includes no difference |
| densenet169 | VIRIDIS | 0.837293 | -0.003000 | [-0.025288, +0.020331] | 0.646 | 7 | bootstrap interval includes no difference |
| densenet201 | No enhancement | 0.838713 | +0.000000 | [+0.000000, +0.000000] | 0.700 | 10 | control |
| densenet201 | CLAHE | 0.841733 | +0.003021 | [-0.019921, +0.024824] | 0.677 | 6 | bootstrap interval includes no difference |
| densenet201 | Global HE | 0.844714 | +0.006001 | [-0.017618, +0.028911] | 0.664 | 6 | bootstrap interval includes no difference |
| densenet201 | VIRIDIS | 0.837062 | -0.001651 | [-0.025580, +0.019865] | 0.570 | 6 | bootstrap interval includes no difference |
| resnet50 | No enhancement | 0.836919 | +0.000000 | [+0.000000, +0.000000] | 0.637 | 5 | control |
| resnet50 | CLAHE | 0.823620 | -0.013299 | [-0.035738, +0.007975] | 0.619 | 4 | bootstrap interval includes no difference |
| resnet50 | Global HE | 0.819553 | -0.017365 | [-0.042769, +0.008418] | 0.713 | 4 | bootstrap interval includes no difference |
| resnet50 | VIRIDIS | 0.838547 | +0.001628 | [-0.022698, +0.026951] | 0.578 | 6 | bootstrap interval includes no difference |

## Representative winners

- **densenet169:** `baseline_densenet169_s42`
- **densenet201:** `ablation_densenet201_histogram_equalization_s42`
- **resnet50:** `ablation_resnet50_pseudo_color_s42`

The primary selection metric is validation macro F1. Ties are resolved by balanced accuracy, minimum per-class recall, log loss, model size, compute-only latency, then run ID. The paired class-stratified bootstrap uses 2,000 resamples and describes uncertainty on this same validation set; it is not an independent test and does not unlock test data.

VIRIDIS remains labeled exploratory even if it scores well. A score increase would show only that this fixed colormap helped the present validation pipeline; it would not prove that artificial color adds clinical information.
