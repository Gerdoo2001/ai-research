# Verified Research Evidence Matrix

Generated: 2026-08-02T12:50:27+08:00

The supplied spreadsheet screenshot is treated as a supplementary index only. The repository contains no original RRL spreadsheet or independent reference-list document, and the screenshot's Reference Link cells are blank. Each of its ten visible study rows was therefore located by title/authors/year and checked against a publisher page, DOI record, official repository, or another verified primary source.

## Required reference-link rule

> Open and verify every available reference link relevant to the eight supported models. When the link column is blank, incomplete, inaccessible, or only visible through the supplied image, locate the original publication using the exact title, authors, year, DOI, or publisher.

## Applicability summary

The complete machine-readable matrix contains 25 verified records: **4 Direct**, **10 Partial**, **9 Background**, and **2 Excluded**. “Direct” means the source closely matches the Normal/Viral/Bacterial CXR task or its primary dataset. It does not mean that every reported hyperparameter should be copied.

| Screenshot ID | First author | Verified year | Verified title | Applicability | Reason |
|---|---|---|---|---|---|
| 6 | Dejun Zhang et al. | 2021 | Pneumonia Detection from Chest X-ray Images Based on Convolutional Neural Network | Partial | Directly concerns CXR contrast enhancement and CNN classification, but the task is binary and the architecture/split differ. |
| 7 | Halit Bakır et al. | 2023 | Detection of Pneumonia from X-ray Images Using Deep Learning Techniques | Direct | Uses the exact Normal/Viral/Bacterial three-class CXR task and includes a ResNet feature extractor. |
| 4 | Tawsifur Rahman et al. | 2020 | Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-ray | Direct | Exact task and class labels; directly relevant DenseNet201 and ResNet-family transfer learning. |
| 3 | Rohit Kundu et al. | 2021 | Pneumonia Detection in Chest X-ray Images Using an Ensemble of Deep Learning Models | Partial | Relevant DenseNet121/ResNet ensemble evidence, but task is binary and method is an ensemble. |
| 11 | T. S. Arulananth et al. | 2024 | Classification of Paediatric Pneumonia Using Modified DenseNet-121 Deep-Learning Model | Partial | Relevant DenseNet121 pediatric CXR study, but not the exact three-class task. |
| 5 | Amit Kumar Jaiswal et al. | 2019 | Identifying Pneumonia in Chest X-Rays: A Deep Learning Approach | Partial | Relevant CXR localization and ResNet backbones, but not a three-class classifier. |
| not yet | Fausto Milletari et al. | 2016 | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | Excluded | Volumetric prostate MRI segmentation cannot determine 2D chest-X-ray classification settings. |
| 2 | Zhenxing Huang et al. | 2021 journal issue; online 2020 | FaNet: Fast Assessment Network for COVID-19 Pneumonia Based on 3D CT Imaging and Clinical Symptoms | Background | Provides multimodal context only and must not set the present image-classifier configuration. |
| 1 | Katy Stokes et al. | 2022 | The Use of AI Systems in Diagnosis of Pneumonia via Signs and Symptoms: A Systematic Review | Background | Useful clinical-AI context only; no direct model-configuration evidence for the image classifier. |
| 9 | Hicham Moujahid et al. | 2022 journal issue; online 2021 | Combining CNN and Grad-CAM for COVID-19 Disease Prediction and Visual Explanation | Partial | Supports the final explainability workflow, not the present classifier settings. |

## Corrections to the supplied screenshot

- **Zhang et al.:** the verified title is *Pneumonia Detection from Chest X-ray Images Based on Convolutional Neural Network*. The 97.51% merit is parameter reduction relative to VGG16; accuracy was 96.068%.
- **Rahman et al.:** the screenshot listed binary and pairwise results but omitted the directly relevant exact three-class accuracy of 93.3%.
- **Jaiswal et al.:** the paper reports RSNA prediction/localization scores, not a generic “segmentation accuracy.” Detection/localization evidence is not classifier-hyperparameter evidence.
- **FaNet:** published in a 2021 journal issue after 2020 online publication; both dates are retained.
- **Moujahid et al.:** published in a 2022 issue after 2021 online activity; both dates are retained.
- **V-Net:** concerns 3D prostate MRI segmentation and is excluded from classifier configuration evidence.

## Preprocessing evidence decision

- **CLAHE is included as a controlled candidate** because verified CXR studies report potential benefit. However, the strongest located MobileNetV2 paper combines CLAHE with white balance, so the current isolated CLAHE transform still requires a direct ablation.
- **Global histogram equalization is included as a controlled candidate** because Ferreira et al. studied the same pediatric CXR domain and bacterial/viral distinction; every preprocessed strategy exceeded its original-image baseline AUC. Their segmentation/cropping and ensemble prevent automatic transfer of the result.
- **VIRIDIS is included only as an exploratory candidate.** The verified pseudo-color CXR paper constructs three channels from original, histogram-equalized, and bilateral-filtered images. It does not use VIRIDIS. No primary paper was found that validates OpenCV VIRIDIS for this exact task.
- **PCA remains deferred/unsupported** as raw-image preprocessing because no leakage-safe, train-only feature-space protocol is defined.
- **FILM and SAS tuning remain unsupported** because the repository and supplied references do not define a reproducible conditioning signal or algorithm.

## Architecture evidence decision

- DenseNet121/169/201 and ResNet50/101/152 have official torchvision implementation provenance and directly relevant CXR studies, but all eight models continue to use one common transform/trainer for fairness.
- YOLOv5s-cls v7.0 and YOLOv8n-cls v8.3.0 are genuine image-classification checkpoints with verified Classify heads; they are not detector heads repurposed by name.
- The project intentionally does not invoke native Ultralytics training transforms. Installed Ultralytics 8.3.168 contains RandomResizedCrop, RandAugment, and random-erasing defaults; using them only for YOLO would invalidate the controlled architecture comparison.
- YOLOv10 remains excluded because the verified official model is an object detector and no compatible official image-classification checkpoint/head was found.

Complete tables:

- `model/results/evidence/research_evidence_matrix.csv`
- `model/results/evidence/research_evidence_matrix.json`
