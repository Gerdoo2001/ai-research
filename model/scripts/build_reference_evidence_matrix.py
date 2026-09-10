"""Build a verified, applicability-tagged evidence matrix for the continuation."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model" / "results" / "evidence"
DOCS = ROOT / "model" / "docs"


COLUMNS = [
    "Reference ID", "Screenshot Journal ID", "Research Category", "Authors", "Year",
    "Article Title", "Venue", "DOI", "Official Link", "Research Objective", "Methodology",
    "Task and Classes", "Dataset", "Architecture or Method", "Input Size", "Preprocessing",
    "Augmentation", "Balancing", "Pretrained Weights", "Optimizer", "Initial Learning Rate",
    "Weight Decay", "Batch Size", "Epochs", "Scheduler", "Fine-Tuning Strategy", "Loss",
    "Selection or Evaluation Metric", "Reported Results", "Claimed Merit", "Limitations",
    "Evidence Applicability", "Evidence Applicability Reason", "Source Obtained From",
    "Verification Status", "Notes or Corrections",
]


def row(**kwargs):
    result = {column: "Not reported" for column in COLUMNS}
    result.update(kwargs)
    return result


RECORDS = [
    row(**{
        "Reference ID": "IMG-06", "Screenshot Journal ID": "6", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "Dejun Zhang; Fuquan Ren; Yushuang Li; Lei Na; Yue Ma", "Year": "2021",
        "Article Title": "Pneumonia Detection from Chest X-ray Images Based on Convolutional Neural Network",
        "Venue": "Electronics 10(13):1512", "DOI": "10.3390/electronics10131512",
        "Official Link": "https://www.mdpi.com/2079-9292/10/13/1512",
        "Research Objective": "Develop a small VGG-inspired CNN for Normal versus Pneumonia chest-X-ray classification.",
        "Methodology": "Six-layer VGG-inspired binary CNN; dynamic histogram enhancement (DHE); random 70/10/20 split of 5,786 images.",
        "Task and Classes": "Binary CXR classification: Normal, Pneumonia", "Dataset": "5,786 pediatric CXRs",
        "Architecture or Method": "Simplified VGG-inspired six-layer CNN", "Input Size": "224 x 224 selected among tested sizes",
        "Preprocessing": "DHE; one-channel values scaled to [0,1]", "Augmentation": "Rotation 0-30 degrees; zoom 0.2; width/height shift 0.1; horizontal and vertical flips",
        "Optimizer": "Adam", "Initial Learning Rate": "0.001", "Weight Decay": "0.0001 reported as decay",
        "Loss": "Binary cross entropy", "Selection or Evaluation Metric": "Accuracy, AUC, precision, recall, F1",
        "Reported Results": "Accuracy 96.068%; AUC 0.99107; precision 94.408%; recall 90.823%; F1 92.851%.",
        "Claimed Merit": "Small deployable classifier; 97.51% fewer parameters than VGG16.",
        "Limitations": "Binary task and independent random split; does not establish settings for viral-versus-bacterial discrimination.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Directly concerns CXR contrast enhancement and CNN classification, but the task is binary and the architecture/split differ.",
        "Source Obtained From": "Supplied image; Publisher page", "Verification Status": "Verified",
        "Notes or Corrections": "Screenshot title was corrected to the publisher title. The >97% claim is parameter reduction, not an accuracy or generic support claim.",
    }),
    row(**{
        "Reference ID": "IMG-07", "Screenshot Journal ID": "7", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "Halit Bakır; Semih Oktay; Emre Tabaru", "Year": "2023",
        "Article Title": "Detection of Pneumonia from X-ray Images Using Deep Learning Techniques",
        "Venue": "Journal of Scientific Reports-A 52:419-440", "DOI": "10.59313/jsr-a.1219363",
        "Official Link": "https://dergipark.org.tr/en/pub/jsr-a/article/1219363",
        "Research Objective": "Compare ANN and CNN-based feature-extractor combinations for pneumonia detection.",
        "Methodology": "Pretrained feature extractors including ResNet and MobileNet were combined with ANN/CNN classifiers for binary and multiclass experiments.",
        "Task and Classes": "Binary and three-class CXR classification; three-class includes Normal, Bacterial Pneumonia, Viral Pneumonia",
        "Architecture or Method": "ANN/CNN hybrids with ResNet and MobileNet feature extraction",
        "Selection or Evaluation Metric": "Accuracy", "Reported Results": "Best binary ANN+MobileNet accuracy 95.67%; best exact three-class ANN+ResNet accuracy 81.67%.",
        "Claimed Merit": "Demonstrates transfer-feature hybrids on the exact three clinical labels.",
        "Limitations": "Different training pipeline and hybrid classifier; incomplete transferability of its hyperparameters to end-to-end fine-tuning.",
        "Evidence Applicability": "Direct", "Evidence Applicability Reason": "Uses the exact Normal/Viral/Bacterial three-class CXR task and includes a ResNet feature extractor.",
        "Source Obtained From": "Supplied image; Publisher page", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMG-04", "Screenshot Journal ID": "4", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "Tawsifur Rahman; Muhammad E. H. Chowdhury; Amith Khandakar; Khandaker R. Islam; Khandaker F. Islam; Zaid B. Mahbub; Muhammad A. Kadir; Saad Kashem",
        "Year": "2020", "Article Title": "Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-ray",
        "Venue": "Applied Sciences 10(9):3233", "DOI": "10.3390/app10093233", "Official Link": "https://www.mdpi.com/2076-3417/10/9/3233",
        "Research Objective": "Evaluate transfer-learning CNNs for Normal, Bacterial Pneumonia, and Viral Pneumonia CXR classification.",
        "Methodology": "Compared AlexNet, ResNet18, DenseNet201, and SqueezeNet using 5,247 chest radiographs.",
        "Task and Classes": "Binary and exact three-class CXR classification", "Dataset": "5,247 pediatric CXRs",
        "Architecture or Method": "AlexNet; ResNet18; DenseNet201; SqueezeNet", "Pretrained Weights": "Transfer learning",
        "Selection or Evaluation Metric": "Accuracy and related classification metrics",
        "Reported Results": "98% Normal-versus-Pneumonia accuracy; 95% Bacterial-versus-Viral accuracy; 93.3% exact three-class accuracy.",
        "Claimed Merit": "Strong, directly comparable evidence for DenseNet/ResNet transfer learning on the exact label set.",
        "Limitations": "Does not by itself justify the present optimizer, augmentation, split, or leakage-control protocol.",
        "Evidence Applicability": "Direct", "Evidence Applicability Reason": "Exact task and class labels; directly relevant DenseNet201 and ResNet-family transfer learning.",
        "Source Obtained From": "Supplied image; Publisher page", "Verification Status": "Verified",
        "Notes or Corrections": "Screenshot omitted the exact three-class result; publisher reports 93.3%.",
    }),
    row(**{
        "Reference ID": "IMG-03", "Screenshot Journal ID": "3", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "Rohit Kundu; Ritacheta Das; Zong Woo Geem; Gi-Tae Han; Ram Sarkar", "Year": "2021",
        "Article Title": "Pneumonia Detection in Chest X-ray Images Using an Ensemble of Deep Learning Models",
        "Venue": "PLOS ONE 16(9):e0256630", "DOI": "10.1371/journal.pone.0256630",
        "Official Link": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256630",
        "Research Objective": "Improve binary pneumonia detection through a weighted ensemble.",
        "Methodology": "Five-fold evaluation of weighted GoogLeNet, ResNet18, and DenseNet121 predictions on Kermany and RSNA data.",
        "Task and Classes": "Binary CXR classification", "Dataset": "Kermany pediatric CXR and RSNA pneumonia data",
        "Architecture or Method": "Weighted GoogLeNet/ResNet18/DenseNet121 ensemble", "Selection or Evaluation Metric": "Accuracy, sensitivity, precision, F1, AUC",
        "Reported Results": "Accuracy 98.81% on Kermany and 86.85% on RSNA; sensitivity 98.80% and 87.02%, respectively.",
        "Claimed Merit": "Weighted complementary CNN predictions outperform individual members.",
        "Limitations": "Binary ensemble evidence does not establish single-model three-class hyperparameters.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Relevant DenseNet121/ResNet ensemble evidence, but task is binary and method is an ensemble.",
        "Source Obtained From": "Supplied image; Publisher page", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMG-11", "Screenshot Journal ID": "11", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "T. S. Arulananth; S. Wilson Prakash; Ramesh Kumar Ayyasamy; V. P. Kavitha; P. G. Kuppusamy; P. Chinnasamy", "Year": "2024",
        "Article Title": "Classification of Paediatric Pneumonia Using Modified DenseNet-121 Deep-Learning Model",
        "Venue": "IEEE Access 12:35716-35727", "DOI": "10.1109/ACCESS.2024.3371151",
        "Official Link": "https://doaj.org/article/8d40e456925949dbb8aa0b6b189f8d44",
        "Research Objective": "Improve pediatric pneumonia classification with a modified DenseNet121.",
        "Methodology": "DenseNet121 modified with batch normalization, max pooling, and dropout; trained on pediatric CXR.",
        "Task and Classes": "Binary pediatric CXR classification", "Architecture or Method": "Modified DenseNet121",
        "Selection or Evaluation Metric": "Classification metrics", "Reported Results": "Accuracy 97.03% with strong precision and recall.",
        "Claimed Merit": "Recent DenseNet121-focused pediatric CXR evidence.", "Limitations": "Binary task and modified head prevent direct hyperparameter transfer.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Relevant DenseNet121 pediatric CXR study, but not the exact three-class task.",
        "Source Obtained From": "Supplied image; DOAJ publisher metadata; IEEE DOI metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMG-05", "Screenshot Journal ID": "5", "Research Category": "Segmentation and localization of pneumonia regions",
        "Authors": "Amit Kumar Jaiswal; Prayag Tiwari; Sachin Kumar; Deepak Gupta; Ashish Khanna; Joel J. P. C. Rodrigues", "Year": "2019",
        "Article Title": "Identifying Pneumonia in Chest X-Rays: A Deep Learning Approach",
        "Venue": "Measurement 145:511-518", "DOI": "10.1016/j.measurement.2019.05.076",
        "Official Link": "https://www.sciencedirect.com/science/article/pii/S0263224119305202",
        "Research Objective": "Identify and localize pneumonia opacities on chest X-rays.",
        "Methodology": "Mask R-CNN with ResNet50 and ResNet101 backbones on RSNA challenge images; ensembled predictions.",
        "Task and Classes": "CXR opacity detection/localization", "Dataset": "RSNA Pneumonia Detection Challenge, about 30,000 training and 4,500 test images",
        "Architecture or Method": "Mask R-CNN; ResNet50/101", "Selection or Evaluation Metric": "RSNA challenge prediction/localization score",
        "Reported Results": "Stage-2 scores reported as 0.183720 (ResNet50), 0.199352 (ResNet101), and 0.218051 (ensemble).",
        "Claimed Merit": "Supports localization/ROI discussion and ResNet backbones.",
        "Limitations": "Detection/localization scores are not classification accuracy and do not determine classifier hyperparameters.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Relevant CXR localization and ResNet backbones, but not a three-class classifier.",
        "Source Obtained From": "Supplied image; Publisher page; primary paper PDF", "Verification Status": "Verified",
        "Notes or Corrections": "Replaces the screenshot's unsupported phrase 'strong localization & segmentation accuracy' with the reported challenge scores.",
    }),
    row(**{
        "Reference ID": "IMG-VNET", "Screenshot Journal ID": "not yet", "Research Category": "Segmentation and localization of pneumonia regions",
        "Authors": "Fausto Milletari; Nassir Navab; Seyed-Ahmad Ahmadi", "Year": "2016",
        "Article Title": "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation",
        "Venue": "2016 Fourth International Conference on 3D Vision (3DV), 565-571", "DOI": "10.1109/3DV.2016.79",
        "Official Link": "https://arxiv.org/abs/1606.04797", "Research Objective": "Segment the prostate in volumetric MRI.",
        "Methodology": "3D fully convolutional residual network optimized with Dice loss.", "Task and Classes": "3D prostate MRI segmentation",
        "Dataset": "Volumetric prostate MRI", "Architecture or Method": "V-Net", "Preprocessing": "Nonlinear intensity transformations and histogram matching are discussed",
        "Loss": "Dice-based objective", "Reported Results": "Outperformed compared 2D approaches on the study's volumetric segmentation task.",
        "Claimed Merit": "Foundational 3D segmentation architecture.", "Limitations": "Different modality, dimensionality, anatomy, task, loss, and output contract.",
        "Evidence Applicability": "Excluded", "Evidence Applicability Reason": "Volumetric prostate MRI segmentation cannot determine 2D chest-X-ray classification settings.",
        "Source Obtained From": "Supplied image; Official repository; DOI metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMG-02", "Screenshot Journal ID": "2", "Research Category": "Multimodal or non-imaging AI for pneumonia",
        "Authors": "Zhenxing Huang; Xinfeng Liu; Rongpin Wang; Mudan Zhang; Xianchun Zeng; Jun Liu; Yongfeng Yang; Xin Liu; Hairong Zheng; Dong Liang; Zhanli Hu",
        "Year": "2021 journal issue; online 2020", "Article Title": "FaNet: Fast Assessment Network for COVID-19 Pneumonia Based on 3D CT Imaging and Clinical Symptoms",
        "Venue": "Applied Intelligence 51(5):2838-2849", "DOI": "10.1007/s10489-020-01965-0",
        "Official Link": "https://pubmed.ncbi.nlm.nih.gov/34764567/", "Research Objective": "Combine 3D CT and clinical symptoms to diagnose COVID-19 and assess severity.",
        "Methodology": "Dual-branch 3D CNN and symptom encoding on 416 patients.", "Task and Classes": "Multimodal COVID-19 diagnosis and severity assessment",
        "Dataset": "416-patient CT plus clinical-symptom cohort", "Architecture or Method": "3D CNN plus symptom branch",
        "Reported Results": "98.28% diagnosis accuracy; 94.83% severity accuracy.", "Claimed Merit": "Demonstrates complementary imaging and clinical features.",
        "Limitations": "3D CT, symptom fusion, and COVID severity differ from 2D CXR image-only classification.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Provides multimodal context only and must not set the present image-classifier configuration.",
        "Source Obtained From": "Supplied image; PubMed; Publisher DOI metadata", "Verification Status": "Verified",
        "Notes or Corrections": "Both the online-first 2020 date and 2021 journal issue are preserved.",
    }),
    row(**{
        "Reference ID": "IMG-01", "Screenshot Journal ID": "1", "Research Category": "Multimodal or non-imaging AI for pneumonia",
        "Authors": "Katy Stokes; Rossana Castaldo; Carlo Federici; Silvio Pagliara; Alessia Maccaro; Francesco Cappuccio; Giuseppe Fico; Marco Salvatore; Monica Franzese; Leandro Pecchia",
        "Year": "2022", "Article Title": "The Use of AI Systems in Diagnosis of Pneumonia via Signs and Symptoms: A Systematic Review",
        "Venue": "Biomedical Signal Processing and Control 72A:103325", "DOI": "10.1016/j.bspc.2021.103325",
        "Official Link": "https://wrap.warwick.ac.uk/id/eprint/160069/", "Research Objective": "Review machine-learning pneumonia diagnosis from clinical signs and symptoms rather than imaging.",
        "Methodology": "PRISMA review: 876 records screened and 16 studies included.", "Task and Classes": "Non-imaging pneumonia prediction",
        "Architecture or Method": "Systematic review of symptom-based ML", "Reported Results": "Most included models reported greater than 90% accuracy, with substantial inconsistency and heterogeneity.",
        "Claimed Merit": "Summarizes non-imaging diagnostic approaches and reporting gaps.", "Limitations": "Not an imaging model and not transferable to CXR preprocessing or architecture hyperparameters.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Useful clinical-AI context only; no direct model-configuration evidence for the image classifier.",
        "Source Obtained From": "Supplied image; Official university repository; Publisher DOI metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMG-09", "Screenshot Journal ID": "9", "Research Category": "Explainability and Grad-CAM for respiratory imaging",
        "Authors": "Hicham Moujahid; Bouchaib Cherradi; Mohammed Al-Sarem; Lhoussain Bahatti; Bakr Assedik Eljialy; Abdullah Alsaeedi; Faisal Saeed",
        "Year": "2022 journal issue; online 2021", "Article Title": "Combining CNN and Grad-CAM for COVID-19 Disease Prediction and Visual Explanation",
        "Venue": "Intelligent Automation & Soft Computing 32(2):723-745", "DOI": "10.32604/iasc.2022.022179",
        "Official Link": "https://www.techscience.com/iasc/v32n2/45598/html", "Research Objective": "Build a three-class respiratory CXR classifier and explain predictions with Grad-CAM.",
        "Methodology": "VGG16, VGG19, and MobileNetV2 with Grad-CAM; 70/15/15 split; 840 COVID, 1,341 Normal, 1,345 Pneumonia images.",
        "Task and Classes": "Three-class CXR classification: Normal, Pneumonia, COVID-19", "Dataset": "3,526 CXRs from combined public sources",
        "Architecture or Method": "VGG16; VGG19; MobileNetV2; Grad-CAM", "Augmentation": "No augmentation reported/applied",
        "Selection or Evaluation Metric": "Accuracy, precision, recall, F1, heatmap inspection", "Reported Results": "Best VGG19: 96.97% accuracy; reported precision 100%, F1 100%, recall 99%.",
        "Claimed Merit": "Provides respiratory-imaging Grad-CAM procedure and examples.", "Limitations": "Different labels and pooled sources; does not justify DenseNet/ResNet/YOLO training hyperparameters.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Supports the final explainability workflow, not the present classifier settings.",
        "Source Obtained From": "Supplied image; Publisher page", "Verification Status": "Verified",
        "Notes or Corrections": "Both the 2021 online date and 2022 issue year are preserved.",
    }),
    row(**{
        "Reference ID": "DATA-KERMANY", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Dataset and exact task provenance",
        "Authors": "Daniel S. Kermany et al.", "Year": "2018", "Article Title": "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning",
        "Venue": "Cell 172(5):1122-1131.e9", "DOI": "10.1016/j.cell.2018.02.010",
        "Official Link": "https://www.sciencedirect.com/science/article/pii/S0092867418301545",
        "Research Objective": "Demonstrate transfer learning for retinal OCT and pediatric CXR diagnosis.",
        "Methodology": "Image-based transfer learning, including pediatric chest radiographs labeled Normal, Bacterial Pneumonia, and Viral Pneumonia.",
        "Task and Classes": "Includes the exact three-class pediatric CXR labels", "Dataset": "Original public pediatric CXR source used by the repository",
        "Architecture or Method": "Transfer-learning image classifiers", "Claimed Merit": "Primary dataset/task provenance.",
        "Limitations": "Dataset paper does not automatically determine the present split, deduplication, or optimizer settings.",
        "Evidence Applicability": "Direct", "Evidence Applicability Reason": "Primary evidence for the exact labels and source-domain task.",
        "Source Obtained From": "Publisher page", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "ARCH-DENSENET", "Screenshot Journal ID": "Not in supplied image", "Research Category": "General medical-imaging architecture background",
        "Authors": "Gao Huang; Zhuang Liu; Laurens van der Maaten; Kilian Q. Weinberger", "Year": "2017",
        "Article Title": "Densely Connected Convolutional Networks", "Venue": "CVPR 2017:4700-4708",
        "Official Link": "https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html",
        "Research Objective": "Introduce dense feed-forward connectivity for feature reuse and gradient flow.",
        "Methodology": "Dense blocks connect each layer to all later layers; ImageNet/CIFAR/SVHN benchmarks include DenseNet121/169/201 families.",
        "Task and Classes": "General image classification", "Architecture or Method": "DenseNet",
        "Claimed Merit": "Architecture provenance for DenseNet121, DenseNet169, and DenseNet201.",
        "Limitations": "Not pneumonia-specific and does not determine medical preprocessing or training configuration.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "General architecture paper; supports model identity, not experiment settings.",
        "Source Obtained From": "Official CVF paper repository", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "ARCH-RESNET", "Screenshot Journal ID": "Not in supplied image", "Research Category": "General medical-imaging architecture background",
        "Authors": "Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun", "Year": "2016", "Article Title": "Deep Residual Learning for Image Recognition",
        "Venue": "CVPR 2016:770-778", "Official Link": "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
        "Research Objective": "Introduce residual learning to train very deep image networks.", "Methodology": "Residual blocks evaluated on ImageNet and other vision tasks.",
        "Task and Classes": "General image classification", "Architecture or Method": "ResNet50/101/152 family",
        "Claimed Merit": "Architecture provenance for the three ResNet baselines.", "Limitations": "Not pneumonia-specific and does not determine medical preprocessing or training hyperparameters.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "General architecture paper; supports model identity, not experiment settings.",
        "Source Obtained From": "Official CVF paper repository", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "CXR-CHEXNET", "Screenshot Journal ID": "Not in supplied image", "Research Category": "CNN chest X-ray pneumonia classification",
        "Authors": "Pranav Rajpurkar; Jeremy Irvin; Kaylie Zhu; Brandon Yang; Hershel Mehta; Tony Duan; Daisy Ding; Aarti Bagul; Curtis Langlotz; Katie Shpanskaya; Matthew P. Lungren; Andrew Y. Ng",
        "Year": "2017", "Article Title": "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning",
        "Venue": "arXiv:1711.05225", "Official Link": "https://arxiv.org/abs/1711.05225",
        "Research Objective": "Detect pneumonia and chest pathologies on frontal CXR.", "Methodology": "DenseNet121 trained on ChestX-ray14; heatmap localization discussed.",
        "Task and Classes": "Binary/multilabel CXR pathology classification", "Dataset": "NIH ChestX-ray14", "Architecture or Method": "DenseNet121",
        "Claimed Merit": "Strong chest-radiograph precedent for DenseNet121.", "Limitations": "Different adult dataset and labels; arXiv study does not justify the present three-class configuration.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Directly supports DenseNet on CXR, but not the exact classes or dataset.",
        "Source Obtained From": "Official repository", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "XAI-GRADCAM", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Explainability and Grad-CAM for respiratory imaging",
        "Authors": "Ramprasaath R. Selvaraju et al.", "Year": "2017", "Article Title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization",
        "Venue": "ICCV 2017:618-626", "Official Link": "https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html",
        "Research Objective": "Generate class-discriminative visual explanations from convolutional networks.",
        "Methodology": "Gradient-weighted activation maps from a target convolutional layer.", "Task and Classes": "General visual explainability",
        "Architecture or Method": "Grad-CAM", "Claimed Merit": "Primary method reference for final localization heatmaps.",
        "Limitations": "Heatmaps are explanations, not proof of causality, segmentation accuracy, or clinical correctness.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Directly supports explainability procedures but not classifier hyperparameters.",
        "Source Obtained From": "Official CVF paper repository", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "PREP-CLAHE", "Screenshot Journal ID": "Not in supplied image", "Research Category": "CXR preprocessing evidence",
        "Authors": "Anggi Muhammad Rifai; Suwanto Raharjo; Ema Utami; Dhani Ariatmanto", "Year": "2024 journal issue; online 2023",
        "Article Title": "Analysis for diagnosis of pneumonia symptoms using chest X-ray based on MobileNetV2 models with image enhancement using white balance and contrast limited adaptive histogram equalization (CLAHE)",
        "Venue": "Biomedical Signal Processing and Control 105857", "DOI": "10.1016/j.bspc.2023.105857",
        "Official Link": "https://www.sciencedirect.com/science/article/pii/S1746809423012909",
        "Research Objective": "Test MobileNetV2 with white balance and CLAHE for pneumonia CXR classification.",
        "Methodology": "Pretrained MobileNetV2; white-balance correction plus CLAHE; two- and three-class experiments, including a 50-epoch comparison.",
        "Task and Classes": "Binary and three-class CXR classification", "Architecture or Method": "MobileNetV2",
        "Preprocessing": "White balance plus CLAHE", "Epochs": "Best reported at 50 epochs",
        "Reported Results": "91.17% three-class accuracy; 99.76% binary accuracy.", "Claimed Merit": "Supports a controlled CLAHE ablation on CXR.",
        "Limitations": "CLAHE is combined with white balance, so its isolated effect is not established; architecture and protocol differ.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "CXR and three-class contrast enhancement are relevant, but the transform is bundled and the model differs.",
        "Source Obtained From": "Publisher page", "Verification Status": "Verified",
        "Notes or Corrections": "Bibliographic author metadata cross-checked through the DOI-linked publisher record and DBLP.",
    }),
    row(**{
        "Reference ID": "PREP-HE", "Screenshot Journal ID": "Not in supplied image", "Research Category": "CXR preprocessing evidence",
        "Authors": "Jose Raniery Ferreira; Diego Armando Cardona Cardenas; Ramon Alfredo Moreno; Marina de Fatima de Sa Rebelo; Jose Eduardo Krieger; Marco Antonio Gutierrez",
        "Year": "2020", "Article Title": "Multi-View Ensemble Convolutional Neural Network to Improve Classification of Pneumonia in Low Contrast Chest X-Ray Images",
        "Venue": "2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society, 1238-1241",
        "DOI": "10.1109/EMBC44109.2020.9176517", "Official Link": "https://pubmed.ncbi.nlm.nih.gov/33018211/",
        "Research Objective": "Improve low-contrast CXR classification and distinguish bacterial from viral pneumonia.",
        "Methodology": "VGG16 with custom MLP; compared original, chest-cavity cropping, histogram-equalized segmented images, and a multi-view ensemble.",
        "Task and Classes": "Normal versus Pneumonia and Bacterial versus Viral CXR classification", "Dataset": "Public pediatric pneumonia CXR dataset",
        "Architecture or Method": "VGG16 multi-view ensemble", "Preprocessing": "Chest-cavity cropping; global histogram equalization",
        "Selection or Evaluation Metric": "AUC with 95% CI plus accuracy, sensitivity, specificity, F1",
        "Reported Results": "Ensemble AUC 0.97 (95% CI 0.96-0.99) for Pneumonia versus Normal and 0.91 (0.88-0.94) for Bacterial versus Viral; all preprocessed models exceeded the original-image baseline AUC.",
        "Claimed Merit": "Strong task-specific rationale for a global histogram-equalization ablation.",
        "Limitations": "Uses segmentation/cropping and an ensemble, so the isolated global-HE effect on the current models remains experimental.",
        "Evidence Applicability": "Direct", "Evidence Applicability Reason": "Same CXR source-domain and target distinctions; directly motivates—but does not predetermine—the controlled HE ablation.",
        "Source Obtained From": "PubMed; IEEE DOI metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "PREP-PSEUDOCOLOR", "Screenshot Journal ID": "Not in supplied image", "Research Category": "CXR preprocessing evidence",
        "Authors": "Morteza Heidari; Seyedehnafiseh Mirniaharikandehei; Abolfazl Zargari Khuzani; Gopichandh Danala; Yuchen Qiu; Bin Zheng",
        "Year": "2020", "Article Title": "Improving the performance of CNN to predict the likelihood of COVID-19 using chest X-ray images with preprocessing algorithms",
        "Venue": "International Journal of Medical Informatics 144:104284", "DOI": "10.1016/j.ijmedinf.2020.104284",
        "Official Link": "https://pubmed.ncbi.nlm.nih.gov/32992136/", "Research Objective": "Test a pseudo-color three-channel CXR representation for COVID/other-pneumonia/normal classification.",
        "Methodology": "Crop diaphragm, histogram equalization, bilateral filtering, then place original/equalized/filtered images into separate channels of a transfer-learning CNN.",
        "Task and Classes": "Three-class CXR: COVID-19 pneumonia, other community-acquired pneumonia, Normal", "Dataset": "8,474 pooled CXRs",
        "Architecture or Method": "Transfer-learning CNN/VGG16", "Preprocessing": "Constructed pseudo-color from original, histogram-equalized, and bilateral-filtered channels",
        "Reported Results": "94.5% three-class accuracy versus 88.0% without the two preprocessing steps; 98.4% COVID sensitivity and 98.0% specificity.",
        "Claimed Merit": "Shows that information-bearing multi-channel preprocessing can help a related three-class CXR task.",
        "Limitations": "Not a VIRIDIS colormap; different labels and pooled datasets. It cannot be cited as evidence that VIRIDIS is beneficial.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Related three-class CXR pseudo-color evidence, but the construction is fundamentally different from VIRIDIS.",
        "Source Obtained From": "PubMed; PubMed Central; Publisher DOI metadata", "Verification Status": "Verified",
        "Notes or Corrections": "Explicitly do not conflate this constructed three-channel pseudo-color method with OpenCV COLORMAP_VIRIDIS.",
    }),
    row(**{
        "Reference ID": "BIAS-SHORTCUT", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Validation and shortcut-learning evidence",
        "Authors": "Anusua Trivedi; Caleb Robinson; Marian Blazes; Anthony Ortiz; Jocelyn Desbiens; Sunil Gupta; Rahul Dodhia; Pavan K. Bhatraju; W. Conrad Liles; Jayashree Kalpathy-Cramer; Aaron Y. Lee; Juan M. Lavista Ferres",
        "Year": "2022", "Article Title": "Deep learning models for COVID-19 chest x-ray classification: Preventing shortcut learning using feature disentanglement",
        "Venue": "PLOS ONE 17(10):e0274098", "DOI": "10.1371/journal.pone.0274098", "Official Link": "https://pubmed.ncbi.nlm.nih.gov/36201483/",
        "Research Objective": "Reduce source-dataset shortcut learning in pooled CXR classification.",
        "Methodology": "Penalized source-discriminative representations using feature disentanglement and evaluated unseen-source generalization.",
        "Task and Classes": "COVID-19 CXR classification across sources", "Architecture or Method": "Feature disentanglement",
        "Preprocessing": "Compared with masking and histogram equalization", "Reported Results": "Best held-out AUC improvement 0.13; feature disentanglement outperformed masking and histogram equalization.",
        "Claimed Merit": "Justifies leakage control, source-aware caution, and validation of enhancements rather than assuming benefit.",
        "Limitations": "COVID/source-bias task differs from the exact pediatric classes.",
        "Evidence Applicability": "Partial", "Evidence Applicability Reason": "Directly relevant to CXR shortcut risk and preprocessing interpretation, not to model architecture selection alone.",
        "Source Obtained From": "PubMed; Publisher page", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "VALID-CLINICAL", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Clinical validation evidence",
        "Authors": "Jeremy Hofmeister; Nicolas Garin; Xavier Montet; Max Scheffler; Alexandra Platon; Pierre-Alexandre Poletti; Jerome Stirnemann; Marie-Pierre Debray; Yann-Erick Claessens; Xavier Duval; Virginie Prendki", "Year": "2024", "Article Title": "Validating the accuracy of deep learning for the diagnosis of pneumonia on chest x-ray against a robust multimodal reference diagnosis: a post hoc analysis of two prospective studies",
        "Venue": "European Radiology Experimental 8:20", "DOI": "10.1186/s41747-023-00416-y", "Official Link": "https://pubmed.ncbi.nlm.nih.gov/38302850/",
        "Research Objective": "Validate pneumonia CXR deep learning against a multidisciplinary multimodal reference diagnosis.",
        "Methodology": "Post hoc analysis of two prospective cohorts totaling 430 patients; compared DL, clinicians, radiologists, and CT.",
        "Task and Classes": "Clinical pneumonia diagnosis", "Reported Results": "DL AUC 0.74; accuracy 71.0% and 72.6% in the two cohorts, below many benchmark reports but similar to radiologists on CXR.",
        "Claimed Merit": "Provides an essential boundary between benchmark performance and clinical diagnostic validity.",
        "Limitations": "Different cohorts and binary clinical diagnosis; not a source for training hyperparameters.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Clinical-translation caution only; it constrains claims, not model configuration.",
        "Source Obtained From": "PubMed; Publisher metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "CALIBRATION", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Calibration and reliability evidence",
        "Authors": "Chuan Guo; Geoff Pleiss; Yu Sun; Kilian Q. Weinberger", "Year": "2017", "Article Title": "On Calibration of Modern Neural Networks",
        "Venue": "ICML 2017, PMLR 70:1321-1330", "Official Link": "https://proceedings.mlr.press/v70/guo17a.html",
        "Research Objective": "Measure and improve probabilistic calibration of modern classifiers.", "Methodology": "Compared calibration properties and post-hoc methods including temperature scaling.",
        "Task and Classes": "General classification calibration", "Architecture or Method": "Expected calibration error; temperature scaling",
        "Claimed Merit": "Primary evidence for post-hoc temperature scaling and calibration reporting.",
        "Limitations": "Not medical or pneumonia-specific; calibration must be fit on validation only and assessed after final freeze.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Supports reliability methods but not base classifier hyperparameters.",
        "Source Obtained From": "Official PMLR paper repository", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMPL-TORCHVISION", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Official implementation provenance",
        "Authors": "PyTorch/Torchvision project", "Year": "torchvision 0.20.1 used by repository", "Article Title": "Torchvision DenseNet and ResNet model/weight documentation",
        "Venue": "Official torchvision documentation", "Official Link": "https://docs.pytorch.org/vision/0.20/models.html",
        "Research Objective": "Document official model constructors, weight enums, and reference transforms.",
        "Methodology": "Repository uses DEFAULT ImageNet weights for DenseNet121/169/201 and ResNet50/101/152, then replaces the classifier with a three-logit linear layer.",
        "Task and Classes": "Implementation provenance for six supported classifiers", "Architecture or Method": "DenseNet121/169/201; ResNet50/101/152",
        "Input Size": "Official weight recipes use 224 crops", "Preprocessing": "Official reference weights scale to [0,1] and use ImageNet mean/std; resize/crop recipes vary by weight enum",
        "Pretrained Weights": "DEFAULT ImageNet weight enums", "Fine-Tuning Strategy": "Repository fine-tunes the full network unless freeze_backbone is explicitly enabled",
        "Claimed Merit": "Verifies the exact code/weight source for six baselines.", "Limitations": "Official recipes are implementation references, not pneumonia-specific experimental evidence.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Authoritative implementation provenance; no direct medical result.",
        "Source Obtained From": "Official documentation; Existing repository code; Checkpoint metadata", "Verification Status": "Verified",
        "Notes or Corrections": "The common repository transform is intentionally shared across all eight models for a fair comparison; native per-model recipes are not silently mixed.",
    }),
    row(**{
        "Reference ID": "IMPL-YOLOV5", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Official implementation provenance",
        "Authors": "Ultralytics", "Year": "YOLOv5 v7.0 asset; classification introduced in v6.2", "Article Title": "Ultralytics YOLOv5 classification release and repository",
        "Venue": "Official GitHub software release", "Official Link": "https://github.com/ultralytics/yolov5/releases",
        "Research Objective": "Provide official YOLOv5 image-classification architecture and ImageNet checkpoints.",
        "Methodology": "Repository loads yolov5s-cls-v7.0.pt, verifies a Classify head, replaces the linear output with three logits, and trains through the common PyTorch pipeline.",
        "Task and Classes": "Implementation provenance for YOLOv5s-cls adapted to three classes", "Architecture or Method": "YOLOv5s-cls",
        "Input Size": "224 x 224 in current repository", "Preprocessing": "Common repository transform, not native YOLO trainer augmentation",
        "Pretrained Weights": "Ultralytics YOLOv5 v7.0 ImageNet classification checkpoint", "Fine-Tuning Strategy": "Full-network fine-tuning in current baseline",
        "Claimed Merit": "Verifies a genuine classification head and official checkpoint.", "Limitations": "Software/release evidence, not pneumonia-specific training evidence.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Authoritative implementation provenance only.",
        "Source Obtained From": "Official repository/release; Existing repository code; Checkpoint metadata", "Verification Status": "Verified",
    }),
    row(**{
        "Reference ID": "IMPL-YOLOV8", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Official implementation provenance",
        "Authors": "Ultralytics", "Year": "Ultralytics 8.3.168 environment; v8.3.0 checkpoint asset", "Article Title": "Ultralytics YOLO image-classification documentation and YOLOv8 classification model",
        "Venue": "Official Ultralytics documentation/software", "Official Link": "https://docs.ultralytics.com/tasks/classify/",
        "Research Objective": "Document official YOLO classification models and training/inference transforms.",
        "Methodology": "Repository loads yolov8n-cls-v8.3.0.pt, verifies/replaces its Classify head with three logits, adapts tuple eval outputs to raw logits, and trains through the common pipeline.",
        "Task and Classes": "Implementation provenance for YOLOv8n-cls adapted to three classes", "Architecture or Method": "YOLOv8n-cls",
        "Input Size": "224 x 224 in current repository", "Preprocessing": "Common repository transform; native Ultralytics training defaults such as RandomResizedCrop/RandAugment/erasing are not invoked",
        "Pretrained Weights": "Ultralytics v8.3.0 ImageNet classification checkpoint", "Fine-Tuning Strategy": "Full-network fine-tuning in current baseline",
        "Claimed Merit": "Verifies a genuine classification head, checkpoint, and the tuple-output adapter correction.",
        "Limitations": "No formal YOLOv8 paper; software docs do not provide pneumonia-specific hyperparameters.",
        "Evidence Applicability": "Background", "Evidence Applicability Reason": "Authoritative implementation provenance only.",
        "Source Obtained From": "Official documentation; Installed package source; Existing repository code; Checkpoint metadata", "Verification Status": "Verified",
        "Notes or Corrections": "Native Ultralytics classification defaults in 8.3.168 include RandomResizedCrop, RandAugment, and random erasing, but the controlled repository deliberately bypasses them for identical transforms across architectures.",
    }),
    row(**{
        "Reference ID": "EXCL-YOLOV10", "Screenshot Journal ID": "Not in supplied image", "Research Category": "Excluded architecture verification",
        "Authors": "Ao Wang et al.; THU-MIG", "Year": "2024", "Article Title": "YOLOv10: Real-Time End-to-End Object Detection",
        "Venue": "NeurIPS 2024; official THU-MIG repository", "Official Link": "https://github.com/THU-MIG/yolov10",
        "Research Objective": "Develop an end-to-end real-time object detector without non-maximum suppression.",
        "Methodology": "Detection architecture and checkpoints; official task support is object detection.",
        "Task and Classes": "Object detection", "Architecture or Method": "YOLOv10 detector",
        "Claimed Merit": "Authoritative evidence for the model's actual supported task.",
        "Limitations": "No genuine official YOLOv10 image-classification checkpoint/head compatible with the eight-model classifier protocol was found.",
        "Evidence Applicability": "Excluded", "Evidence Applicability Reason": "Detection-only evidence cannot be relabeled as three-class image classification.",
        "Source Obtained From": "Official repository; Official NeurIPS paper; Official Ultralytics model documentation", "Verification Status": "Verified",
        "Notes or Corrections": "A generic classification class present in a forked codebase is not evidence of a trained/supported YOLOv10 classifier.",
    }),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "research_evidence_matrix.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(RECORDS)

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds"),
        "record_count": len(RECORDS),
        "screenshot_visible_entry_count": 10,
        "applicability_allowed_values": ["Direct", "Partial", "Background", "Excluded"],
        "source_policy": "The supplied screenshot is a supplementary reference index. Claims are recorded only after primary-source verification; missing fields remain 'Not reported'.",
        "required_link_instruction": "Open and verify every available reference link relevant to the eight supported models. When the link column is blank, incomplete, inaccessible, or only visible through the supplied image, locate the original publication using the exact title, authors, year, DOI, or publisher.",
        "records": RECORDS,
    }
    (OUT / "research_evidence_matrix.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    screen_rows = [record for record in RECORDS if record["Reference ID"].startswith("IMG-")]
    counts = {value: sum(record["Evidence Applicability"] == value for record in RECORDS) for value in ("Direct", "Partial", "Background", "Excluded")}
    screen_table = "\n".join(
        f"| {record['Screenshot Journal ID']} | {record['Authors'].split(';')[0]} et al. | {record['Year']} | {record['Article Title']} | {record['Evidence Applicability']} | {record['Evidence Applicability Reason']} |"
        for record in screen_rows
    )
    md = f"""# Verified Research Evidence Matrix

Generated: {payload['generated_at']}

The supplied spreadsheet screenshot is treated as a supplementary index only. The repository contains no original RRL spreadsheet or independent reference-list document, and the screenshot's Reference Link cells are blank. Each of its ten visible study rows was therefore located by title/authors/year and checked against a publisher page, DOI record, official repository, or another verified primary source.

## Required reference-link rule

> {payload['required_link_instruction']}

## Applicability summary

The complete machine-readable matrix contains {len(RECORDS)} verified records: **{counts['Direct']} Direct**, **{counts['Partial']} Partial**, **{counts['Background']} Background**, and **{counts['Excluded']} Excluded**. “Direct” means the source closely matches the Normal/Viral/Bacterial CXR task or its primary dataset. It does not mean that every reported hyperparameter should be copied.

| Screenshot ID | First author | Verified year | Verified title | Applicability | Reason |
|---|---|---|---|---|---|
{screen_table}

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
"""
    (DOCS / "07_reference_evidence_matrix_continuation_v2.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
