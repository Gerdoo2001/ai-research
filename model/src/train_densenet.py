"""
Train DenseNet Models (121 / 169 / 201) for Pneumonia Classification
Author: John Mark Bolongan

Description:
    Loops through DenseNet variants, trains on augmented dataset,
    and generates:
        - Accuracy & Loss curves
        - Confusion matrix
        - Classification report (text + heatmap)
        - Precision/Recall/F1 bar chart
        - ROC curve with AUC
"""

import os
import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DENSENET_MODELS = ["121", "169", "201"]

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parents[2]  # -> ai-research/
TRAIN_DIR = BASE_DIR / "model" / "augmented_data" / "train"
OUTPUT_ROOT = BASE_DIR / "model" / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ==============================
# DATASET LOADING
# ==============================
print("📂 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

if train_gen.samples == 0 or val_gen.samples == 0:
    raise RuntimeError(f"❌ No images found in {TRAIN_DIR}. Check folder structure.")

print("✅ Dataset loaded successfully.")
print(f"Classes: {train_gen.class_indices}")

# ==============================
# BUILD FUNCTION
# ==============================
def build_densenet(version: str):
    print(f"🧠 Building DenseNet{version}...")
    if version == "121":
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif version == "169":
        base_model = DenseNet169(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif version == "201":
        base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Invalid DenseNet version")

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

# ==============================
# LOOP THROUGH EACH MODEL
# ==============================
for version in DENSENET_MODELS:
    print(f"\n===============================")
    print(f"🚀 TRAINING DENSENET{version}")
    print(f"===============================")

    OUTPUT_DIR = OUTPUT_ROOT / f"densenet{version}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save paths
    MODEL_PATH = OUTPUT_DIR / f"densenet{version}_pneumonia.h5"
    HISTORY_PLOT_PATH = OUTPUT_DIR / "training_history.png"
    CONFUSION_PLOT_PATH = OUTPUT_DIR / "confusion_matrix.png"
    REPORT_TXT_PATH = OUTPUT_DIR / "classification_report.txt"
    REPORT_HEATMAP_PATH = OUTPUT_DIR / "classification_report_heatmap.png"
    PRF_BAR_PATH = OUTPUT_DIR / "precision_recall_f1_bars.png"
    ROC_PATH = OUTPUT_DIR / "roc_curve.png"

    # Build + train
    model = build_densenet(version)
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, verbose=1)

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

    # Predictions
    val_gen.reset()
    pred_probs = model.predict(val_gen)
    preds = (pred_probs > 0.5).astype("int32").flatten()
    true_labels = val_gen.classes
    labels = list(val_gen.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"📊 Confusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(f"DenseNet{version} Confusion Matrix")
    plt.savefig(CONFUSION_PLOT_PATH)
    plt.close()

    # Classification Report (text)
    report_str = classification_report(true_labels, preds, target_names=labels)
    with open(REPORT_TXT_PATH, "w") as f:
        f.write(report_str)

    # Classification Report (heatmap)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, preds, zero_division=0)
    metrics_matrix = np.array([precision, recall, f1]).T
    plt.figure(figsize=(6, 4))
    sns.heatmap(metrics_matrix, annot=True, cmap="YlGnBu",
                xticklabels=["Precision", "Recall", "F1-Score"],
                yticklabels=labels, fmt=".2f")
    plt.title(f"DenseNet{version} Classification Report")
    plt.tight_layout()
    plt.savefig(REPORT_HEATMAP_PATH)
    plt.close()

    # Precision/Recall/F1 Bar Chart
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-Score")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title(f"DenseNet{version} Metrics per Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRF_BAR_PATH)
    plt.close()

    # ROC Curve + AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"DenseNet{version} ROC Curve (AUC = {roc_auc:.3f})")
    plt.legend(loc="lower right")
    plt.savefig(ROC_PATH)
    plt.close()

    # Accuracy/Loss Curves
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend()
    plt.title(f"DenseNet{version} Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend()
    plt.title(f"DenseNet{version} Loss")

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close()

    print(f"📈 Saved all plots for DenseNet{version}")
    del model
    gc.collect()

print("✅ Training complete for all DenseNet models!")
