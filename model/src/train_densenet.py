"""
Train DenseNet121 for Pneumonia Classification
Author: John Mark

Description:
    Loads augmented dataset (original + flipped + CLAHE)
    Trains a binary classifier: NORMAL vs PNEUMONIA
    Saves model as densenet121_pneumonia.h5 (or .keras if .h5 fails)
"""

import os
import gc
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# ==============================
# PATHS (absolute, robust)
# ==============================
BASE_DIR = Path(__file__).resolve().parents[2]  # -> ai-research/
TRAIN_DIR = BASE_DIR / "model" / "augmented_data" / "train"
OUTPUT_DIR = BASE_DIR / "model" / "outputs"
MODEL_SAVE_PATH = OUTPUT_DIR / "densenet121_pneumonia.h5"
MODEL_SAFE_PATH = OUTPUT_DIR / "densenet121_pneumonia_safe.h5"
KERAS_SAVE_PATH = OUTPUT_DIR / "densenet121_pneumonia.keras"
HISTORY_PLOT_PATH = OUTPUT_DIR / "training_history.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

if train_gen.samples == 0 or val_gen.samples == 0:
    raise RuntimeError(f"❌ No images found in {TRAIN_DIR}. Check folder structure.")

print("✅ Dataset loaded successfully.")
print(f"Classes: {train_gen.class_indices}")

# ==============================
# MODEL CREATION
# ==============================
print("🧠 Building DenseNet121 model...")

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAINING
# ==============================
print("🚀 Starting training...")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ==============================
# SAVE MODEL (Safe Save)
# ==============================
print(f"💾 Saving trained model to {MODEL_SAVE_PATH} ...")
try:
    model.save(MODEL_SAVE_PATH, save_format="h5")
    print(f"✅ Model saved successfully at: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"⚠️ H5 save failed: {e}")
    try:
        print(f"💾 Retrying save to {MODEL_SAFE_PATH} ...")
        model.save(MODEL_SAFE_PATH, save_format="h5")
        print(f"✅ Backup H5 model saved at: {MODEL_SAFE_PATH}")
    except Exception as e2:
        print(f"⚠️ Backup H5 failed too: {e2}")
        print(f"💾 Saving instead as native .keras format...")
        model.save(KERAS_SAVE_PATH)
        print(f"✅ Model saved as .keras: {KERAS_SAVE_PATH}")

# Free memory
del model
gc.collect()

# ==============================
# PLOT TRAINING HISTORY
# ==============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig(HISTORY_PLOT_PATH)
plt.show()

print(f"✅ Training complete! Charts saved as {HISTORY_PLOT_PATH}")
