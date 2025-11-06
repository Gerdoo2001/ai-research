# src/model/unet/train_unet.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from model.unet.model import U_Net, dice, dice_loss
from model.unet.data_augment import AUGMENTATIONS_TRAIN
from model.unet.data_generator import getData

# GPU safety
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50

# Prepare data
print("📂 Loading dataset...")
X, Y = getData(IMG_SIZE, flag="train")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print("✅ Dataset loaded:", X_train.shape, Y_train.shape)

# Optional: apply augmentations
# from model.unet.utils import apply_augmentations  # if you want to modularize it
# X_train, Y_train = apply_augmentations(X_train, Y_train)

# Model
model = U_Net(input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
              loss=dice_loss,
              metrics=[dice, "binary_accuracy"])

# Ensure output directory
os.makedirs("model/outputs", exist_ok=True)

# Callbacks
weight_path = "model/outputs/unet_lung_segmentation.best.hdf5"
callbacks = [
    ModelCheckpoint(weight_path, monitor="val_dice", save_best_only=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_dice", factor=0.5, patience=3, mode="max", verbose=1),
    EarlyStopping(monitor="val_dice", patience=8, mode="max", restore_best_weights=True)
]

# Training
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("✅ Training complete. Best weights saved to:", weight_path)
