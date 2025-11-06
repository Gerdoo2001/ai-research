# src/model/unet/model.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

# ================================
# DICE METRICS
# ================================
def dice(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

# ================================
# U-NET MODEL WITH EfficientNetB4 BACKBONE
# ================================
def U_Net(input_shape=(256, 256, 3), dropout_rate=0.25):
    """U-Net with EfficientNetB4 encoder backbone"""
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Skip connections (same layers referenced in original code)
    skip_connections = [
        base_model.get_layer('block2a_expand_activation').output,
        base_model.get_layer('block3a_expand_activation').output,
        base_model.get_layer('block4a_expand_activation').output,
        base_model.get_layer('block6a_expand_activation').output
    ]

    x = base_model.output

    # Decoder
    for skip in reversed(skip_connections):
        x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# ================================
# COMPILE FUNCTION
# ================================
def compile_model(model, lr=2e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=dice_loss,
        metrics=[dice, 'binary_accuracy']
    )
    return model

# ================================
# MAIN CHECK
# ================================
if __name__ == "__main__":
    model = U_Net()
    model = compile_model(model)
    model.summary()
