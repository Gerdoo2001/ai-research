import streamlit as st

# =====================================================
# STREAMLIT PAGE CONFIG (FIRST & ONLY)
# =====================================================
st.set_page_config(
    page_title="Lung Segmentation Comparison",
    layout="wide"
)

# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

# =====================================================
# MODEL ARCHITECTURES (weights-only)
# =====================================================
def unet_full(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([
        Conv2DTranspose(256, 2, strides=2, padding='same')(conv5),
        conv4
    ])
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([
        Conv2DTranspose(128, 2, strides=2, padding='same')(conv6),
        conv3
    ])
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([
        Conv2DTranspose(64, 2, strides=2, padding='same')(conv7),
        conv2
    ])
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([
        Conv2DTranspose(32, 2, strides=2, padding='same')(conv8),
        conv1
    ])
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    return Model(inputs, outputs)

# =====================================================
# LOAD BOTH MODELS (CACHED)
# =====================================================
@st.cache_resource
def load_models():
    model_a = unet_full()
    model_a.load_weights("../outputs/cxr_reg_weights.best.hdf5")

    model_b = load_model("../outputs/trained_model.hdf5", compile=False)
    return model_a, model_b

model_a, model_b = load_models()

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🫁 Lung Segmentation — Model Comparison")

st.write(
    "This view runs **two segmentation models on the same X-ray** "
    "to visually compare performance and differences."
)

threshold = st.slider(
    "Segmentation Threshold",
    0.1, 0.9, 0.5, 0.05
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img_norm = img / 255.0
    input_tensor = np.expand_dims(img_norm, axis=(0, -1))

    pred_a = model_a.predict(input_tensor)[0, :, :, 0]
    pred_b = model_b.predict(input_tensor)[0, :, :, 0]

    bin_a = (pred_a > threshold).astype(np.uint8)
    bin_b = (pred_b > threshold).astype(np.uint8)

    def make_overlay(base, mask):
        heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(
            cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
            0.7,
            heat,
            0.3,
            0
        )

    overlay_a = make_overlay(img, pred_a)
    overlay_b = make_overlay(img, pred_b)

    diff_map = np.abs(bin_a - bin_b)

    st.subheader("Original X-ray")
    st.image(img, clamp=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🅰️ U-Net (CXR-Reg)")
        st.image(pred_a, clamp=True)
        st.image(overlay_a, channels="BGR")

    with col2:
        st.markdown("### 🅱️ U-Net (Legacy Trained Model)")
        st.image(pred_b, clamp=True)
        st.image(overlay_b, channels="BGR")

    st.subheader("🔍 Difference Map")
    st.image(diff_map * 255, clamp=True)

    st.success("Model comparison completed ✅")
else:
    st.info("Upload a chest X-ray image to start comparison.")
