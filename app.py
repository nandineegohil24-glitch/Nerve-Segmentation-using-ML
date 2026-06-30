import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 128
MODEL_PATH = "attention_unet.h5"

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

st.set_page_config(page_title="Ultrasound Segmentation", layout="wide")
st.title("Ultrasound Segmentation")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    with st.spinner("Running segmentation..."):
        pred = model.predict(img_input)[0, :, :, 0]
        pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Create overlay
    overlay = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay[pred_mask == 255] = [255, 0, 0]

    # -----------------------------
    # Display Results
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_resized, caption="Original Image", use_column_width=True, channels="GRAY")
    with col2:
        st.image(pred_mask, caption="Predicted Mask", use_column_width=True, channels="GRAY")
    with col3:
        st.image(overlay, caption="Overlay (Prediction)", use_column_width=True)

    # Option to download mask
    result_img = Image.fromarray(pred_mask)
    st.download_button(
        label="⬇️ Download Predicted Mask",
        data=cv2.imencode('.png', pred_mask)[1].tobytes(),
        file_name="predicted_mask.png",
        mime="image/png"
    )
