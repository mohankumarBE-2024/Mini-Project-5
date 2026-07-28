import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Solar Panel Fault Detector",
    page_icon="☀️",
    layout="centered"
)

st.title("☀️ Solar Panel Condition Classifier")
st.write(
    "Upload an image of a solar panel to detect faults, "
    "environmental coverage, or physical damage."
)

CLASS_NAMES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

MODEL_PATH = "solar_panel_model.keras"
MODEL_URL = "https://huggingface.co/mohanbe2024/solar-guard-model/resolve/main/solar_panel_model.keras"


def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (first run only)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


@st.cache_resource
def load_solar_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)


model = None

try:
    model = load_solar_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")


uploaded_file = st.file_uploader(
    "Choose a solar panel image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if model is None:

        st.warning(
            "Prediction is currently unavailable.\n\n"
            "The trained model has not been added yet."
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Condition", "Not Available")

        with col2:
            st.metric("Confidence", "-- %")

    else:

        with st.spinner("🔄 Analyzing panel condition..."):

            img = image.resize((224, 224))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0) / 255.0

            probabilities = model.predict(img, verbose=0)[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index] * 100

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Condition", predicted_class)

            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
