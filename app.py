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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #2b1a05 0%, #0E1117 55%) !important;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
}
[data-testid="stFileUploader"] {
    background-color: #1C1F26;
    border: 1px solid #F7931E33;
    border-radius: 12px;
    padding: 1.2rem;
}
[data-testid="stMetric"] {
    background-color: #1C1F26;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #F7931E33;
}
div[data-baseweb="notification"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>☀️ SolarGuard</h1>"
    "<p style='text-align:center; color:gray;'>AI-Powered Solar Panel Condition Classifier</p>",
    unsafe_allow_html=True
)

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

STATUS_MAP = {
    "Clean":              {"icon": "✅", "level": "success", "note": "Panel is operating at optimal efficiency."},
    "Dusty":               {"icon": "🌫️", "level": "warning", "note": "Dust buildup may be reducing output. Cleaning recommended."},
    "Bird-drop":           {"icon": "🐦", "level": "warning", "note": "Localized obstruction detected. Spot-cleaning recommended."},
    "Snow-Covered":        {"icon": "❄️", "level": "info",    "note": "Panel currently obstructed by snow."},
    "Electrical-damage":   {"icon": "⚡", "level": "error",   "note": "Potential electrical fault — inspection advised."},
    "Physical-Damage":     {"icon": "🔨", "level": "error",   "note": "Structural damage detected — inspection advised."},
}

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

    if model is None:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.warning(
            "Prediction is currently unavailable.\n\n"
            "The trained model has not been added yet."
        )
    else:
        with st.spinner("🔄 Analyzing panel condition..."):
            img = image.resize((224, 224))
            img_arr = tf.keras.utils.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0) / 255.0
            probabilities = model.predict(img_arr, verbose=0)[0]
            predicted_index = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index] * 100

        status = STATUS_MAP[predicted_class]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown(f"### {status['icon']} {predicted_class}")
            getattr(st, status["level"])(status["note"])
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(float(confidence) / 100)

        st.divider()
        st.markdown("#### Full Prediction Breakdown")
        for name, prob in sorted(zip(CLASS_NAMES, probabilities), key=lambda x: -x[1]):
            st.markdown(f"{STATUS_MAP[name]['icon']} **{name}**")
            st.progress(float(prob))
