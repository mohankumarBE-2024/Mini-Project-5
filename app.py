import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

st.set_page_config(
    page_title="Solar Panel Fault Detector",
    page_icon="☀️",
    layout="centered"
)


BACKGROUND_IMAGE = os.path.join(
    os.path.dirname(__file__),
    "solar_panel.jpg"
)

with open(BACKGROUND_IMAGE, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode()

st.markdown(
    f"""
    <style>

    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.65)
            ),
            url("data:image/jpeg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0) !important;
    }}

    [data-testid="stFileUploader"] {{
        background-color: rgba(28, 31, 38, 0.88);
        border: 1px solid rgba(247, 147, 30, 0.35);
        border-radius: 12px;
        padding: 1.2rem;
    }}

    [data-testid="stMetric"] {{
        background-color: rgba(28, 31, 38, 0.88);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(247, 147, 30, 0.35);
    }}

    div[data-baseweb="notification"] {{
        border-radius: 10px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)
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

SAMPLE_IMAGES = {
    "Clean":              "samples/clean.jpg",
    "Dusty":               "samples/dusty.jpg",
    "Bird-drop":           "samples/bird_drop.jpg",
    "Snow-Covered":        "samples/snow_covered.jpg",
    "Electrical-damage":   "samples/electrical_damage.jpg",
    "Physical-Damage":     "samples/physical_damage.jpg",
}

st.markdown("#### Try a Sample Image")
sample_cols = st.columns(len(SAMPLE_IMAGES))
for col, (label, path) in zip(sample_cols, SAMPLE_IMAGES.items()):
    with col:
        if st.button(label, key=f"sample_{label}", use_container_width=True):
            st.session_state.selected_sample = path

st.divider()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif st.session_state.get("selected_sample"):
    image = Image.open(st.session_state.selected_sample).convert("RGB")
else:
    image = None

if image is not None:
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
