import os
import requests
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Helmet Detection",
    page_icon="⛑️"
)

# ----------------------------
# Model Configuration
# ----------------------------
MODEL_URL = "https://github.com/malipriti210/DL_CNN_Helmet_Detection/releases/download/v1.0/helmet_final.keras"
MODEL_PATH = "helmet_final.keras"


@st.cache_resource
def load_helmet_model():
    """Download model if not present and load it."""

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... Please wait..."):

            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    model = load_model(MODEL_PATH)
    return model


# Load model
try:
    model = load_helmet_model()
except Exception as e:
    st.error("❌ Failed to load the model.")
    st.exception(e)
    st.stop()

# ----------------------------
# Title
# ----------------------------
st.title("⛑️ Helmet Detection")
st.write("Upload an image or use your camera to detect whether a helmet is present.")

# ----------------------------
# Select Input
# ----------------------------
option = st.radio(
    "Select Input Method",
    ["📁 Upload Image", "📷 Open Camera"]
)

img = None

# Upload
if option == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

# Camera
else:

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")

# ----------------------------
# Prediction
# ----------------------------
if img is not None:

    st.image(img, caption="Input Image", use_container_width=True)

    img_resize = img.resize((128, 128))

    img_array = image.img_to_array(img_resize)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array, verbose=0)

    confidence = float(prediction[0][0])

    if confidence < 0.5:
        st.success("✅ Helmet Detected")
        st.metric("Confidence", f"{(1-confidence)*100:.2f}%")
    else:
        st.error("❌ No Helmet Detected")
        st.metric("Confidence", f"{confidence*100:.2f}%")
