import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import time

import tensorflow as tf
import matplotlib.pyplot as plt

# ─── Config ─────────────────────────────────────
IMG_SIZE = 224
MODEL_PATHS = [
    Path("models/best_model.keras"),
    Path("models/cnn_model.keras")
]

# ─── Page Config ────────────────────────────────
st.set_page_config(
    page_title="PawSense AI",
    page_icon="🐾",
    layout="wide"
)

# ─── CSS ────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-name {
    font-size: 1rem;
    text-align: center;
    letter-spacing: 0.25rem;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 2rem;
}
.result-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.dog-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
.cat-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}
.uncertain-card {
    background: linear-gradient(135deg, #777 0%, #333 100%);
}
.result-label {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}
.confidence-text {
    font-size: 1.1rem;
    color: white;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# ─── Load Model ─────────────────────────────────
@st.cache_resource
def load_trained_model():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path), path
    return None, None

model, model_path = load_trained_model()

# ─── Header ─────────────────────────────────────
st.markdown('<p class="brand-name">PawSense AI</p>', unsafe_allow_html=True)
st.markdown('<p class="main-title">🐾 Dog vs Cat Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">EfficientNetB0 Transfer Learning · Upload or snap a photo to classify</p>',
    unsafe_allow_html=True
)

# ─── Stop if model missing ──────────────────────
if model is None:
    st.error("No trained model found. Please run `python train.py` first.")
    st.info("Expected model path: `models/best_model.keras` or `models/cnn_model.keras`")
    st.stop()

# ─── Prediction Function ────────────────────────
def predict_image(model, image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr, verbose=0)[0][0]
    cat_prob = float(prob)
    dog_prob = 1.0 - cat_prob
    if cat_prob >= 0.5:
        label, confidence = "Cat", cat_prob
    else:
        label, confidence = "Dog", dog_prob
    return label, confidence

# ─── Upload Section ─────────────────────────────
st.markdown("### 📤 Upload an Image")

col_up, col_cam = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

with col_cam:
    if st.button("📷 Take a Photo", width="stretch"):
        st.session_state.show_camera = not st.session_state.show_camera

camera_img = None
if st.session_state.show_camera:
    camera_img = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_img:
        st.session_state.show_camera = False

file_obj = uploaded_file or camera_img

# ─── Inference ──────────────────────────────────
if file_obj:
    image = Image.open(file_obj).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", width="stretch")

    with col2:
        with st.spinner("Analyzing image..."):
            start = time.time()
            label, confidence = predict_image(model, image)
            elapsed = time.time() - start

        final_label = label

        card_class = {"Dog": "dog-card", "Cat": "cat-card"}.get(final_label, "uncertain-card")
        emoji      = {"Dog": "🐶",       "Cat": "🐱"      }.get(final_label, "🤔")

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">{emoji} {final_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"⚡ Inference time: {elapsed * 1000:.1f} ms")

        if final_label != "Uncertain":
            st.session_state.history.append({
                "label": final_label,
                "confidence": confidence
            })

else:
    st.markdown("""
    <div style="border:2px dashed #555; border-radius:16px; padding:3rem; text-align:center; color:#888;">
        <div style="font-size:3rem;">🐾</div>
        <div>Upload an image or take a photo to classify Dog or Cat</div>
    </div>
    """, unsafe_allow_html=True)

# ─── History Chart ──────────────────────────────
if len(st.session_state.history) >= 2:
    st.markdown("---")
    st.markdown("### 📈 Session Prediction History")

    confidences = [item["confidence"] for item in st.session_state.history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(confidences) + 1), confidences, marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence History")

    st.pyplot(fig)