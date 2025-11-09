import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="MindGuard - Smart Voice Recognition", layout="centered")

st.markdown("""
    <style>
        body {background-color: #f7f9fb;}
        .main-title {text-align:center; font-size:32px; color:#2E86C1; font-weight:bold;}
        .subtitle {text-align:center; font-size:18px; color:#5D6D7E; margin-bottom:20px;}
        .section {background:#fff; padding:25px; border-radius:20px; box-shadow:0 2px 10px rgba(0,0,0,0.1); margin-bottom:25px;}
        .success-box {background:#E8F8F5; border-left:5px solid #1ABC9C; padding:10px 20px; border-radius:10px;}
        .error-box {background:#FDEDEC; border-left:5px solid #E74C3C; padding:10px 20px; border-radius:10px;}
    </style>
""", unsafe_allow_html=True)

# --------------------------
# App Header
# --------------------------
st.markdown('<div class="main-title">🧠 MindGuard - Smart Voice Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify speakers to assist Alzheimer\'s patients.</div>', unsafe_allow_html=True)

# --------------------------
# Load Model
# --------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="speaker_model.tflite")
    interpreter.allocate_tensors()
    st.markdown('<div class="success-box">✅ Model loaded successfully.</div>', unsafe_allow_html=True)
except Exception as e:
    st.markdown(f'<div class="error-box">❌ Error loading model: {e}</div>', unsafe_allow_html=True)

# --------------------------
# Speaker Classes
# --------------------------
speaker_classes = {0: "Family", 1: "Friends", 2: "Unknown"}

# --------------------------
# Feature Extraction
# --------------------------
def extract_mfcc(file_path, n_mfcc=40, target_frames=200, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or crop to match the model input
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0,0),(0,target_frames - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]

    mfcc = np.expand_dims(mfcc, axis=-1)  # (40, 200, 1)
    mfcc = np.expand_dims(mfcc, axis=0)   # (1, 40, 200, 1)
    return mfcc.astype(np.float32)

# --------------------------
# UI Part 1 - Upload Section
# --------------------------
st.markdown('<div class="section"><h4>1️⃣ Upload Voice Sample</h4>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# UI Part 2 - Prediction
# --------------------------
st.markdown('<div class="section"><h4>2️⃣ Run Prediction</h4>', unsafe_allow_html=True)

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        input_data = extract_mfcc("temp.wav")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure input shape compatibility
        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interprete
