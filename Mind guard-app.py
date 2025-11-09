# -----------------------------
# MindGuard-app.py cell: Streamlit app

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
import io

# -----------------------------
st.set_page_config(page_title="MindGuard Voice Recognition", layout="centered")

st.markdown("<h1 style='text-align: center;'>MindGuard - Smart Voice Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identify speakers to assist Alzheimer's patients.</p>", unsafe_allow_html=True)
st.markdown("---")

# Load TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "speaker_model.tflite")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Class mapping
speaker_classes = {0: "family", 1: "known", 2: "unknown"}

# MFCC extraction for uploaded audio
def extract_mfcc_strict(file_obj, sr=16000, n_mfcc=40, duration=2.0, n_fft=2048):
    if isinstance(file_obj, io.BytesIO) or hasattr(file_obj, 'read'):
        audio, _ = librosa.load(file_obj, sr=sr)
    else:
        audio, _ = librosa.load(file_obj, sr=sr)

    max_samples = int(sr * duration)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
    else:
        audio = audio[:max_samples]

    hop_length = (len(audio) - n_fft) // 199  # 200 frames total
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc.astype(np.float32)

# -----------------------------
# PART 1: Upload audio
st.header("1️⃣ Upload Voice Sample")
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# PART 2: Running Prediction placeholder
st.header("2️⃣ Running Prediction")
prediction_placeholder = st.empty()
prediction_placeholder.info("Upload a file to start prediction...")

# PART 3: Prediction Result placeholder
st.header("3️⃣ Prediction Result")
result_placeholder = st.empty()

# -----------------------------
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    prediction_placeholder.info("Processing audio...")

    try:
        mfcc = extract_mfcc_strict(uploaded_file)
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(prediction)
        predicted_speaker = speaker_classes.get(predicted_index, "unknown")
        confidence = np.max(prediction)

        prediction_placeholder.success("Audio processed successfully!")

        result_placeholder.markdown(
            f"""
            <div style='border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; text-align: center; background-color: #f9f9f9;'>
                <h3>Predicted Speaker</h3>
                <p><b>Name:</b> {predicted_speaker}</p>
                <p><b>Confidence:</b> {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        prediction_placeholder.error("Error processing the audio file.")
        result_placeholder.error(f"{e}")

st.markdown("---")
st.caption("MindGuard - Alzheimer's Smart Assistant | Developed by Suchita")
