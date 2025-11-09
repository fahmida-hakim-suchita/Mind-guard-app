import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os

# -----------------------------
# Streamlit page config
st.set_page_config(page_title="MindGuard Voice Recognition", layout="centered")

# App header
st.markdown("<h1 style='text-align: center;'>MindGuard - Smart Voice Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identify speakers to assist Alzheimer's patients.</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
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

# -----------------------------
# Dataset class mapping
speaker_classes = {0: "family", 1: "known", 2: "unknown"}

# -----------------------------
# PART 1: Upload audio
st.header("1️⃣ Upload Voice Sample")
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # -----------------------------
    # PART 2: Processing / Prediction
    st.header("2️⃣ Running Prediction")
    st.info("Processing audio...")

    try:
        # Extract MFCC features
        audio, sr = librosa.load(uploaded_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Make 4D input for TFLite model
        mfcc = np.expand_dims(mfcc, axis=-1)  # (n_mfcc, time_frames, 1)
        mfcc = np.expand_dims(mfcc, axis=0)   # (1, n_mfcc, time_frames, 1)
        mfcc = mfcc.astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(prediction)
        predicted_speaker = speaker_classes.get(predicted_index, "unknown")
        confidence = np.max(prediction)

        # -----------------------------
        # PART 3: Display Prediction Result
        st.header("3️⃣ Prediction Result")
        st.markdown(
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
        st.error(f"Error processing the audio file: {e}")

# -----------------------------
st.markdown("---")
st.caption("MindGuard - Alzheimer's Smart Assistant | Developed by Suchita")
