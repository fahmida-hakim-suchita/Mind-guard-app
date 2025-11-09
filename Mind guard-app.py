import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os

# Set page configuration
st.set_page_config(page_title="MindGuard Voice Recognition", layout="centered")

# App header
st.markdown("<h1 style='text-align: center;'>MindGuard - Smart Voice Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identify speakers to assist Alzheimer's patients.</p>", unsafe_allow_html=True)
st.markdown("---")

# Path to TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "speaker_model.tflite")

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Dataset class mapping
speaker_classes = {0: "family", 1: "known", 2: "unknown"}

# File uploader
uploaded_file = st.file_uploader("Upload a voice sample (.wav format):", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Processing audio...")

    try:
        # Extract MFCC features
        audio, sr = librosa.load(uploaded_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(prediction)
        predicted_speaker = speaker_classes.get(predicted_index, "unknown")
        confidence = np.max(prediction)

        # Display result in professional card style
        st.markdown(
            f"""
            <div style='border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; text-align: center; background-color: #f9f9f9;'>
                <h3>Prediction Result</h3>
                <p><b>Predicted Speaker:</b> {predicted_speaker}</p>
                <p><b>Confidence:</b> {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error processing the audio file: {e}")

st.markdown("---")
st.caption("MindGuard - Alzheimer's Smart Assistant | Developed by Suchita")
