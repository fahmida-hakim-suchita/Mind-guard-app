import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf

# --- Load trained model ---
model = tf.keras.models.load_model("speaker_model.h5")

# --- Feature Extraction ---
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# --- Prediction Function ---
def predict_speaker(file_path):
    feature = extract_features(file_path)
    feature = np.expand_dims(feature, axis=0)
    prediction = model.predict(feature)
    speaker_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return speaker_idx, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Mind Guard", page_icon="🧠", layout="centered")

st.title("🧠 Mind Guard – Smart Speaker Identification System")
st.write("Upload a voice file (.wav) to identify the speaker.")

uploaded_file = st.file_uploader("🎵 Upload a voice file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save temp file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Analyzing voice... 🔍")
    speaker_idx, conf = predict_speaker("temp_audio.wav")

    st.success(f"✅ Speaker Identified: Class {speaker_idx}")
    st.info(f"Confidence: {conf*100:.2f}%")

    st.balloons()
