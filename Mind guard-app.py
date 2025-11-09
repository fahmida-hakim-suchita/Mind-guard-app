import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os

# App title
st.title("MindGuard - Smart Voice Recognition System")
st.write("This app identifies the speaker using your trained model.")

# Define the path to the TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "speaker_model.tflite")

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Map class indices to dataset labels
speaker_classes = {0: "family", 1: "known", 2: "unknown"}

# Upload audio file
uploaded_file = st.file_uploader("Upload a voice sample (.wav format):", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("Processing the audio...")

    # Extract MFCC features
    try:
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

        # Display result
        st.write("Prediction Result:")
        st.write(f"Predicted Speaker: {predicted_speaker}")

    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
