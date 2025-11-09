import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import io

# --------------------------
# 1. APP HEADER
# --------------------------
st.set_page_config(page_title="MindGuard - Voice Recognition", layout="centered")

st.title("MindGuard - Smart Voice Recognition System")
st.subheader("Identify speakers to assist Alzheimer's patients.")
st.markdown("---")

# --------------------------
# 2. LOAD MODEL
# --------------------------
MODEL_PATH = "speaker_model.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# 3. SPEAKER LABELS
# --------------------------
speaker_classes = {0: "Known", 1: "Family", 2: "Unknown"}

# --------------------------
# 4. MFCC FEATURE EXTRACTION (FIXED SHAPE)
# --------------------------
def extract_mfcc_fixed(file_obj, sr=16000, n_mfcc=40, fixed_frames=200):
    # Load and resample audio
    audio, _ = librosa.load(file_obj, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Force shape to (n_mfcc, fixed_frames)
    if mfcc.shape[1] < fixed_frames:
        pad_width = fixed_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > fixed_frames:
        mfcc = mfcc[:, :fixed_frames]

    # Ensure (1, 40, 200, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc.astype(np.float32)

# --------------------------
# 5. STREAMLIT UI
# --------------------------
st.markdown("### 1. Upload Voice Sample")
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

st.markdown("### 2. Running Prediction")
status_box = st.empty()

st.markdown("### 3. Prediction Result")

if uploaded_file is not None:
    try:
        status_box.info("Processing audio...")

        mfcc_input = extract_mfcc_fixed(uploaded_file)
        interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)

        status_box.success("Prediction completed successfully.")
        st.success(f"Predicted Speaker: {speaker_classes[predicted_class]}")

    except Exception as e:
        status_box.error("Error processing the audio file.")
        st.error(str(e))
else:
    st.info("Please upload a .wav file to begin prediction.")
