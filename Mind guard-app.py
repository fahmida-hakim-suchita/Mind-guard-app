import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="MindGuard - Smart Voice Recognition", layout="wide")

st.markdown("""
    <style>
        body {background-color: #f7f9fb; font-family: 'Poppins', sans-serif;}
        .main-title {text-align:center; font-size:32px; color:#2E86C1; font-weight:bold; margin-bottom:5px;}
        .subtitle {text-align:center; font-size:18px; color:#5D6D7E; margin-bottom:20px;}
        .section {background:#fff; padding:25px; border-radius:20px; 
                  box-shadow:0 2px 10px rgba(0,0,0,0.1); margin-bottom:25px;}
        .result-box {background:#EDF6F9; padding:20px; border-radius:15px; 
                     text-align:center; font-weight:600; color:#1e6091; font-size:22px;}
        .footer {text-align:center; color:#888; font-size:12px; margin-top:20px;}
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown('<div class="main-title">MindGuard - Smart Voice Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify speakers to assist Alzheimer\'s patients.</div>', unsafe_allow_html=True)

# --------------------------
# Load TFLite Model
# --------------------------
MODEL_PATH = "speaker_model.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Class Labels
# --------------------------
speaker_classes = {0: "Family", 1: "Friends", 2: "Unknown"}

# --------------------------
# MFCC Extraction
# --------------------------
def extract_mfcc(file_path, n_mfcc=40, target_frames=200, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0,0),(0,target_frames - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc.astype(np.float32)

# --------------------------
# Two Column Layout
# --------------------------
left_col, right_col = st.columns([1, 1])

# --------------------------
# LEFT: Upload + Status
# --------------------------
with left_col:
    st.markdown('<div class="section"><h4>📤 Upload Voice Sample</h4>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    status_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# RIGHT: Prediction Output
# --------------------------
with right_col:
    st.markdown('<div class="section"><h4>🔍 Prediction Output</h4>', unsafe_allow_html=True)
    result_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# AUTO PREDICTION (Runs Immediately After Upload)
# --------------------------
if uploaded_file is not None:
    try:
        status_box.info("Processing audio...")

        # Save uploaded file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Extract MFCC
        mfcc_input = extract_mfcc("temp.wav")

        # Resize input tensor
        interpreter.resize_tensor_input(input_details[0]['index'], mfcc_input.shape)
        interpreter.allocate_tensors()

        # Prediction
        interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)

        status_box.success("Prediction completed!")
        result_box.markdown(
            f'<div class="result-box">Predicted Speaker:<br>{speaker_classes[predicted_class]}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        status_box.error(f"❌ Error processing file: {e}")
        result_box.error("Prediction failed.")

else:
    result_box.info("Please upload a voice sample to see prediction.")

# --------------------------
# Footer
# --------------------------
st.markdown('<div class="footer">© 2025 MindGuard | Designed by Suchita</div>', unsafe_allow_html=True)
