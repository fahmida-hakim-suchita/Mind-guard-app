import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import base64

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(page_title="MindGuard - Smart Voice Recognition", layout="wide")

# -----------------------------------------------------
# Background Image Function
# -----------------------------------------------------
def add_bg(image_path):
    encoded = base64.b64encode(open(image_path, 'rb').read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        body {{
            background-color: rgba(255,255,255,0.70);
        }}
        .section {{
            background: rgba(255, 255, 255, 0.80);
            backdrop-filter: blur(4px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ADD BACKGROUND (your image)
add_bg("alzheimer.jpg")


# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
    <style>
        body {font-family: 'Poppins', sans-serif;}
        .main-title {text-align:center; font-size:32px; color:#ffffff; 
                     font-weight:bold; margin-bottom:5px; 
                     text-shadow: 2px 2px 4px black;}
        .subtitle {text-align:center; font-size:18px; color:#f0f0f0; 
                   margin-bottom:20px; text-shadow: 1px 1px 3px black;}
        .section {
            padding:25px; 
            border-radius:20px;
            box-shadow:0 2px 10px rgba(0,0,0,0.3); 
            margin-bottom:25px;
        }
        .result-box {
            background:#EDF6F9; padding:20px; border-radius:15px;
            text-align:center; font-weight:600; color:#1e6091; font-size:22px;
        }
        .footer {text-align:center; color:white; font-size:12px; margin-top:20px;}
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown('<div class="main-title">🧠 MindGuard - Smart Voice Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify speakers to assist Alzheimer\'s patients.</div>', unsafe_allow_html=True)


# -----------------------------------------------------
# Load TFLite Model
# -----------------------------------------------------
MODEL_PATH = "speaker_model.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

speaker_classes = {0: "Family", 1: "Friends", 2: "Unknown"}


# -----------------------------------------------------
# MFCC Extraction
# -----------------------------------------------------
def extract_mfcc(file_path, n_mfcc=40, target_frames=200, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0,0),(0,target_frames-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc.astype(np.float32)


# -----------------------------------------------------
# Two Column Layout
# -----------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section"><h4> 1️⃣ Upload Voice Sample</h4>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h4> 2️⃣ Running Prediction</h4>', unsafe_allow_html=True)
    running_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section"><h4> 3️⃣ Prediction Result</h4>', unsafe_allow_html=True)
    result_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------
# Auto Prediction Logic
# -----------------------------------------------------
if uploaded_file is not None:
    try:
        running_box.info("Processing audio...")

        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        mfcc_input = extract_mfcc("temp.wav")

        interpreter.resize_tensor_input(input_details[0]['index'], mfcc_input.shape)
        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)

        running_box.success("Prediction completed!")

        result_box.markdown(
            f'<div class="result-box">Predicted Speaker:<br>{speaker_classes[predicted_class]}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        running_box.error(f"Error: {e}")
else:
    result_box.info("Please upload a voice file.")


# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown('<div class="footer">© 2025 MindGuard | Designed by Suchita</div>', unsafe_allow_html=True)
