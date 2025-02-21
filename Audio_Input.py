import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import torch
import os
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from audio_recorder_streamlit import audio_recorder
def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

# Employee Credentials (Replace with a database if needed)
USER_CREDENTIALS = {
    "emp001": "password123",
    "emp002": "securepass",
    "admin": "adminpass"
}

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login Page
def login():
    st.title("🔐 Employee Login")

    emp_id = st.text_input("Employee ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if emp_id in USER_CREDENTIALS and USER_CREDENTIALS[emp_id] == password:
            st.session_state.logged_in = True
            st.session_state.emp_id = emp_id  # Store logged-in user
            st.rerun()
        else:
            st.error("Invalid Employee ID or Password")

# Logout Function
def logout():
    st.session_state.logged_in = False
    st.rerun()

# If not logged in, show login page
if not st.session_state.logged_in:
    login()
    st.stop()

# Show logout button
st.sidebar.button("Logout", on_click=logout)

# Emotion Detection App (Only accessible after login)
st.title(f"🎤🎙️ Emotion Detection - Welcome {st.session_state.emp_id}")

# Load Model and Processor
model_path = "./results"
try:
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion Labels
label_map = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
    6: "surprise"
}

# Function to split audio and return first & last segments
def split_audio(audio_path, segment_duration=20):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        segment_samples = segment_duration * sr
        segments = [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]
        
        if len(segments) > 1:
            return segments[0], segments[-1], sr
        return segments[0], segments[0], sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# Function to predict emotion from an audio segment
def predict_emotion(audio_segment, sr, max_length=32000):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            sf.write(temp_filename, audio_segment, sr)

        speech, _ = sf.read(temp_filename)  
        speech = np.pad(speech, (0, max(0, max_length - len(speech))), 'constant')[:max_length]

        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_emotion = label_map.get(predicted_id, "Unknown")

        os.remove(temp_filename)

        return predicted_emotion

    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return "Error"

# Streamlit UI for Emotion Detection
choice = st.selectbox("Choose an option:", ("Upload File", "Record Audio"))

if choice == "Upload File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        save_path = "audio.wav"
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.audio(save_path, format="audio/wav")

            first_segment, last_segment, sr = split_audio(save_path)
            if first_segment is not None:
                st.success("Extracted First & Last Audio Segments")

                e1 = predict_emotion(first_segment, sr)
                e2 = predict_emotion(last_segment, sr)

                #st.write(f"🎭 **Emotion from First Segment:** {e1}")
                #st.write(f"🎭 **Emotion from Last Segment:** {e2}")
        except Exception as e:
            st.error(f"Error handling uploaded file: {e}")

if choice == "Record Audio":
    audio_bytes = audio_recorder("Click to record", neutral_color="#F47174", recording_color="#6FC276")
    if audio_bytes is not None:
        save_path = "recorded_audio.wav"
        try:
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            st.audio(save_path, format="audio/wav")

            first_segment, last_segment, sr = split_audio(save_path)
            if first_segment is not None:
                st.success("Extracted First & Last Audio Segments")

                e1 = predict_emotion(first_segment, sr)
                e2 = predict_emotion(last_segment, sr)

                #st.write(f"🎭 **Emotion from First Segment:** {e1}")
                #st.write(f"🎭 **Emotion from Last Segment:** {e2}")
        except Exception as e:
            st.error(f"Error processing recorded audio: {e}")

if 'e1' in locals() and 'e2' in locals() and e1 and e2:
    good = {"surprise", "neutral", "happy"}
    bad = {"disgust", "fear", "angry", "sad"}

    if e1 in good and e2 in bad:
         display_message("assistant", "🔴 **The Customer is Not Satisfied**")
    elif e1 in bad and e2 in good:
          display_message("assistant", "🟢**The Customer is Satisfied**")
    elif e1 in good and e2 in good:
         display_message("assistant", "🟢**The Customer is Satisfied**")
    elif e1 in bad and e2 in bad:
         display_message("assistant", "🔴 **The Customer is Not Satisfied**")
        
