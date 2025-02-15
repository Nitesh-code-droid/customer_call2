import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import pandas as pd
import pydub

def start_model_file():
  s=st.chat_input("the work is under process")
  return s
def start_model_live():
  s=st.chat_input("the work is under process")
  return s

st.title("🎤🎙️Emotion Detection")
choice= st.selectbox(
    "Choose one of the following ",
    ("upload_file","Record Audio"),
)

if choice=="upload_file" :
  uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "wav", "mp3", "m4a","MPEG"])
  if uploaded_file is not None:
    save_directory = os.getcwd()
    os.makedirs(save_directory, exist_ok=True)
    filename = uploaded_file.name
    save_path = os.path.join(save_directory, "audio.wav")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.audio("audio.wav", format="audio/wav")    
    st.success(f"File saved successfully")
    st.button('Run', on_click=start_model_file)
    #st.button('Run', on_click=None)




if choice=="Record Audio":
  audio_bytes = audio_recorder(text="Click to record: ", neutral_color="#F47174", recording_color="#6FC276")
  if audio_bytes is not None:
    voice = st.audio(audio_bytes, format="audio/wav")
    save_directory = os.getcwd()
    os.makedirs(save_directory, exist_ok=True)
    filename = "audio.wav"
    save_path = os.path.join(save_directory, filename)
    with open(save_path, "wb") as f:
        f.write(audio_bytes)
    st.success(f"File saved successfully at {save_path}")
    st.button('Run', on_click=start_model_live)
    #st.button('Run', on_click=None)


