import os
import librosa
import numpy as np
import streamlit as st
import soundfile as sf
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🎧 Voice Tagging with Auto-Annotation")

UPLOAD_DIR = "uploaded_audios"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# File upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded: {uploaded_file.name}")

# List of available files
audio_files = sorted([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".wav")])
selected_file = st.selectbox("Choose a file to tag", audio_files)

if selected_file:
    file_path = os.path.join(UPLOAD_DIR, selected_file)
    st.audio(file_path, format="audio/wav")

    # Load and process audio
    y, sr = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(f"Duration: {duration:.2f} seconds")

    # Extract features for tagging
    pitch = librosa.yin(y, fmin=80, fmax=400)
    energy = np.mean(librosa.feature.rms(y=y))

    # Heuristic Gender Detection (based on pitch)
    pitch_median = np.median(pitch)
    gender = "Female" if pitch_median > 165 else "Male"

    # Heuristic Emotion Detection (based on energy)
    emotion = "Neutral"
    if energy > 0.1:
        emotion = "Angry/Loud"
    elif energy < 0.03:
        emotion = "Calm/Sad"

    # Tags (editable)
    st.subheader("📝 Tags")
    gender_tag = st.selectbox("Gender", ["Auto", "Male", "Female"], index=0)
    emotion_tag = st.selectbox("Emotion", ["Auto", "Neutral", "Angry/Loud", "Calm/Sad"], index=0)

    final_gender = gender if gender_tag == "Auto" else gender_tag
    final_emotion = emotion if emotion_tag == "Auto" else emotion_tag

    st.markdown(f"""
    - **Auto Gender**: `{gender}`
    - **Auto Emotion**: `{emotion}`
    - **Final Gender**: `{final_gender}`
    - **Final Emotion**: `{final_emotion}`
    """)

    # Save tags to CSV
    import pandas as pd
    TAG_CSV = os.path.join(UPLOAD_DIR, "tags.csv")
    tags = pd.read_csv(TAG_CSV) if os.path.exists(TAG_CSV) else pd.DataFrame(columns=["File", "Gender", "Emotion"])
    tags = tags[tags["File"] != selected_file]
    tags.loc[len(tags)] = [selected_file, final_gender, final_emotion]
    tags.to_csv(TAG_CSV, index=False)

    st.success("Tags saved successfully.")

    # Show full tag list
    with st.expander("📋 Show all tagged files"):
        st.dataframe(tags)

else:
    st.info("Please upload or select a WAV file to begin tagging.")
