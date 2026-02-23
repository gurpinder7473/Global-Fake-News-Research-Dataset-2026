
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import tempfile
import os

st.set_page_config(page_title="Global Fake News Detector", layout="wide")

# Load model
model = joblib.load("fake_news_xgboost_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]
    return prediction, probability

# UI
st.title("🌍 Global Fake News Detection System")
st.markdown("### Choose Input Mode")

mode = st.radio("Select Mode:", ["Text", "Image", "Video", "Audio"], horizontal=True)

if mode == "Text":
    text_input = st.text_area("Enter News Content")
    if st.button("Analyze"):
        if text_input.strip():
            pred, prob = predict_news(text_input)
            if pred == 1:
                st.error(f"Fake News | Confidence: {prob:.2f}")
            else:
                st.success(f"Real News | Confidence: {prob:.2f}")
        else:
            st.warning("Please enter text.")

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload Image (Text will be extracted if OCR added)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.info("Note: Add OCR for extracting text from images if required.")

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        st.info("Note: Add speech-to-text for analyzing video content.")

elif mode == "Audio":
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        st.info("Note: Add speech-to-text for analyzing audio content.")
