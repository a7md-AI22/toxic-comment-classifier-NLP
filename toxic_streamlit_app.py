import streamlit as st
import re, emoji, joblib, nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z:_\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("🔍 Toxic Comment Classifier")
st.write("أدخل تعليقًا وسيتم تصنيفه إذا كان مسيئًا أم لا.")

model, vectorizer = load_artifacts()

user_input = st.text_area("💬 أدخل تعليقك هنا:")

if st.button("تحليل"):
    if not user_input.strip():
        st.warning("من فضلك أدخل نصًا أولًا.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("⚠️ هذا التعليق مسيء.")
        else:
            st.success("✅ هذا التعليق غير مسيء.")
