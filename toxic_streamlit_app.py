
import streamlit as st
import joblib

# تحميل النموذج والـ TF-IDF
model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🧠 تصنيف التعليقات المسيئة (Toxic Comment Classifier)")

user_input = st.text_area("✍️ اكتب تعليقك هنا:")

if st.button("تحليل التعليق"):
    if user_input.strip() == "":
        st.warning("من فضلك اكتب تعليقًا أولًا.")
    else:
        X_input = tfidf.transform([user_input])
        prediction = model.predict(X_input)[0]
        label = "Toxic ❌" if prediction == 1 else "Not Toxic ✅"
        st.success(f"النتيجة: {label}")
