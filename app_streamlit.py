
import streamlit as st
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    st.title("AG News Topic Classifier")
    st.write("Enter a news headline or short article and get the predicted topic.")

    user_text = st.text_area("News text", height=150)

    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = user_text.lower()
            X = tfidf_vectorizer.transform([cleaned])
            proba = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            st.subheader(f"Predicted topic: {pred}")
            st.write("Class probabilities:")
            for cls, p in zip(model.classes_, proba):
                st.write(f"{cls}: {p:.3f}")

if __name__ == '__main__':
    main()
