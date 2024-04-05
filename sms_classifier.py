import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load the vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Function to preprocess text input
def preprocess_text(text):
    text = text.lower()
    text = " ".join([word for word in text.split() if word.isalnum()])
    return text

# Function to classify text
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vectorized)[0]
    return prediction

# Streamlit App
def main():
    st.title("SMS Spam Classifier")
    st.markdown("This app classifies text messages as spam or ham (non-spam).")

    # User input for text message
    user_input = st.text_input("Enter the text message:")

    # Classify text on button click
    if st.button("Classify"):
        if user_input:
            prediction = classify_text(user_input)
            if prediction == 1:
                st.error("Spam")
            else:
                st.success("Ham (Non-spam)")
        else:
            st.warning("Please enter a text message.")

if __name__ == "__main__":
    main()
