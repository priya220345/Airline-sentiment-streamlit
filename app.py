import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Skytrax Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Skytrax Airline Reviews – Sentiment Analysis")
st.write("AI-based system to analyze airline customer reviews")

nltk.download("stopwords")
nltk.download("wordnet")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    airline = pd.read_csv("airline.csv")
    airport = pd.read_csv("airport.csv")
    lounge = pd.read_csv("lounge.csv")
    seat = pd.read_csv("seat.csv")

    airline["category"] = "Airline"
    airport["category"] = "Airport"
    lounge["category"] = "Lounge"
    seat["category"] = "Seat"

    df = pd.concat([airline, airport, lounge, seat], ignore_index=True)
    df = df[["content", "recommended", "category"]]
    df.dropna(inplace=True)
    return df

df = load_data()

# -------------------- TEXT CLEANING --------------------
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned_review"] = df["content"].apply(clean_text)

# -------------------- SMART SENTIMENT LABELING --------------------
def assign_sentiment(rec):
    if rec == 1:
        return "Positive"
    elif rec == 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["recommended"].apply(assign_sentiment)

# -------------------- SIDEBAR --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Dataset Insights", "Visual Analysis", "Predict Review Sentiment"]
)

# -------------------- PAGES --------------------
if page == "Project Overview":
    st.header("Project Overview")
    st.markdown("""
    **Objective:**  
    To classify airline customer reviews into  
    **Positive, Negative, or Neutral** sentiment.

    **Key Enhancements:**
    - NLP Text Cleaning
    - TF-IDF Feature Extraction
    - Logistic Regression
    - Confidence-Based Neutral Detection
    - Real-time Prediction

    **Use Case:**  
    Helps airlines understand customer satisfaction and improve services.
    """)
    st.success("Production-ready sentiment analysis system")

elif page == "Dataset Insights":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Positive", (df["sentiment"] == "Positive").sum())
    col3.metric("Negative", (df["sentiment"] == "Negative").sum())

    st.subheader("Sample Reviews")
    st.dataframe(df.sample(10))

elif page == "Visual Analysis":
    st.header("Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="category", hue="sentiment", data=df, ax=ax)
        st.pyplot(fig)

# -------------------- PREDICTION PAGE --------------------
elif page == "Predict Review Sentiment":
    st.header("Predict Review Sentiment")

    X = df["cleaned_review"]
    y = df["sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_vec, y)

    user_review = st.text_area("Enter a customer review")

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(user_review)
            vec = vectorizer.transform([cleaned])

            probs = model.predict_proba(vec)[0]
            confidence = np.max(probs)
            prediction = np.argmax(probs)

            # CONFIDENCE-BASED NEUTRAL FIX
            if confidence < 0.55:
                sentiment = "Neutral 😐"
            else:
                sentiment = ["Negative 😡", "Neutral 😐", "Positive 😊"][prediction]

            st.subheader("Prediction Result")
            st.success(f"Sentiment: **{sentiment}**")
            st.info(f"Confidence Score: **{confidence:.2f}**")

            st.markdown("""
            **Why confidence matters?**  
            Low confidence predictions are treated as **Neutral**  
            to avoid incorrect Positive/Negative classification.
            """)
