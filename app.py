import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Skytrax Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Skytrax Airline Reviews – Sentiment Analysis")
st.write("Analyze airline customer reviews using NLP and Machine Learning")

nltk.download("stopwords")
nltk.download("wordnet")

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
    df.dropna(subset=["content", "recommended"], inplace=True)
    return df

df = load_data()

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
df["sentiment"] = df["recommended"].map({1: "Positive", 0: "Negative"})

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Dataset Insights", "Visual Analysis", "Predict Review Sentiment"]
)

if page == "Project Overview":
    st.header("Project Overview")
    st.markdown("""
    **Objective:**  
    To analyze airline customer reviews and classify them into  
    **Positive** or **Negative** sentiment using Machine Learning.

    **Technologies Used:**
    - Python
    - NLP
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """)
    st.success("This is a machine learning–based sentiment analysis system.")

elif page == "Dataset Insights":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", df.shape[0])
    col2.metric("Positive Reviews", (df["sentiment"] == "Positive").sum())
    col3.metric("Negative Reviews", (df["sentiment"] == "Negative").sum())

    st.subheader("Sample Reviews")
    st.dataframe(df.sample(10))

elif page == "Visual Analysis":
    st.header("Sentiment Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="category", hue="sentiment", data=df, ax=ax)
        st.pyplot(fig)

elif page == "Predict Review Sentiment":
    st.header("Predict Review Sentiment")

    X = df["cleaned_review"]
    y = df["sentiment"].map({"Negative": 0, "Positive": 1})

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    user_review = st.text_area("Write your review here")

    if st.button("Predict"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            cleaned_input = clean_text(user_review)
            vec_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vec_input)[0]

            if prediction == 1:
                st.success("Predicted Sentiment: **Positive 😊**")
            else:
                st.error("Predicted Sentiment: **Negative 😡**")

