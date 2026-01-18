import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Skytrax Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

nltk.download('stopwords')
nltk.download('wordnet')

st.title("✈️ Skytrax Airline Reviews – Sentiment Analysis")
st.write("Analyze airline customer reviews using NLP and Machine Learning")

@st.cache_data
def load_data():
    airline = pd.read_csv("airline.csv")
    airport = pd.read_csv("airport.csv")
    lounge = pd.read_csv("lounge.csv")
    seat = pd.read_csv("seat.csv")

    airline['category'] = 'Airline'
    airport['category'] = 'Airport'
    lounge['category'] = 'Lounge'
    seat['category'] = 'Seat'

    df = pd.concat([airline, airport, lounge, seat], ignore_index=True)
    df = df[['content', 'recommended', 'category']]
    df.dropna(subset=['content'], inplace=True)
    return df

df = load_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_review'] = df['content'].apply(clean_text)
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['cleaned_review'].apply(get_sentiment)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Dataset Insights", "Visual Analysis", "Predict Review Sentiment"]
)

if page == "Project Overview":
    st.header("Project Overview")

    st.markdown("""
    **Objective:**  
    This project analyzes airline customer reviews and classifies them into
    **Positive**, **Negative**, or **Neutral** sentiments.

    **Why Sentiment Analysis?**
    - Understand customer satisfaction
    - Identify service issues
    - Improve airline experience

    **Technologies Used:**
    - Python
    - Natural Language Processing
    - Machine Learning
    - Streamlit
    """)

    st.success("This is an interactive web-based sentiment analysis system.")
elif page == "Dataset Insights":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", df.shape[0])
    col2.metric("Positive Reviews", (df['sentiment'] == 'Positive').sum())
    col3.metric("Negative Reviews", (df['sentiment'] == 'Negative').sum())

    st.subheader("Sample Reviews")
    st.dataframe(df.sample(10))
elif page == "Visual Analysis":
    st.header("Sentiment Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='sentiment', data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Category-wise Sentiment")
        fig, ax = plt.subplots()
        sns.countplot(x='category', hue='sentiment', data=df, ax=ax)
        st.pyplot(fig)
elif page == "Predict Review Sentiment":
    st.header("Predict Review Sentiment")

    st.write("Enter a customer review to predict its sentiment")

    X = df['cleaned_review']
    y = df['sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    user_review = st.text_area("Write your review here")

    if st.button("Predict"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(user_review)
            vec_input = vectorizer.transform([cleaned])
            prediction = model.predict(vec_input)[0]

            sentiment_map = {
                0: "Negative 😡",
                1: "Neutral 😐",
                2: "Positive 😊"
            }

            st.success(f"Predicted Sentiment: **{sentiment_map[prediction]}**")
