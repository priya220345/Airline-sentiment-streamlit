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
from sklearn.svm import LinearSVC

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skytrax Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Skytrax Airline Reviews – Sentiment Analysis")
st.write("Customer Review Analysis using NLP and Support Vector Machine (SVM)")

# ---------------- NLTK ----------------
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------- LOAD DATA ----------------
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
    df.dropna(subset=["content"], inplace=True)

    return df

df = load_data()

# ---------------- TEXT PREPROCESSING ----------------
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

# ---------------- SENTIMENT LABELING ----------------
neutral_words = [
    "average", "okay", "fine", "acceptable",
    "not bad", "nothing special", "satisfactory"
]

def assign_sentiment(text, recommended):
    text = text.lower()
    if any(word in text for word in neutral_words):
        return "Neutral"
    elif recommended == 1:
        return "Positive"
    elif recommended == 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df.apply(
    lambda x: assign_sentiment(x["content"], x["recommended"]),
    axis=1
)

sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["sentiment_label"] = df["sentiment"].map(sentiment_map)

# ---------------- TRAIN MODEL (CACHED) ----------------
@st.cache_resource
def train_svm_model(df):
    X = df["cleaned_review"]
    y = df["sentiment_label"]

    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2)
    )
    X_vec = vectorizer.fit_transform(X)

    model = LinearSVC(dual=False)
    model.fit(X_vec, y)

    return model, vectorizer

svm_model, vectorizer = train_svm_model(df)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Dataset Insights",
        "Visual Analysis",
        "Predict Review Sentiment"
    ]
)

# ---------------- PROJECT OVERVIEW ----------------
if page == "Project Overview":
    st.header("Project Overview")

    st.markdown("""
**Objective:**  
To analyze airline customer reviews and classify them into  
**Positive**, **Neutral**, or **Negative** sentiments.

**Technologies Used:**
- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Support Vector Machine (LinearSVC)
- Streamlit

**Why SVM?**
- Best performance on high-dimensional text data
- Achieved highest accuracy during experiments
- Widely used in real-world sentiment analysis
    """)

    st.success("This is a complete Machine Learning–based Customer Review Analysis System.")

# ---------------- DATASET INSIGHTS ----------------
elif page == "Dataset Insights":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", df.shape[0])
    col2.metric("Positive Reviews", (df["sentiment"] == "Positive").sum())
    col3.metric("Negative Reviews", (df["sentiment"] == "Negative").sum())

    st.subheader("Sample Customer Reviews")
    st.dataframe(df.sample(10))

# ---------------- VISUAL ANALYSIS ----------------
elif page == "Visual Analysis":
    st.header("Visual Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, ax=ax)
        ax.set_title("Overall Sentiment Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="category", hue="sentiment", data=df, ax=ax)
        ax.set_title("Category-wise Sentiment Distribution")
        st.pyplot(fig)

# ---------------- PREDICTION ----------------
elif page == "Predict Review Sentiment":
    st.header("Predict Review Sentiment")

    user_review = st.text_area(
        "Enter a customer review",
        placeholder="Example: The flight was okay, seats were comfortable but service was average."
    )

    if st.button("Predict"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            cleaned_input = clean_text(user_review)
            vec_input = vectorizer.transform([cleaned_input])
            prediction = svm_model.predict(vec_input)[0]

            if prediction == 2:
                st.success("Predicted Sentiment: **Positive 😊**")
            elif prediction == 1:
                st.info("Predicted Sentiment: **Neutral 😐**")
            else:
                st.error("Predicted Sentiment: **Negative 😡**")
