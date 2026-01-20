from sklearn.svm import LinearSVC

# -------------------- TRAIN MODEL --------------------
X = df["cleaned_review"]
y = df["sentiment_label"]

vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

svm_model = LinearSVC()
svm_model.fit(X_vec, y)

# -------------------- USER INPUT --------------------
user_review = st.text_area(
    "Enter a customer review",
    placeholder="Example: The flight was okay, service was average, nothing special."
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
