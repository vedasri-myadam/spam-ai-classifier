import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model():
    # Load dataset
    df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"])

    # Convert labels to numbers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Convert text to TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer