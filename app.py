from flask import Flask, render_template, request
from model import train_model

app = Flask(__name__)

# Train model when server starts
model, vectorizer = train_model()

model_name = type(model).__name__


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]

    # Spam keyword highlighting
    spam_keywords = ["free", "win", "urgent", "offer", "money", "prize", "lottery", "cash"]

    highlighted_text = email_text
    for word in spam_keywords:
        highlighted_text = highlighted_text.replace(
            word,
            f"<span style='color:red; font-weight:bold'>{word}</span>"
        )

    # Convert text using vectorizer
    text_vector = vectorizer.transform([email_text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()

    result = "Spam 🚨" if prediction == 1 else "Not Spam ✅"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=round(probability * 100, 2),
        original_text=highlighted_text,
        model_used=model_name
    )


if __name__ == "__main__":
    app.run()