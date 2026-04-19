from flask import Flask, render_template, request
import pickle
from utils import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]

    confidence = max(prob) * 100

    if prediction == 1:
        result = "News is REAL"
    else:
        result = "News is FAKE"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)