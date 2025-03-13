from flask import Flask, request, jsonify
import joblib  # To load ML model
import re

app = Flask(__name__)

# Load Fake News Detection Model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    """Preprocess input text (remove special characters, convert to lowercase)"""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if the news is Fake or Real"""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    result = "Fake News" if prediction[0] == 1 else "Real News"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
