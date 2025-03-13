import pandas as pd
import plotly.express as px
import requests
import joblib
import logging
import spacy
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from yake import KeywordExtractor
from textblob import TextBlob

# Import News Today Blueprint (Ensure it's correctly defined in modules/news_today.py)
try:
    from modules.news_today import news_bp
except ModuleNotFoundError:
    logging.warning("⚠️ Warning: news_today module not found! Skipping blueprint registration.")

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Register Blueprints
try:
    app.register_blueprint(news_bp, url_prefix="/news-today")
except NameError:
    logging.warning("⚠️ Blueprint not registered. Ensure news_today module exists.")

# ✅ Load Fake News Model & Vectorizer
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    logging.error("⚠️ Fake News Model or Vectorizer not found!")
    model, vectorizer = None, None

# ✅ Initialize External API
FAKE_NEWS_API_URL = "http://127.0.0.1:5000/predict"

# ✅ Load NLP Models
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
bias_classifier = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# ✅ Keyword Extractor (YAKE)
kw_extractor = KeywordExtractor(lan="en", n=3, top=5)

# ✅ Load Bias Data (Ensure the file exists)
try:
    df = pd.read_csv("data/clustered_media_bias.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["PCA1", "PCA2", "Cluster", "news_source", "rating", "type"])
    logging.warning("⚠️ Warning: Dataset file not found!")

# ✅ Create Bias Map Visualization
fig = px.scatter(
    df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str),
    hover_data=["news_source", "rating", "type"], title="Media Bias Clustering"
)
fig.update_layout(
    xaxis_title="PCA1: Political Bias Spectrum (Left to Right)",
    yaxis_title="PCA2: Media Type or Credibility",
    template="plotly_dark"
)
plot_html = fig.to_html(full_html=False)

# ✅ Fake News Checking Function
def check_fact(article_text):
    """Send article text to the Fake News API and return prediction."""
    try:
        response = requests.post(FAKE_NEWS_API_URL, json={"text": article_text})
        return response.json() if response.status_code == 200 else {"error": "API response failed"}
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ API Request Failed: {e}")
        return {"error": "API request error"}

# ✅ Utility Functions
def extract_headline(text):
    """Extract headline (first sentence)."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences[0] if sentences else "No headline found"

def extract_keywords(text):
    """Extract important keywords using YAKE."""
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def generate_summary(text, keywords):
    """Summarize article using keywords."""
    keyword_text = " ".join(keywords)
    input_text = f"{keyword_text} {text}"
    summary = summarizer(input_text, max_length=90, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    sentiment_score = TextBlob(text).sentiment.polarity
    return "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

def detect_bias(text):
    """Detect bias based on keywords."""
    bias_terms = {
        "Left-Leaning": ["climate change", "universal healthcare", "social justice", "wealth tax"],
        "Right-Leaning": ["gun rights", "border security", "tax cuts", "military spending"]
    }
    bias_counts = {bias: sum(text.lower().count(word) for word in words) for bias, words in bias_terms.items()}
    max_bias = max(bias_counts, key=bias_counts.get)
    return max_bias if bias_counts[max_bias] > 0 else "Neutral"

# ✅ Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/bias-map")
def bias_map():
    return render_template("bias_map.html", plot=plot_html)

@app.route("/news-summary", methods=["GET", "POST"])
def news_summary():
    if request.method == "POST":
        article_text = request.form.get("news_text", "").strip()
        if not article_text:
            return jsonify({"error": "No text provided"}), 400

        headline = extract_headline(article_text)
        keywords = extract_keywords(article_text)
        summary = generate_summary(article_text, keywords)
        sentiment = analyze_sentiment(article_text)
        bias = detect_bias(article_text)

        return jsonify({
            "headline": headline, "keywords": keywords,
            "summary": summary, "sentiment": sentiment, "bias": bias
        })
    return render_template("news_summary.html")

@app.route("/fact-check", methods=["POST"])
def fact_check():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Invalid JSON format. Expected {'url': '<article_url>'}"}), 400
    result = check_fact(data["url"])
    return jsonify(result)

@app.route("/fake-news", methods=["GET", "POST"])
def fake_news():
    prediction = None
    if request.method == "POST":
        if not model or not vectorizer:
            return jsonify({"error": "Model files not found!"}), 500

        news_text = request.form.get("news", "").strip()
        if not news_text:
            return jsonify({"error": "No news text provided!"}), 400

        transformed_text = vectorizer.transform([news_text])
        pred = model.predict(transformed_text)[0]
        prediction = "🟢Real News" if pred == 0 else "🔴Fake News"

    return render_template("fake_news.html", prediction=prediction)

# ✅ Run App
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Changed port to avoid conflicts
