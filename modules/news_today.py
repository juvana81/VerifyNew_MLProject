from flask import Blueprint, render_template, request, jsonify
import requests
from transformers import pipeline
import spacy
from textblob import TextBlob
from yake import KeywordExtractor

news_bp = Blueprint("news_today", __name__, template_folder="templates")

API_KEY = "fdfe153ca83b4dc38ff5bd095f272e9b"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

# Keyword Extractor
kw_extractor = KeywordExtractor(lan="en", n=3, top=5)

def analyze_sentiment(text):
    """Returns sentiment classification (Positive, Negative, Neutral)."""
    if not text:  # ✅ Prevent NoneType errors
        return "Neutral"
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    return "Neutral"

def detect_bias(text):
    """Detects political bias based on keyword occurrence."""
    if not text:  # ✅ Prevent NoneType errors
        return "Neutral"
    
    left_words = ["climate change", "universal healthcare", "social justice", "wealth tax"]
    right_words = ["gun rights", "border security", "tax cuts", "military spending"]
    
    left_count = sum(text.lower().count(word) for word in left_words)
    right_count = sum(text.lower().count(word) for word in right_words)
    
    if left_count > right_count:
        return "Left-Leaning"
    elif right_count > left_count:
        return "Right-Leaning"
    return "Neutral"

def extract_keywords(text):
    """Extracts important keywords using YAKE."""
    if not text:  # ✅ Prevent NoneType errors
        return []
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def generate_summary(text, keywords):
    """Summarizes news content with additional keyword context."""
    if not text.strip():  # ✅ Prevent NoneType errors
        return "Summary not available."
    
    keyword_text = " ".join(keywords)
    input_text = f"{keyword_text} {text}"
    
    try:
        summary = summarizer(input_text, max_length=90, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing: {str(e)}"

@news_bp.route("/")
def news_today():
    """Fetches top news and processes sentiment, bias, and keywords."""
    params = {
        "apiKey": API_KEY,
        "country": request.args.get("country", "us"),
        "category": request.args.get("category", "general"),
        "q": request.args.get("q", "")
    }
    response = requests.get(NEWS_API_URL, params=params)
    news_data = response.json().get("articles", [])

    processed_news = []
    for article in news_data:
        content = article.get("content", "") or article.get("description", "") or "No content available"
        keywords = extract_keywords(content)
        sentiment = analyze_sentiment(content)
        bias = detect_bias(content)
        processed_news.append({
            "title": article.get("title", "No title"),
            "content": content,
            "url": article.get("url", "#"),
            "sentiment": sentiment,
            "bias": bias,
            "keywords": keywords
        })
    
    return render_template("news_today.html", news=processed_news)

@news_bp.route("/summarize", methods=["POST"])
def summarize_news():
    """Handles news summarization."""
    article_text = request.form.get("news_text", "").strip()
    
    if not article_text:
        return jsonify({"error": "No text provided"}), 400
    
    keywords = extract_keywords(article_text)
    summary = generate_summary(article_text, keywords)
    
    return jsonify({"summary": summary})
