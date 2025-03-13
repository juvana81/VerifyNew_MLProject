import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict if a news article is Fake or Real
def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]
    return "🟢 Real News" if prediction == 0 else "🔴 Fake  News"

# Test Cases
news_articles = [
    "Breaking: The government has announced a new economic policy to boost the economy.",
    "SHOCKING! Scientists discover a secret alien base on the Moon!",
    "The stock market is expected to rise due to recent policy changes."
]

# Predict for each article
for news in news_articles:
    print(f"News: {news}\nPrediction: {predict_news(news)}\n")
