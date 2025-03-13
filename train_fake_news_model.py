import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Define dataset path
dataset_path = r"C:\Users\VICTUS\MEDIA BIAS\data\cleaned_fake_news_dataset.csv.csv"

# Check if the dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset file not found at {dataset_path}")
    exit()

# Load Fake News Dataset
df = pd.read_csv(dataset_path)

# Ensure required columns exist
required_columns = {"text", "label"}
if not required_columns.issubset(df.columns):
    print("❌ Error: Dataset must contain 'text' and 'label' columns.")
    exit()

# Drop missing values
df.dropna(subset=["text", "label"], inplace=True)

# Prepare Data
X = df["text"].astype(str)  # Ensure text column is string
y = df["label"]

# Convert Text to Vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Fake News Model Trained & Saved Successfully!")
