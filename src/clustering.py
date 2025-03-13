import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model

# Load preprocessed data
df = pd.read_csv("data/preprocessed_media_bias.csv")

# Select relevant numerical features for clustering
features = ["Engagement", "Citation_Rate", "Readership_Size"]
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save the model and scaler
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save clustered data
df.to_csv("data/clustered_media_bias.csv", index=False)

print("✅ Clustering completed. Results saved in 'data/clustered_media_bias.csv'.")
