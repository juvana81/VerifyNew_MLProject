import pandas as pd

# Load dataset
df = pd.read_csv("data/media_bias.csv")

# Convert all non-numeric values to NaN and drop non-numeric columns
df_numeric = df.select_dtypes(include=["number"])  # Only keep numeric columns

# Fill missing values with column means
df_numeric.fillna(df_numeric.mean(numeric_only=True), inplace=True)

# Merge numeric data back with original dataset
df_cleaned = pd.concat([df_numeric, df.select_dtypes(exclude=["number"])], axis=1)

# Save cleaned data
df_cleaned.to_csv("data/preprocessed_media_bias.csv", index=False)

print("✅ Preprocessing complete! Check preprocessed_media_bias.csv")
