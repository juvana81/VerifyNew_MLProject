import pandas as pd

# Load with correct encoding & handle bad lines
dataset_path = r"C:\Users\VICTUS\MEDIA BIAS\data\fake_news_dataset.csv.csv"


df = pd.read_csv(dataset_path, encoding="utf-8", on_bad_lines="skip")



# Check for missing values
print(df.isnull().sum())

# Drop NaN rows if necessary
df.dropna(subset=["text", "label"], inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_fake_news_dataset.csv", index=False)
