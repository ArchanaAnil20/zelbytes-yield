import pandas as pd

# Load CSV file
df = pd.read_csv("data/raw/polyhouse_sensor.csv")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values with column averages
df = df.fillna(df.mean(numeric_only=True))

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned data
df.to_csv("data/processed/cleaned_data.csv", index=False)
df.head(50).to_csv("data/processed/sample_50_rows.csv",index=False)
df.to_parquet("data/processed/02_cleaned.parquet",index=False)
print("\nCleaning completed successfully!")