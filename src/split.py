import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the cleaned data from your processed folder path and sort chronologically
df = pd.read_parquet("data/processed/02_cleaned.parquet").sort_values("timestamp")

# Standardize columns to lowercase and map names to match what the platform expects
df.columns = df.columns.str.lower()
rename_map = {
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "co2": "co2_ppm",
    "yield": "yield_kg"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Define the reference feature columns
feature_cols = ["temperature_c", "humidity_pct", "co2_ppm"]

# Perform chronological 80/20 split (No random shuffling for time-series logs)
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# CRITICAL FIX FOR LEAKAGE: Fit & transform on train features, only transform test features
X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])

# Isolate target metrics arrays
y_train = train["yield_kg"].values
y_test = test["yield_kg"].values

# Save the train-isolated scaler object 
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/minmax_scaler_train.joblib")

# Print training and test window validations to terminal
print(f"Train: {train['timestamp'].min()} -> {train['timestamp'].max()}")
print(f"Test:  {test['timestamp'].min()} -> {test['timestamp'].max()}")

# Persist files safely back under data/processed/ for upcoming regression tasks
os.makedirs("data/processed", exist_ok=True)

train_split = pd.DataFrame(X_train, columns=[c + "_scaled" for c in feature_cols])
train_split["yield_kg"] = y_train
train_split.to_parquet("data/processed/train_features.parquet", index=False)

test_split = pd.DataFrame(X_test, columns=[c + "_scaled" for c in feature_cols])
test_split["yield_kg"] = y_test
test_split.to_parquet("data/processed/test_features.parquet", index=False)

print("\nTask successfully completed! Leak-free split datasets and scaler have been saved.")