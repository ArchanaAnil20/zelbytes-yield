import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Load data from the processed folder and sort by timestamp
df = pd.read_parquet("data/processed/02_cleaned.parquet")

# Standardize names to prevent the KeyError: 'temperature_c'
df.columns = df.columns.str.lower()
rename_map = {
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "co2": "co2_ppm",
    "yield": "yield_kg"
}
# Apply name fixes if columns are missing their standard suffixes
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

# 2. Feature engineering: create the interaction term
df["temp_humid_interaction"] = df["temperature_c"] * df["humidity_pct"] / 100

# 3. Define feature columns and target matrix
feature_cols = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]
X = df[feature_cols]
y = df["yield_kg"]

# 4. Initialize and fit MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. Save the fitted scaler object to the models folder
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/minmax_scaler.joblib")

# 6. Reconstruct processed DataFrame with scaled columns
processed = pd.DataFrame(X_scaled, columns=[c + "_scaled" for c in feature_cols])
processed["yield_kg"] = y.values

# 7. Save the final dataset to the destination parquet file
os.makedirs("data/processed", exist_ok=True)
processed.to_parquet("data/processed/features.parquet", index=False)

print("Task successfully completed! Features and scaler have been saved.")