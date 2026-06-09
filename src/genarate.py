import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
NUM_ROWS = 300
START_DATE = "2024-01-01"

np.random.seed(42)

# -----------------------------
# CREATE DATE RANGE
# -----------------------------
dates = pd.date_range(start=START_DATE, periods=NUM_ROWS, freq="D")

# -----------------------------
# GENERATE SYNTHETIC DATA
# -----------------------------
df = pd.DataFrame({
    "timestamp": dates,
    "temperature_c": np.random.normal(loc=22, scale=2.5, size=NUM_ROWS).round(2),
    "humidity_pct": np.random.normal(loc=85, scale=4, size=NUM_ROWS).round(2),
    "co2_ppm": np.random.normal(loc=900, scale=80, size=NUM_ROWS).round(1),
    "yield_kg": np.random.normal(loc=13.5, scale=3, size=NUM_ROWS).round(2)
})

# -----------------------------
# INSERT MISSING VALUES RANDOMLY
# -----------------------------
def insert_missing(dataframe, col, missing_rate=0.08):
    idx = dataframe.sample(frac=missing_rate).index
    dataframe.loc[idx, col] = np.nan

for column in ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]:
    insert_missing(df, column, missing_rate=0.08)

# -----------------------------
# SAVE DATASET
# -----------------------------
output_path = "polyhouse_sensor_300_with_missing.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("Saved at:", output_path)
print("\nMissing values per column:")
print(df.isnull().sum())