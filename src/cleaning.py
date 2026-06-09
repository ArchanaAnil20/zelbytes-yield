import os
import pandas as pd

# -----------------------------
# Paths (auto project root)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "polyhouse_sensor_300_with_missing.csv")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_PATH = os.path.join(INTERIM_DIR, "02_cleaned.parquet")

# -----------------------------
# Cleaning function
# -----------------------------
def clean_pipeline():
    print("--- STEP 2: Running Cleaning Pipeline ---")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw file not found at: {RAW_PATH}")

    # Load data
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])

    print("\nInitial missing values:\n")
    print(df.isnull().sum())

    # -----------------------------
    # REQUIRED COLUMNS (your dataset)
    # -----------------------------
    cols = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]

    # -----------------------------
    # 1. Convert to numeric safely
    # -----------------------------
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # 2. Sort by time (important for interpolation)
    # -----------------------------
    df = df.sort_values("timestamp")

    # -----------------------------
    # 3. Interpolate missing values
    # -----------------------------
    df[cols] = df[cols].interpolate(method="linear")

    # -----------------------------
    # 4. Fill any remaining NaNs (edge cases)
    # -----------------------------
    df[cols] = df[cols].bfill().ffill()

    # -----------------------------
    # 5. Basic validation
    # -----------------------------
    print("\nMissing values AFTER cleaning:\n")
    print(df.isnull().sum())

    print("\nFinal dataset shape:", df.shape)

    # -----------------------------
    # Save cleaned file
    # -----------------------------
    os.makedirs(INTERIM_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved cleaned dataset → {OUTPUT_PATH}")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    clean_pipeline()