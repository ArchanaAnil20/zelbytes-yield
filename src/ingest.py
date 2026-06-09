import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "polyhouse_sensor_300_with_missing.csv")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_PATH = os.path.join(INTERIM_DIR, "01_loaded.parquet")

def load_and_validate():
    print("--- STEP 1: Ingesting Raw Data ---")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw CSV file at: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])

    print("\nShape:", df.shape)

    print("\nMissing values per column:\n")
    print(df.isnull().sum())

    print("\nSummary statistics:\n")
    print(df.describe())

    os.makedirs(INTERIM_DIR, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved cleaned snapshot → {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_validate()