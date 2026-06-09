import os
import pandas as pd

# Automatically calculate the root folder path (ZELBYTES-YIELD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "polyhouse_sensor.csv")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_PATH = os.path.join(INTERIM_DIR, "01_loaded.parquet")

def load_and_validate():
    print("--- STEP 1: Ingesting Raw Data ---")
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw CSV file at: {RAW_PATH}")
        
    # Read CSV and force the timestamp column to parse as a real date
    df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
    
    print("\n[Data Structure Validation]")
    print(f"Shape: {df.shape} (100 rows expected)")
    print("\nMissing values detected per column:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics (df.describe):")
    print(df.describe())

    # Create interim folder if it doesn't exist yet
    os.makedirs(INTERIM_DIR, exist_ok=True)
    
    # Save snapshot
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSuccessfully stored interim snapshot to: {OUTPUT_PATH}")

if __name__ == "__main__":
    load_and_validate()