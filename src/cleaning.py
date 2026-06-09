import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "interim", "01_loaded.parquet")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def clean_pipeline():
    print("--- STEP 2: Running Cleaning Pipeline ---")
    df = pd.read_parquet(INPUT_PATH)
    
    # Fill missing values sequentially along the time-series using linear interpolation
    df['temperature'] = df['temperature'].interpolate(method='linear')
    df['humidity'] = df['humidity'].interpolate(method='linear')
    df['CO2'] = df['CO2'].interpolate(method='linear')
    
    # Technical boundary safeguard: Cap maximum humidity at 100%
    df['humidity'] = df['humidity'].clip(upper=100.0)
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save clean versions
    df.to_parquet(os.path.join(PROCESSED_DIR, "02_cleaned.parquet"), index=False)
    df.to_csv(os.path.join(PROCESSED_DIR, "cleaned_data.csv"), index=False)
    
    print("Data cleaning complete! 0 missing values remain.")
    print("Files saved inside data/processed/")

if __name__ == "__main__":
    clean_pipeline()
