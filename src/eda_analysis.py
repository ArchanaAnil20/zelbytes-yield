import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "02_cleaned.parquet")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

def generate_eda_reports():
    print("--- STEP 3: Generating Visualizations ---")
    df = pd.read_parquet(INPUT_PATH)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. NEW: Temperature vs Yield Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='temperature', y='yield', color='crimson', alpha=0.7)
    plt.title("Mushroom Yield vs Temperature (°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "temp_vs_yield.png"))
    plt.close()
    print("Generated: temp_vs_yield.png")
    
    # 2. CO2 vs Yield Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='CO2', y='yield', color='green', alpha=0.7)
    plt.title("Mushroom Yield vs CO2 Concentrations")
    plt.xlabel("CO2 (ppm)")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "co2_vs_yield.png"))
    plt.close()
    
    # 3. Humidity vs Yield Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='humidity', y='yield', color='blue', alpha=0.7)
    plt.title("Mushroom Yield vs Relative Humidity")
    plt.xlabel("Humidity (%)")
    plt.ylabel("Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "humidity_vs_yield.png"))
    plt.close()

    # 4. Correlation Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Polyhouse Features Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close()
    
    print(f"\nAll evaluation plots successfully saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda_reports()