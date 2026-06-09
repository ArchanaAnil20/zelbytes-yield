import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "interim", "02_cleaned.parquet")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    return df


# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
def correlation_heatmap(df):
    plt.figure(figsize=(8, 6))

    corr = df[["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Correlation Heatmap")

    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()

    print(f"Saved → {path}")


# -----------------------------
# 2. Scatter Plots
# -----------------------------
def scatter_plots(df):

    features = ["temperature_c", "humidity_pct", "co2_ppm"]

    for col in features:
        plt.figure(figsize=(6, 4))

        sns.scatterplot(data=df, x=col, y="yield_kg")

        plt.title(f"Yield vs {col}")

        path = os.path.join(FIG_DIR, f"{col}_vs_yield.png")
        plt.savefig(path)
        plt.close()

        print(f"Saved → {path}")


# -----------------------------
# MAIN
# -----------------------------
def run_eda():
    print("--- STEP 3: EDA (Minimal Version) ---")

    df = load_data()

    correlation_heatmap(df)
    scatter_plots(df)

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    run_eda()