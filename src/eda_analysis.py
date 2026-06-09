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

    # ==========================================
    # Combined Scatter Plot (like your image)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Humidity vs Yield
    sns.scatterplot(
        data=df,
        x='humidity',
        y='yield',
        ax=axes[0],
        color='skyblue',
        alpha=0.7
    )
    axes[0].set_xlabel("Humidity (%)")
    axes[0].set_ylabel("Yield (kg)")
    axes[0].set_title("")

    # Temperature vs Yield
    sns.scatterplot(
        data=df,
        x='temperature',
        y='yield',
        ax=axes[1],
        color='skyblue',
        alpha=0.7
    )
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Yield (kg)")
    axes[1].set_title("")

    # CO2 vs Yield
    sns.scatterplot(
        data=df,
        x='CO2',
        y='yield',
        ax=axes[2],
        color='skyblue',
        alpha=0.7
    )
    axes[2].set_xlabel("CO₂ (ppm)")
    axes[2].set_ylabel("Yield (kg)")
    axes[2].set_title("")

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_DIR, "scatter_yield.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Generated: scatter_yield.png")

    # ==========================================
    # Correlation Heatmap
    # ==========================================
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Polyhouse Features Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "correlation_heatmap.png"),
        dpi=300
    )
    plt.close()

    print(f"\nAll evaluation plots successfully saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda_reports()