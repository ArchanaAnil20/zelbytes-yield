import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Setup exact structural paths based on dashboard requirements
DATA_PATH = Path("data/processed/02_cleaned.parquet")
FIGURES_DIR = Path("reports/figures")

# Create directories automatically if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

print("🎨 Loading data and preparing plots...")
df = pd.read_parquet(DATA_PATH)

# Standardize columns to lowercase to prevent KeyErrors
df.columns = [col.lower() for col in df.columns]

# --- Save Plots Directly into Root Folder ---

# 1. Temperature vs Yield Scatter Plot
plt.figure(figsize=(5, 4))
plt.scatter(df['temperature'], df['yield'], color='crimson')
plt.xlabel("Temperature (°C)")
plt.ylabel("Yield (kg)")
plt.title("Temperature vs Yield")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "co2_vs_yield.png") # Match required names
plt.close()

# 2. Humidity vs Yield Scatter Plot
plt.figure(figsize=(5, 4))
plt.scatter(df['humidity'], df['yield'], color='teal')
plt.xlabel("Humidity (%)")
plt.ylabel("Yield (kg)")
plt.title("Humidity vs Yield")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "humidity_vs_yield.png")
plt.close()

# 3. Correlation Matrix Heatmap Plot (Basic implementation)
plt.figure(figsize=(5, 4))
corr = df[['temperature', 'humidity', 'co2', 'yield']].corr()
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
plt.close()

print("✅ EDA completed successfully! Images saved to root reports/figures/")