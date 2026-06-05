import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/cleaned_data.csv")

print(df.describe())

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df["humidity"], df["yield"])
plt.xlabel("Humidity (%)")
plt.ylabel("Yield (kg)")
plt.title("Humidity vs Yield")
plt.savefig("reports/figures/humidity_vs_yield.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df["CO2"], df["yield"])
plt.xlabel("CO2")
plt.ylabel("Yield (kg)")
plt.title("CO2 vs Yield")
plt.savefig("reports/figures/co2_vs_yield.png")
plt.close()

print("EDA completed successfully")