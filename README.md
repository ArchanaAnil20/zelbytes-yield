\# zelbytes-yield-



\## Environment Setup



To replicate this workspace and run the project, follow these setup steps:



1\. \*\*Activate the Virtual Environment:\*\*

&#x20;  From the project directory, run:

&#x20;  ```cmd

&#x20;  ..\\venv\\Scripts\\activate
## Phase 1 & 2: Project Progress Summary

### Task 2: Data Cleaning & Preprocessing
- **Missing Values:** Handled rows with missing attributes by applying targeted fill strategies (forward-fill/mean imputation) to ensure clean sensor logs.
- **Outliers:** Detected and handled anomalies in environmental logs (temperature, humidity, CO2) using the IQR (Interquartile Range) method to prevent extreme spikes from skewing model weights.
- **Artifact Saved:** Cleaned base dataset stored under `data/processed/02_cleaned.parquet`.

### Task 3: Exploratory Data Analysis & Feature Selection
- **Correlation Analysis:** Generated a Pearson correlation matrix targeting `yield_kg`.
- **Key Findings:** Identified strong linear relationships between environmental inputs and yield, providing the statistical foundation for feature engineering.

### Task 4: Feature Engineering & Data Splitting (Day 7 & 8)
- **Engineered Feature:** `temp_humid_interaction`
  - *Formula:* (temperature_c * humidity_pct) / 100
  - *Biological Justification:* Combined temperature and humidity approximate the Vapor Pressure Deficit (VPD). This directly regulates stomatal conductance and transpiration rates, serving as a vital proxy for crop health and ultimate yield.
- **Chronological Split:** Applied a leak-free 80% Train / 20% Test chronological split to preserve time-series sequence logic.
  - *Train Window:* 2026-06-01 08:00:00 $\rightarrow$ 2026-06-04 15:00:00
  - *Test Window:* 2026-06-04 16:00:00 $\rightarrow$ 2026-06-05 11:00:00
- **Data Leakage Mitigation:** Scaler statistics (`MinMaxScaler`) calculated strictly on training partitions; test feature columns transformed using train statistics only.
-
# Inference

Run the app:

```bash
streamlit run app.py
```

Example Input

- Temperature = 22°C
- Humidity = 88%
- CO₂ = 900 ppm

Example Output

Estimated daily yield: 11.50 kg