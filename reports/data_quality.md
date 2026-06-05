# Data Quality Report
**Project Name:** Zelbytes Yield Forecast
**Date Generated:** June 5, 2026

## 1. Dataset Scope & Summary
* **Total Observations Checked:** 10 sensor records parsed from `02_cleaned.parquet`.
* **Missing Data / Null Values:** 0 dropouts detected.

## 2. Descriptive Statistics Summary Table

| Metric | Temperature (°C) | Humidity (%) | CO₂ (ppm) | Yield (kg) |
| :--- | :---: | :---: | :---: | :---: |
| **count** | 10.00000 | 10.00000 | 10.000000 | 10.00000 |
| **mean** | 26.47500 | 76.00000 | 732.222222 | 13.64000 |
| **std** | 1.06432 | 3.16227 | 51.591941 | 0.52535 |
| **min** | 24.50000 | 72.00000 | 650.000000 | 12.80000 |
| **25%** | 26.11875 | 74.00000 | 695.000000 | 13.25000 |
| **50%** | 26.48750 | 75.50000 | 736.111111 | 13.70000 |
| **75%** | 27.15000 | 77.50000 | 775.000000 | 14.07500 |
| **max** | 28.00000 | 82.00000 | 800.000000 | 14.30000 |

## 3. Boundary & Validation Rules Verification
* **Humidity Bound Check:** ✅ PASS. Min value 72.0% and Max value 82.0% fall safely within natural baseline parameters.
* **CO₂ Bound Check:** ✅ PASS. Stable microclimate gas levels between 650 ppm and 800 ppm without sensor dropouts.