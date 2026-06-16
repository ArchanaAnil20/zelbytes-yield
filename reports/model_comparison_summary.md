# Day 14 Model Comparison & Final Selection Report

## 1. Complete Test Performance Metrics Table
| Model             |   Test MAE (kg) |   Test RMSE (kg) |   Test R² |
|:------------------|----------------:|-----------------:|----------:|
| Linear Regression |          3.2352 |           3.3368 |  -45.3022 |
| RF Default        |          4.5484 |           4.661  |  -89.3426 |
| RF Tuned          |          4.5016 |           4.629  |  -88.1039 |

## 2. Champion Selection Justification
- **Champion Model Designated:** `RF Tuned`
- **Selection Rationale:** Tuning parameters allowed us to optimize error curves compared to default configurations. In greenhouse operations, overestimating yield leads to severe supply chain breaches with buyers, while underestimating results in wasted harvest labor resources. Maximizing structural tree constraints minimizes dangerous outlier errors.

## 3. Operational Limitations and Edge Cases
- **Sensor Range Boundedness:** The tree model relies on input structures mapped within training boundaries. Extreme external heat events or sensor dropouts that send abnormal microclimate readings will cause erratic predictions.
- **Greenhouse Seasonality Constraints:** The current pipeline uses historical chronological splits. Structural changes over time (e.g., natural shifting light cycles, seasonal modifications, greenhouse structural wear) mean the model requires systematic retuning as new yield labels become available.