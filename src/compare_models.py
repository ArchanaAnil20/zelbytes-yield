import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=== Day 14: Final Model Comparison & Metrics Evaluation ===")
    
    # 1. Coordinate database directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_features.parquet")
    test_path = os.path.join(base_dir, "data", "processed", "test_features.parquet")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    features = ["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]
    X_train = train_df[features].values
    y_train = train_df["yield_kg"].values
    X_test = test_df[features].values
    y_test = test_df["yield_kg"].values
    
    # 2. Load all three models
    lr_model = joblib.load(os.path.join(base_dir, "models", "linear_regression.joblib"))
    rf_default = joblib.load(os.path.join(base_dir, "models", "random_forest.joblib"))
    rf_tuned = joblib.load(os.path.join(base_dir, "models", "random_forest_tuned.joblib"))
    
    models_dict = {
        "Linear Regression": lr_model,
        "RF Default": rf_default,
        "RF Tuned": rf_tuned
    }
    
    summary_data = []
    
    # 3. Compute test performance for every model pipeline
    print("\nComputing test set performance metrics...")
    for name, model in models_dict.items():
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        summary_data.append({
            "Model": name,
            "Test MAE (kg)": round(mae, 4),
            "Test RMSE (kg)": round(rmse, 4),
            "Test R²": round(r2, 4)
        })
        
    results_df = pd.DataFrame(summary_data)
    print("\n=== FINAL TEST METRICS SUMMARY TABLE ===")
    print(results_df.to_markdown(index=False))
    
    # 4. Generate Predicted vs Actual Plot for Champion (RF Tuned)
    print("\nGenerating Champion diagnostics scatter chart...")
    champion_preds = rf_tuned.predict(X_test)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, champion_preds, alpha=0.6, color="#1f77b4", edgecolors="k", label="Test Data Points")
    
    # Add identity reference line
    mn, mx = min(y_test.min(), champion_preds.min()), max(y_test.max(), champion_preds.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Alignment Line")
    
    plt.xlabel("Actual Yield (kg)")
    plt.ylabel("Predicted Yield (kg)")
    plt.title("Champion Model (RF Tuned): Predicted vs Actual Yield")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "pred_vs_actual.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Diagnostic plot exported to: {fig_path}")
    
    # 5. Document comprehensive agritech deployment report
    report_path = os.path.join(base_dir, "reports", "model_comparison_summary.md")
    report_markdown = (
        f"# Day 14 Model Comparison & Final Selection Report\n\n"
        f"## 1. Complete Test Performance Metrics Table\n"
        f"{results_df.to_markdown(index=False)}\n\n"
        f"## 2. Champion Selection Justification\n"
        f"- **Champion Model Designated:** `RF Tuned`\n"
        f"- **Selection Rationale:** Tuning parameters allowed us to optimize error curves compared to default configurations. In greenhouse operations, overestimating yield leads to severe supply chain breaches with buyers, while underestimating results in wasted harvest labor resources. Maximizing structural tree constraints minimizes dangerous outlier errors.\n\n"
        f"## 3. Operational Limitations and Edge Cases\n"
        f"- **Sensor Range Boundedness:** The tree model relies on input structures mapped within training boundaries. Extreme external heat events or sensor dropouts that send abnormal microclimate readings will cause erratic predictions.\n"
        f"- **Greenhouse Seasonality Constraints:** The current pipeline uses historical chronological splits. Structural changes over time (e.g., natural shifting light cycles, seasonal modifications, greenhouse structural wear) mean the model requires systematic retuning as new yield labels become available."
    )
    
    with open(report_path, "w") as f:
        f.write(report_markdown)
    print(f"[SUCCESS] Full comparative report written to: {report_path}")

if __name__ == "__main__":
    main()