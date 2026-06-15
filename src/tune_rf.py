import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def main():
    print("=== Day 13: Running Hyperparameter Tuning via GridSearchCV ===")
    
    # 1. Resolve repository directory structure paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_features.parquet")
    test_path = os.path.join(base_dir, "data", "processed", "test_features.parquet")
    
    # Ensure targets exist
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
    
    # Load dataset slices
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    features = ["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]
    X_train = train_df[features].values
    y_train = train_df["yield_kg"].values
    X_test = test_df[features].values
    y_test = test_df["yield_kg"].values
    
    # Track runtime optimization overhead
    start_time = time.time()
    
    # --- Exact Code Logic Block From Portal Template ---
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3, 5],
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    search = GridSearchCV(
        rf,
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)
    
    best_model = search.best_estimator_
    
    with open(os.path.join(base_dir, "models", "rf_best_params.json"), "w") as f:
        json.dump(search.best_params_, f, indent=2)
    # ----------------------------------------------------
    
    # Measure execution duration elapsed
    elapsed_time = time.time() - start_time
    print(f"Grid Search Runtime: {elapsed_time:.2f} seconds")
    
    # Save the refitted production candidate model to disk
    model_output_path = os.path.join(base_dir, "models", "random_forest_tuned.joblib")
    joblib.dump(best_model, model_output_path)
    
    # Evaluate optimal model once on the hold-out test set
    test_preds = best_model.predict(X_test)
    tuned_test_mae = mean_absolute_error(y_test, test_preds)
    print(f"Tuned Model Hold-Out Test MAE: {tuned_test_mae:.4f} kg")
    
    # Save a detailed log for the checklist parameters
    tuning_log_path = os.path.join(base_dir, "reports", "tuning_rationale.md")
    rationale_content = (
        f"# Day 13 Tuning Rationale and Performance Review\n\n"
        f"## 1. Parameter Knob Selection Strategies\n"
        f"- `n_estimators`: Tested [50, 100, 200] to scale ensemble voting numbers. More trees reduce variance without changing bias.\n"
        f"- `max_depth`: Tested [None, 8, 16] to directly bound decision branch limits. Constraining depth limits structural overfitting.\n"
        f"- `min_samples_leaf`: Tested [1, 3, 5] to force trees to require minimum support samples per leaf, smoothing out local variance spikes.\n\n"
        f"## 2. Experimental Benchmark Results\n"
        f"- **Optimal Found Hyperparameters:** {json.dumps(search.best_params_)}\n"
        f"- **Best Calculated Validation CV MAE:** {(-search.best_score_):.4f} kg\n"
        f"- **Isolated Hold-Out Test MAE:** {tuned_test_mae:.4f} kg\n"
        f"- **Grid Process Wall Runtime Time:** {elapsed_time:.2f} seconds\n"
    )
    with open(tuning_log_path, "w") as f:
        f.write(rationale_content)
    print(f"Tuning documentation written to: {tuning_log_path}")

if __name__ == "__main__":
    main()