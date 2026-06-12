import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # 1. Setup workspace directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_features.parquet")
    test_path = os.path.join(base_dir, "data", "processed", "test_features.parquet")
    
    # Ensure target output directories exist
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports", "figures"), exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Feature columns aligned with the train matrix slice index mapping
    features = ["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]
    X_train = train_df[features].values
    y_train = train_df["yield_kg"].values
    X_test = test_df[features].values
    y_test = test_df["yield_kg"].values
    
    # --- Code block from website template ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    pred = rf.predict(X_test)
    print(f"RF Test MAE: {mean_absolute_error(y_test, pred):.2f} kg")
    print(f"RF Test R2: {r2_score(y_test, pred):.3f}")
    
    importances = rf.feature_importances_
    Labels = ["temperature", "humidity", "co2"]
    
    plt.barh(Labels, importances)
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    
    # Save precisely where the portal expects it
    output_fig_path = os.path.join(base_dir, "reports", "figures", "rf_importance.png")
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    
    # Save the serialized model artifact
    output_model_path = os.path.join(base_dir, "models", "random_forest.joblib")
    joblib.dump(rf, output_model_path)
    # ----------------------------------------
    
    print(f"\nSuccess! Figure saved to {output_fig_path}")
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    main()