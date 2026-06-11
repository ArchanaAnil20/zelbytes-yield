
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assume X_train, X_test, y_train, y_test from Day 8
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_parquet(os.path.join(base_dir, "data", "processed", "train_features.parquet"))
test_df = pd.read_parquet(os.path.join(base_dir, "data", "processed", "test_features.parquet"))

# Match your exact features
features = ["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]
X_train = train_df[features]
y_train = train_df["yield_kg"]
X_test = test_df[features]
y_test = test_df["yield_kg"]

model = LinearRegression()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

print(f"Test MAE:  {mae:.2f} kg")
print(f"Test RMSE: {rmse:.2f} kg")
print(f"Test R²:   {r2:.3f}")

for name, coef in zip(["temp", "humidity", "co2"], model.coef_):
    print(f" print coef {name}: {coef:.3f}")

# Save model artifact to required path
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
joblib.dump(model, os.path.join(base_dir, "models", "linear_regression.joblib"))

# Save metrics to reports/metrics_linear.json for the checklist requirement
os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
metrics_data = {"mae": mae, "rmse": rmse, "r2": r2}
with open(os.path.join(base_dir, "reports", "metrics_linear.json"), "w") as f:
    json.dump(metrics_data, f, indent=4)
