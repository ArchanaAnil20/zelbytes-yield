# Monitoring Plan

## Metrics to Monitor
- Temperature
- Humidity
- CO₂ level
- Predicted yield

## Data Drift Scenarios
- Sensor calibration changes
- Seasonal weather changes
- Firmware updates affecting sensor readings

## Retraining Trigger
- Test MAE increases significantly.
- Prediction exceeds historical maximum.
- New harvest data becomes available.

## Future Improvements
1. Weekly model retraining.
2. Add more environmental features.
3. Create dashboard for log analysis.