# Day 13 Tuning Rationale and Performance Review

## 1. Parameter Knob Selection Strategies
- `n_estimators`: Tested [50, 100, 200] to scale ensemble voting numbers. More trees reduce variance without changing bias.
- `max_depth`: Tested [None, 8, 16] to directly bound decision branch limits. Constraining depth limits structural overfitting.
- `min_samples_leaf`: Tested [1, 3, 5] to force trees to require minimum support samples per leaf, smoothing out local variance spikes.

## 2. Experimental Benchmark Results
- **Optimal Found Hyperparameters:** {"max_depth": null, "min_samples_leaf": 1, "n_estimators": 50}
- **Best Calculated Validation CV MAE:** 2.9632 kg
- **Isolated Hold-Out Test MAE:** 4.5016 kg
- **Grid Process Wall Runtime Time:** 8.01 seconds
