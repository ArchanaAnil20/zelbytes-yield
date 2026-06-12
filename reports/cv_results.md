# Day 12 Validation Analysis Log

## Cross-Validation Scores Summary
- **Random Forest Mean CV MAE:** 2.5608 kg (Variance Std: 1.1845)
- **Linear Regression Mean CV MAE:** 2.6081 kg (Variance Std: 0.9567)

## Overfitting & Variance Analysis
- **Train vs Test Discrepancy:** Full Training MAE sits at 0.9122 kg, while hold-out Test MAE is 4.5484 kg.
- **Interpretation:** The significant difference between Training error and Cross-Validation validation error confirms that the default Random Forest model is overlearning local training trends. The non-zero standard deviation across folds indicates performance variance, verifying that greenhouse microclimate signals display cyclical shifts over our data timeline.