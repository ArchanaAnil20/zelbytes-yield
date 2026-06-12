# Day 10 Baseline Model Diagnostics Report

## 1. Residual Calculation Verification
The residuals were calculated using the standard definition: `Residual = Actual - Predicted`.

## 2. Key Diagnostic Findings
- **Non-Random Curvature Pattern:** The plot of residuals versus scaled humidity shows clear quadratic or structural curvature rather than a uniform cloud of random points around the zero mark. This indicates a severe linear underfit.
- **Heteroscedasticity:** Error spread varies significantly across the prediction space, confirming that fixed linear constraints struggle with dynamic time-split data changes.

## 3. Modeling Recommendation
**Recommendation: Try Nonlinear Model.** Because the baseline linear regression model collapsed with an R² of -45.302 and shows distinct curved structural errors, the biological relationship between polyhouse climate features and crop yield is strictly non-linear. Moving forward, we must implement a non-linear model family (such as Random Forest or Gradient Boosting) on Day 11 to handle these patterns.