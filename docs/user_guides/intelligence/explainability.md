# Model Explainability & SHAP Feature Importance

KiteML includes built-in explainability tools in `kiteml.intelligence.explainability` to compute SHAP values, feature importances, and decision breakdowns.

---

## 1. Feature Importance

Retrieve permutation and tree-based feature importance rankings from any `Result` object:

```python
from kiteml import train

result = train("housing.csv", target="price", problem_type="regression")

# Inspect feature importance ranking
importances = result.feature_importances()
for feature, score in importances.items():
    print(f"{feature:20s}: {score:.4f}")
```

---

## 2. SHAP (SHapley Additive exPlanations)

Calculate global and local SHAP feature impact scores:

```python
from kiteml.intelligence import ExplainabilityEngine

explainer = ExplainabilityEngine(model=result.model, data=X_train)
shap_values = explainer.compute_shap_values(X_test)

# Plot summary impact
explainer.plot_summary(shap_values, X_test, save_path="shap_summary.png")
```
