# Model Training & Cross-Validation

KiteML automates model training using stratified k-fold cross-validation across a candidate algorithm suite.

---

## 1. Supported Model Algorithms

- **Classification**: Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, LightGBM.
- **Regression**: Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor, LightGBM Regressor.

---

## 2. Programmatic Training

```python
from kiteml import train

result = train(data="housing.csv", target="price", problem_type="regression")
print(result.summary())
```
