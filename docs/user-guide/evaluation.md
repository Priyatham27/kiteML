# Evaluation & Metrics

KiteML computes comprehensive evaluation metrics for classification and regression tasks.

---

## 1. Classification Metrics

- **Accuracy**, **Precision**, **Recall**, **F1 Score**, **ROC-AUC**, **Confusion Matrix**.

---

## 2. Regression Metrics

- **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, **R² Score**.

```python
result = train("housing.csv", target="price", problem_type="regression")
print("RMSE:", result.metrics["rmse"])
print("R2 Score:", result.metrics["r2"])
```
