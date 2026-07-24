# Prediction & Inference

Generate real-time or batch predictions from trained `Result` objects or `.kml` pipeline packages.

---

## 1. Batch Prediction via CLI

```bash
kiteml predict model.pkl new_data.csv --output predictions.csv
```

---

## 2. Python API Inference

```python
predictions = result.predict(new_dataframe)
```
