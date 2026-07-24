# Quick Start Guide

Get up and running with KiteML in 60 seconds!

---

## 1. Train a Model in 3 Lines of Python

```python
from kiteml import train

# 1. Train an optimal classification model
result = train("data.csv", target="churn")

# 2. View evaluation summary and diagnostics
print(result.summary())
print(result.diagnostics())

# 3. Make predictions on new dataset
predictions = result.predict(new_data)
```

---

## 2. CLI Quick Start

```bash
# Profile dataset quality
kiteml profile data.csv --target churn

# Train model and save artifact
kiteml train data.csv --target churn --save model.pkl

# Serve REST API
kiteml serve model.pkl --port 8000
```
