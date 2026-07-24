# Automated Preprocessing

KiteML automatically cleans, imputes missing values, scales numerical features, and encodes categorical columns.

---

## 1. Preprocessing Strategy

```python
from kiteml.preprocessing import PreprocessingPlanner

planner = PreprocessingPlanner(
    numeric_strategy="median",
    scaling_strategy="standard",
    categorical_strategy="onehot"
)

blueprint = planner.create_blueprint(df, target="target_col")
```

---

## 2. Scalers & Encoders

- **Numeric Scaling**: Standard Scaler, MinMax Scaler, Robust Scaler.
- **Categorical Encoding**: One-Hot Encoding, Target Encoding, Frequency Encoding.
