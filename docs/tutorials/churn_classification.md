# Tutorial: End-to-End Customer Churn Classification

This step-by-step tutorial walks through building, evaluating, and deploying a customer churn prediction model using KiteML.

---

## 1. Prerequisites

Ensure `kiteml-ai` is installed:

```bash
pip install kiteml-ai[all]
```

---

## 2. Ingesting & Profiling the Dataset

```python
import pandas as pd
from kiteml import train, validate

# Download sample churn dataset via CLI or load pandas
df = pd.read_csv("https://raw.githubusercontent.com/Priyatham27/kiteML/main/examples/data/churn.csv")

# Run pre-flight data quality validation
report = validate(df, target="Exited")
print(f"Validation Status: {'Passed' if report.is_valid else 'Failed'}")
```

---

## 3. Training the Model (`train()`)

```python
# Train optimal model with automatic cross-validation and diagnostics
result = train(
    data=df,
    target="Exited",
    problem_type="classification",
    test_size=0.2,
    random_state=42
)

# Print evaluation summary and diagnostics
print(result.summary())
print(result.diagnostics())
```

---

## 4. Serving the Model via REST API

Save the trained model and launch the FastAPI web server:

```python
# Save model artifact
result.save_model("churn_model.pkl")
```

Launch server in terminal:
```bash
kiteml serve churn_model.pkl --port 8000
```
