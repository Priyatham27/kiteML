# Building Your First Model

This step-by-step tutorial guides you through training, evaluating, saving, and reloading your first machine learning model with KiteML.

---

## Step 1: Create a Dataset

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 45, 35, 50, 23, 62, 48, 29],
    "income": [50000, 90000, 62000, 110000, 48000, 125000, 98000, 54000],
    "credit_score": [650, 720, 680, 800, 610, 750, 710, 640],
    "purchased": [0, 1, 1, 1, 0, 1, 1, 0]
})
```

---

## Step 2: Train Model (`kiteml.train`)

```python
from kiteml import train

result = train(df, target="purchased", problem_type="classification")
```

---

## Step 3: Inspect Summary & Diagnostics

```python
print(result.summary())
print(result.diagnostics())
```

---

## Step 4: Save & Reload Model

```python
from kiteml import load

# Save artifact
result.save_model("my_first_model.pkl")

# Reload in production
model = load("my_first_model.pkl")
preds = model.predict(df.drop(columns=["purchased"]))
print("Predictions:", preds)
```
