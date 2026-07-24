# Python API Quick Start

This guide demonstrates how to use the **KiteML Python API** to train models, inspect diagnostics, run DAG pipelines, and deploy trained artifacts.

---

## 1. Simple AutoML Training (`kiteml.train`)

The `train()` function is KiteML's single-line entry point for automated model training:

```python
from kiteml import train
import pandas as pd

# Load sample dataset
df = pd.read_csv("customer_churn.csv")

# Train an optimal classification model
result = train(data=df, target="Exited")

# Print human-readable evaluation summary
print(result.summary())
```

### Inspected Output:
```text
============================================================
KiteML Training Summary
============================================================
Problem Type : Classification
Best Model   : Random Forest Classifier
Accuracy     : 0.8650
F1 Score     : 0.8421
ROC-AUC      : 0.9120
CV Strategy  : 5-Fold Stratified K-Fold
============================================================
```

---

## 2. Inspecting Execution Diagnostics

Every `train()` execution captures detailed diagnostic feedback accessible via `result.diagnostics()`:

```python
# Output detailed diagnostic status box
print(result.diagnostics())
```

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🪁 KiteML Execution Diagnostics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status             SUCCESS
  Errors             0
  Warnings           1 (KML-W-201: Moderate class imbalance)
  Suggestions        2 (Apply SMOTE resampling or class_weight)
  Validation         Passed (Zero data leakage detected)
  Training           Completed in 1.42s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 3. Making Predictions & Saving Artifacts

```python
# Make predictions on new test data
new_customers = pd.read_csv("new_customers.csv")
predictions = result.predict(new_customers)

# Save trained model to disk
result.save_model("churn_model.pkl")

# Reload model in another process
from kiteml import load
loaded_result = load("churn_model.pkl")
reloaded_preds = loaded_result.predict(new_customers)
```

---

## 4. Unified DAG Pipelines (`KiteMLPipeline`)

For production environments requiring explicit DAG transformation management, pipeline serialization, and SHA-256 integrity verification, use `KiteMLPipeline`:

```python
from kiteml import KiteMLPipeline
import pandas as pd

df = pd.read_csv("housing.csv")

# 1. Initialize orchestrator
pipeline = KiteMLPipeline()

# 2. Build DAG pipeline (auto preprocessing + feature engineering + selection)
build_result = pipeline.build(df, target="price")

# 3. Print pipeline execution summary and replay timeline
print(build_result.report.summary())

# 4. Transform new incoming DataFrames
transformed_df = pipeline.transform(new_df)

# 5. Save production package (.kml) with SHA-256 checksum
pipeline.save("housing_pipeline.kml")

# 6. Load serialized pipeline in production
deploy_pipeline = KiteMLPipeline.load("housing_pipeline.kml")
```

---

## Next Steps

- Explore [CLI Quick Start](quickstart_cli.md) for command-line operations.
- Deep dive into [Epic 4: DAG Orchestration](../user_guides/pipeline/dag_orchestration.md).
- Check out the complete [Python API Reference](../api/core.md).
