# Data Quality & Schema Validation Contracts

The **Validation Layer** (Epic 2) enforces data quality contracts to prevent invalid schemas, null column violations, target errors, and data leakage across splits.

---

## 1. Top-Level Dataset Validation (`validate()`)

Run pre-flight checks on DataFrames or raw dataset files:

```python
from kiteml import validate

report = validate("data.csv", target="price")

if not report.is_valid:
    print("Validation Errors Found:")
    for error in report.errors:
        print(f"  - [{error.code}] {error.message}")
```

---

## 2. Component Validators

KiteML includes specialized validation classes in `kiteml.validation`:

- `DatasetValidator`: Enforces row count, column count, null percentage thresholds.
- `SchemaValidator`: Enforces column names and data type contracts.
- `TargetValidator`: Validates target variable presence, unique values, and label distributions.
- `LeakageValidator`: Verifies zero feature-target leakage prior to model fitting.

```python
from kiteml.validation import SchemaValidator

validator = SchemaValidator(expected_schema={"age": "int64", "income": "float64"})
result = validator.validate(df)
```
