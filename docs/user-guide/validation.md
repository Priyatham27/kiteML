# Data Quality & Validation

The **Validation Layer** (Epic 2) enforces data quality contracts to prevent invalid schemas, null column violations, target errors, and data leakage across splits.

---

## 1. Top-Level Validation (`validate()`)

```python
from kiteml import validate

report = validate("data.csv", target="price")

if not report.is_valid:
    for error in report.errors:
        print(f"Error [{error.code}]: {error.message}")
```

---

## 2. Validator Components

- `DatasetValidator`: Validates row/column bounds and null percentages.
- `SchemaValidator`: Enforces column names and data type contracts.
- `TargetValidator`: Checks target existence, label distributions, and cardinality.
- `LeakageValidator`: Verifies zero feature-target leakage prior to fitting.
