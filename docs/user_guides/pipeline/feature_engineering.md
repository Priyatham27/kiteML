# Automated Feature Engineering

KiteML's `feature_engineering` engine automatically discovers and generates predictive feature representations from raw data.

---

## 1. Automated Feature Generators

The `FeatureEngineeringEngine` inspects feature types and synthesizes new candidate features:

### Datetime Feature Extraction
Automatically extracts calendar and time components from ISO-8601 datetimes:
- `year`, `month`, `day`, `dayofweek`, `dayofyear`, `hour`, `is_weekend`.

### Numeric Interaction Terms
Combines high-importance numeric pairs via arithmetic operators:
- Ratios (`feature_A / feature_B`)
- Products (`feature_A * feature_B`)

### Text Metrics
Extracts structural properties from raw text columns:
- `char_length`, `word_count`, `digit_count`, `uppercase_ratio`.

---

## 2. Programmatic Usage

```python
from kiteml.feature_engineering import FeatureEngineeringPlanner

planner = FeatureEngineeringPlanner(
    enable_datetime=True,
    enable_interactions=True,
    enable_text=True,
)

transformed_df = planner.fit_transform(df)
```
