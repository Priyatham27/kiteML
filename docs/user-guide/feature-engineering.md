# Automated Feature Engineering

KiteML extracts predictive features from datetimes, text strings, and numeric interactions.

---

## 1. Feature Generators

- **Datetime**: Year, month, day, dayofweek, hour, is_weekend.
- **Interactions**: Numeric ratios and product terms.
- **Text Metrics**: String length, word count, digit ratio.

```python
from kiteml.feature_engineering import FeatureEngineeringPlanner

planner = FeatureEngineeringPlanner(enable_datetime=True, enable_interactions=True)
transformed_df = planner.fit_transform(df)
```
