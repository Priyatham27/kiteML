# Dataset Profiling & Data Leakage Detection

The **Intelligence Layer** (Epic 1) automatically profiles raw datasets to uncover statistical distributions, identify high cardinality categorical columns, detect memory optimization opportunities, and intercept data leakage before training begins.

---

## 1. Overview of Data Profiling

The `kiteml.intelligence` engine scans incoming pandas DataFrames across 18 specialized diagnostic analyzers:

```python
from kiteml.intelligence import DataProfiler
import pandas as pd

df = pd.read_csv("raw_data.csv")

# Create data profiler instance
profiler = DataProfiler()
profile_report = profiler.profile(df, target="target_col")

# View summary details
print(profile_report.summary())
```

### Profile Breakdown Metrics

- **Column Types**: Categorized as `numeric`, `categorical`, `datetime`, `text`, or `binary`.
- **Missingness**: Ratios of null and NaN entries per feature.
- **Cardinality**: Ratios of unique value counts to total row counts.
- **Memory Footprint**: Memory usage before and after downcasting.

---

## 2. Detecting Data Leakage

Data leakage occurs when features contain future information or direct target proxies that would not be available during real-time production inference.

### Automatic Leakage Detection Rules

KiteML's `LeakageDetector` flags high-risk columns matching any of the following criteria:

1. **Target Correlation**: Linear Pearson correlation `|r| > 0.98` with continuous target.
2. **Mutual Information**: Near 1.0 mutual information score with categorical target labels.
3. **ID Column Collisions**: Unique identifier columns (e.g. `customer_id`, `ssn`, `timestamp_idx`) containing perfect target mapping.

### Example Usage:

```python
from kiteml.intelligence import LeakageDetector

detector = LeakageDetector()
leakage_report = detector.detect(df, target="Exited")

if leakage_report.has_leakage:
    print(f"WARNING: High data leakage risk detected in columns: {leakage_report.leaky_columns}")
```

!!! danger "Data Leakage Consequences"
    Models trained on leaky datasets will report artificially inflated cross-validation scores (e.g. 0.999 ROC-AUC) but fail catastrophically when deployed to real-world production environments. Always review `kiteml.profile` output before training.
