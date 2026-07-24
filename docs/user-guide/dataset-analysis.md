# Dataset Analysis & Profiling

KiteML's **Intelligence Layer** (Epic 1) analyzes raw datasets prior to transformation to identify schema types, missingness, cardinality, memory footprint, imbalance, and data leakage.

---

## 1. Profiling a Dataset (`DataProfiler`)

```python
from kiteml.intelligence import DataProfiler
import pandas as pd

df = pd.read_csv("dataset.csv")

profiler = DataProfiler()
report = profiler.profile(df, target="target_col")

print(report.summary())
```

---

## 2. Detecting Data Leakage (`LeakageDetector`)

```python
from kiteml.intelligence import LeakageDetector

detector = LeakageDetector()
leakage_report = detector.detect(df, target="target_col")

if leakage_report.has_leakage:
    print("Warning: Leakage detected in columns:", leakage_report.leaky_columns)
```
