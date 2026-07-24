# Imbalance & Outlier Detection

The **Intelligence Layer** provides specialized detectors to identify class distribution imbalances and statistical outliers before model training.

---

## 1. Class Imbalance Detection

Class imbalance occurs when one target class significantly outnumbers other classes (e.g. 98% negative vs 2% positive in fraud detection).

```python
from kiteml.intelligence import ImbalanceDetector
import pandas as pd

df = pd.read_csv("fraud_data.csv")

detector = ImbalanceDetector()
report = detector.detect(df, target="is_fraud")

print(f"Imbalance Ratio: {report.imbalance_ratio:.2f}")
print(f"Severity Level : {report.severity}")
```

### Escalation Thresholds:
- **Balanced**: Minority class `>= 40%`.
- **Low Imbalance**: Minority class `20% - 39%`.
- **Moderate Imbalance (`KML-W-201`)**: Minority class `5% - 19%`.
- **Critical Imbalance (`KML-W-202`)**: Minority class `< 5%`.

---

## 2. Outlier Detection

The `OutlierDetector` identifies anomalous numeric data rows using Interquartile Range (IQR) and Z-score methods:

```python
from kiteml.intelligence import OutlierDetector

outlier_detector = OutlierDetector(method="iqr", factor=1.5)
outlier_report = outlier_detector.detect(df)

print(f"Outlier Row Count: {outlier_report.outlier_count}")
print(f"Affected Columns: {outlier_report.affected_columns}")
```
