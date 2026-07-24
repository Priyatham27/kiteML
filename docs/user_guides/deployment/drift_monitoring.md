# Production Data & Concept Drift Monitoring

The `kiteml.monitoring` module tracks feature distribution drift and target concept drift in production inference streams.

---

## 1. Population Stability Index (PSI) Drift Detection

Track feature distribution shifts between baseline training data and live inference data:

```python
from kiteml.monitoring import DriftMonitor
import pandas as pd

train_df = pd.read_csv("train_baseline.csv")
inference_df = pd.read_csv("live_inference.csv")

# Initialize monitor
monitor = DriftMonitor(reference_data=train_df)

# Run drift detection report
report = monitor.detect_drift(inference_df)

print(report.summary())
```

### PSI Alert Levels:
- **`PSI < 0.10`**: No significant distribution drift.
- **`0.10 <= PSI < 0.25`**: Moderate drift (Trigger retraining alert).
- **`PSI >= 0.25`**: Significant drift (`KML-W-501`: Immediate retraining recommended).
