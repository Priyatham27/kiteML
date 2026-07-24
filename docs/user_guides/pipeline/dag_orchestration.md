# DAG Orchestration & Pipeline Serialization (.kml)

The **Intelligent ML Pipeline** (Epic 4) organizes data transformations into a Directed Acyclic Graph (DAG) managed by `KiteMLPipeline`.

---

## 1. Unified Pipeline Building (`KiteMLPipeline`)

```python
from kiteml import KiteMLPipeline
import pandas as pd

df = pd.read_csv("dataset.csv")

# Initialize orchestrator
pipeline = KiteMLPipeline()

# Execute DAG build (preprocessing -> engineering -> selection)
build_result = pipeline.build(df, target="target_column")

print(build_result.report.summary())
```

---

## 2. Serialization to `.kml` Bundles

KiteML serializes pipeline DAG states, encodings, scales, and model weights into native binary `.kml` packages protected by SHA-256 integrity checksums:

```python
# Save serialized pipeline package
pipeline.save("production_pipeline.kml")

# Load pipeline in production
deploy_pipeline = KiteMLPipeline.load("production_pipeline.kml")

# Transform inference DataFrame
processed_df = deploy_pipeline.transform(inference_df)
```

!!! note "Integrity Verification"
    Every `.kml` package verifies SHA-256 checksums upon loading. If a binary file has been tampered with or corrupted on disk, `KiteMLPipeline.load()` raises a `ChecksumValidationError`.
