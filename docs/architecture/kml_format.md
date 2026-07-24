# `.kml` Binary Package Format & Security Checksum Specification

The **`.kml` (KiteML Package)** is a native binary container format designed for reproducible deployment of transformation DAG pipelines, categorical encoders, feature scalers, and trained model weights.

---

## 1. Internal Package Archive Structure

A `.kml` package is a compressed archive containing the following components:

```text
model.kml (Archive)
├── manifest.json            # Package metadata, version, SHA-256 checksum
├── pipeline_dag.pkl         # Serialized DAG transformation pipeline state
├── model_weights.joblib     # Fitted model weights artifact
├── feature_schema.json      # Pre-transform & post-transform column schemas
└── model_card.json          # Governance lineage & evaluation metrics
```

---

## 2. Manifest Schema & SHA-256 Checksum Validation

When `pipeline.save("model.kml")` is executed, KiteML computes a SHA-256 hash across `pipeline_dag.pkl` and `model_weights.joblib`, writing the hash into `manifest.json`:

```json
{
  "kiteml_version": "1.0.2",
  "created_at": "2026-07-24T10:00:00Z",
  "problem_type": "classification",
  "sha256_checksum": "a8f5f167f44f4964e6c998dee827110c...",
  "target_column": "Exited"
}
```

Upon invoking `KiteMLPipeline.load("model.kml")`, KiteML recalculates the SHA-256 hash of the archive binaries and compares it against `manifest.json`. If a mismatch occurs (due to disk corruption or tampering), loading is halted and a `ChecksumValidationError` is raised.
