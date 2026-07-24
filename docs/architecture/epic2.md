# Epic 2 Architecture: Validation Layer

The Validation Layer enforces data quality contracts and pre-flight schema checks prior to model training.

- `DatasetValidator`: Checks shapes and null bounds.
- `SchemaValidator`: Enforces column names and data types.
- `TargetValidator`: Validates target labels and problem compatibility.
- `LeakageValidator`: Verifies zero feature leakage across validation splits.
