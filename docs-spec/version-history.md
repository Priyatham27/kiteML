# KiteML Version History & Release Matrix (`docs-spec/version-history.md`)

This specification tracks the evolution of KiteML features, API contracts, and milestone releases.

---

## Release Summary Matrix

| Version | Release Date | Key Focus & Major Features Introduced |
| :--- | :--- | :--- |
| **v0.1.0 – v0.5.0** | Q3 2025 | Initial proof-of-concept AutoML training scripts for CSV datasets. |
| **v0.8.0** | Q4 2025 | Prototype DX Exception framework and CLI subcommands. |
| **v1.0.0** | Q1 2026 | Initial major stable release introducing `train()` and `load()` top-level APIs. |
| **v1.0.1** | Q2 2026 | PyPI package rebrand (`kiteml-ai`), Python 3.10+ requirement enforcement, dependency updates (`pandas >= 2.2`). |
| **v1.0.2** *(Current)* | Q3 2026 | Completion of Epics 1–5: Intelligent Preprocessing, Multi-Selector Feature Selection, `.kml` SHA-256 Serialization, DAG Engine, FastAPI Serving, ONNX Exporter, Docker Packaging, Drift Monitoring, and Governance Model Cards. |

---

## Detailed Milestone Changelog

### Version 1.0.2 (Current Production Milestone)
- **Epic 1 (Intelligence Layer)**: Added `imbalance_detector`, `leakage_detector`, `outlier_detector`, `memory_optimizer`, `cardinality_analyzer`, `datetime_detector`, `text_detector`, and `explainability` (SHAP).
- **Epic 2 (Validation Layer)**: Released `DatasetValidator`, `SchemaValidator`, `TargetValidator`, and `LeakageValidator`.
- **Epic 3 (Developer Experience)**: Standardized structured exception taxonomy `KML-XXX`, warning escalation policy engine `KML-W-XXX`, fuzzy string matcher, and expanded CLI to 14 subcommands.
- **Epic 4 (Intelligent Pipeline)**: Integrated `KiteMLPipeline` DAG orchestrator, preprocessing blueprint engine, voting feature selection system, `.kml` package serializer with SHA-256 checksums, and execution replay timeline.
- **Epic 5 (Training & Deployment)**: Released FastAPI server generator, ONNX model graph converter, Docker export packager, production drift monitor (PSI/KS-test), Model Card governance (`model_card.json`), and WandB/MLflow integration adapters.

### Version 1.0.1
- Standardized PyPI package name as `kiteml-ai` while retaining `import kiteml` module namespace.
- Standardized Python support range: Python 3.10, 3.11, 3.12, 3.13.
- Upgraded dependencies: `pandas >= 2.2`, `scikit-learn >= 1.2.0`, `rich >= 13.0.0`.

### Version 1.0.0
- Initial stable release of core `train()` and `load()` functions.
- Basic cross-validation, model selection (logistic regression, random forest, gradient boosting), and text summary report generation.
