# Changelog

All notable changes to KiteML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-05-11

### 🎉 Initial Public Release

KiteML's first stable, production-ready release.

### Added

- **Core ML Engine** — Train classification & regression models with a single function call
- **Intelligent Preprocessing** — Auto null handling, categorical encoding, feature scaling
- **Auto Model Selection** — Cross-validated benchmarking across multiple algorithms
- **Evaluation Reports** — Accuracy, F1, R², RMSE, confusion matrices, and more
- **CLI Ecosystem** — 14 subcommands: `train`, `serve`, `predict`, `profile`, `doctor`, `monitor`, `export`, `init`, `experiment`, `version`, `benchmark`, `dashboard`, `plugins`, `playground`
- **Production Serving** — FastAPI-based model server with validation middleware
- **ONNX Export** — Convert trained models to ONNX format for edge deployment
- **Docker Export** — Generate production Dockerfiles for model deployments
- **Intelligence Layer** — Explainability (SHAP/feature importance), imbalance detection, data profiling
- **Experiment Tracking** — Built-in experiment logging with WandB and MLflow integrations
- **Plugin System** — Extensible plugin SDK with registry for custom model/preprocessor plugins
- **Monitoring** — Model performance monitoring and drift detection
- **Governance** — Model cards, audit logging, and compliance metadata
- **Batch Inference** — High-throughput batch prediction pipeline
- **Real-time Inference** — Low-latency inference with guardrails
- **Multi-format I/O** — CSV, Excel, JSON, Parquet support
- **Comprehensive test suite** — Unit tests, integration tests, smoke tests
- **CI/CD Pipelines** — GitHub Actions for testing, linting, publishing, and docs deployment
- **Full documentation** — MkDocs Material site with getting started guide, API reference, and tutorials

### Infrastructure

- PyPI publishing with trusted publishing (OIDC)
- Semantic versioning with automated release workflow
- Reproducible builds via `python -m build`
- Installation verification pipeline
- Package health badges

---

## [1.0.2] — 2026-07-24

### Epic 4 — Intelligent Machine Learning Pipeline

#### Added

- **Intelligent Preprocessing Engine** — Automatically profiles datasets, infers column types, and generates optimal preprocessing strategies for imputation, encoding, and scaling
- **Feature Engineering Engine** — Discovers and creates meaningful features from datetime, numeric, categorical, interaction, and text columns using domain-aware heuristics
- **Feature Selection Engine** — Multi-selector voting system (`RuleSelector`, `VarianceSelector`, `MissingValueSelector`, `CorrelationSelector`, `ImportanceEstimatorSelector`) producing `KEEP`, `REMOVE`, `FLAG`, `DEFER` decisions
- **Transformation Pipeline** — DAG-based execution engine applying preprocessing, engineering, and selection blueprints in topologically sorted order
- **Pipeline Serialization** — Native `.kml` package format with SHA-256 integrity verification, version metadata, and manifest tracking
- **Pipeline Reporting** — Transparent execution reports with interactive replay timeline, terminal summary display, HTML dashboards, and JSON exports
- **Pipeline Orchestration** — Unified `KiteMLPipeline` API coordinating Intelligence, Validation, DX Diagnostics, and all transformation subsystems through a single workflow call
- **OrchestrationContext** — Shared pipeline state container propagating dataset, blueprints, report, and diagnostics across all stages
- **Execution Lifecycle** — Structured lifecycle tracking: Created → Validated → Planned → Engineered → Selected → Transformed → Reported → Ready
- **Metrics Collection** — Automatic tracking of initial/final dataset shapes, generated features, dropped features, and execution time
- **Event System** — `OrchestrationEventBus` broadcasting structured stage notifications (`DatasetValidated`, `BlueprintsGenerated`, `TransformationCompleted`, etc.)
- **Hook Framework** — `HookRegistry` supporting 8 lifecycle hooks (`before_validation`, `after_validation`, `before_preprocessing`, `after_preprocessing`, `before_transformation`, `after_transformation`, `before_serialization`, `after_serialization`)
- **Shared Pipeline Architecture** — All Epic 4 components (Intelligence, Validation, DX, Preprocessing, Feature Engineering, Selection, Transformation, Reporting, Serialization) accessible through `from kiteml import KiteMLPipeline`

#### Improved

- End-to-end preprocessing workflow unified behind a single public API
- Explainability via detailed execution replay timeline and HTML dashboard reports
- Diagnostics through DX pipeline integration at the orchestration layer
- Extensibility via event bus subscribers and lifecycle hook callbacks
- Pipeline reproducibility through `.kml` serialization with SHA-256 checksums and version manifests

---

## [Unreleased]

_Nothing yet. Stay tuned!_

---

[1.0.2]: https://github.com/kiteml/kiteml/releases/tag/v1.0.2
[1.0.0]: https://github.com/kiteml/kiteml/releases/tag/v1.0.0
[Unreleased]: https://github.com/kiteml/kiteml/compare/v1.0.2...HEAD
