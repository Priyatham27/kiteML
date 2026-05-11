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

## [Unreleased]

_Nothing yet. Stay tuned!_

---

[1.0.0]: https://github.com/kiteml/kiteml/releases/tag/v1.0.0
[Unreleased]: https://github.com/kiteml/kiteml/compare/v1.0.0...HEAD
