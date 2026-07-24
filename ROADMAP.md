# KiteML Project Roadmap & Development Vision

This document outlines the strategic roadmap, feature milestones, and architectural vision for **KiteML**.

---

## 📍 Current Release: v1.0.2 (Completed Epics 1–5)

KiteML v1.0.2 marks the completion of the 5 core foundational epics:

- [x] **Epic 1: Intelligence Layer** — Data profiling, automatic leakage detection, cardinality analysis, class imbalance detection, SHAP explainability.
- [x] **Epic 2: Validation Layer** — Quality rules, schema contracts, target validation, leakage prevention checks.
- [x] **Epic 3: Developer Experience (DX)** — Structured exception catalog (`KML-XXX`), warning escalation policy engine (`KML-W-XXX`), fuzzy string column matcher, 14-command Rich CLI ecosystem.
- [x] **Epic 4: Intelligent ML Pipeline** — Preprocessing engine, feature engineering, multi-selector voting feature selection, DAG execution engine, `.kml` package serialization with SHA-256 integrity checksums.
- [x] **Epic 5: Intelligent Training & Deployment** — Cross-validated model selection, FastAPI REST serving (`kiteml serve`), ONNX graph export, Docker container packager, PSI drift monitoring, Model Cards (`model_card.json`).

---

## 🚀 Near-Term Milestone: v1.1.0 (Q4 2026)

Targeting advanced time-series, text embeddings, and distributed computing:

- [ ] **Automated Time-Series Forecasting Engine**: Single-line `train_timeseries()` supporting ARIMA, Prophet, and Lag-feature extraction.
- [ ] **Multi-Modal Text Feature Extraction**: Transformer-based embeddings (Hugging Face sentence-transformers integration).
- [ ] **Distributed Hyperparameter Optimization**: Distributed Ray and Optuna cluster tuning support.
- [ ] **Polars Data Engine**: High-performance Rust-backed Polars DataFrame support alongside pandas.

---

## 🔮 Medium-Term Milestone: v1.2.0 (Q1 2027)

Targeting enterprise MLOps and cloud deployments:

- [ ] **Self-Healing Production Pipelines**: Automated model retraining triggers on population stability index (PSI) drift alerts.
- [ ] **Kubernetes Deployment Packager**: Helm chart generation and KServe inference service manifests.
- [ ] **Automated Data Quality Repair**: Automated out-of-range value capping and smart missingness imputation models.

---

## 🌐 Long-Term Vision: v2.0.0 (Q2 2027)

- [ ] **LLM-Assisted Automated Feature Discovery**: Natural language query interface for feature creation and pipeline diagnostics.
- [ ] **End-to-End AutoML Cloud Platform**: Web dashboard for collaborative experiment tracking and model lineage visualization.
