# KiteML Technical Architecture Specification (v1.0.2)

## 1. System Overview

**KiteML** is an intelligent, full-stack AutoML framework designed to automate the complete machine learning lifecycle — from raw dataset ingestion and quality validation to DAG transformation, cross-validated model selection, REST API serving, containerized deployment, and production drift monitoring.

KiteML is architected around five completed epics:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     KiteML Ecosystem                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────────────┐
  │  Epic 1: Intelligence│  │   Epic 2: Validation │  │  Epic 3: Developer Experience    │
  │                      │  │                      │  │                                  │
  │ • Data Profiling     │  │ • Schema Checks      │  │ • Structured Exceptions (KML-X)  │
  │ • Leakage Detection  │  │ • Target Checks      │  │ • Warning Policy Engine (KML-W)  │
  │ • Imbalance Analysis │  │ • Quality Rules      │  │ • Context-Aware Suggestions      │
  │ • SHAP & Importance  │  │ • Leakage Rules      │  │ • 14-Command Rich CLI Ecosystem  │
  └──────────┬───────────┘  └──────────┬───────────┘  └────────────────┬─────────────────┘
             │                         │                               │
             └─────────────────────────┼───────────────────────────────┘
                                       ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────┐
  │                         Epic 4: Intelligent ML Pipeline                               │
  │                                                                                       │
  │ • Auto Preprocessing   • Auto Feature Engineering   • Voting Feature Selection       │
  │ • DAG Pipeline Engine  • Pipeline Serialization (.kml) • Interactive Replay Timeline   │
  │ • Unified KiteMLPipeline Orchestration Engine                                         │
  └────────────────────────────────────┬──────────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌───────────────────────────────────────────────────────────────────────────────────────┐
  │                     Epic 5: Intelligent Training & Deployment                         │
  │                                                                                       │
  │ • Model Catalog & HPO   • FastAPI Model Serving     • ONNX Exporter & Optimization      │
  │ • Docker Packaging     • Realtime Guardrails      • Drift & Concept Monitoring        │
  │ • Model Cards & Audit   • WandB / MLflow Tracking  • Plugin SDK Extensions             │
  └───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Epic Breakdown & Component Design

### Epic 1 — Intelligence Layer (`kiteml.intelligence`, `kiteml.profiling`)
The Intelligence Layer analyzes raw data prior to model construction to surface dataset characteristics, potential pitfalls, and optimization opportunities:
- **`column_analyzer.py` / `data_profiler.py`**: Ingests pandas DataFrames and infers statistical distributions, column types (numeric, categorical, datetime, text), missingness ratios, and memory footprints.
- **`cardinality_analyzer.py`**: Identifies high-cardinality categorical features and recommends appropriate encoding strategies (target encoding vs frequency vs one-hot).
- **`correlation_analyzer.py`**: Calculates linear (Pearson) and non-linear (Spearman) correlations to flag redundant collinear features.
- **`datetime_detector.py` / `text_detector.py`**: Detects ISO-8601 strings and unstructured text columns for feature extraction.
- **`imbalance_detector.py`**: Analyzes classification target class distributions and calculates imbalance ratios to trigger SMOTE/class-weight recommendations.
- **`leakage_detector.py`**: Scans dataset columns against target variables to detect data leakage (features with near 1.0 mutual information or target correlation).
- **`outlier_detector.py`**: Uses IQR and Z-score algorithms to identify anomalous rows.
- **`memory_optimizer.py`**: Downcasts numeric column dtypes (e.g. `int64` to `int32`/`int16`) to reduce RAM usage.
- **`explainability.py`**: Computes SHAP values, permutation feature importances, and model decision breakdowns.

### Epic 2 — Validation Layer (`kiteml.validation`)
The Validation Layer enforces data quality contracts before transformation or training begins:
- **`DatasetValidator`**: Validates overall dataset shape, null thresholds, duplicate rows, and schema integrity.
- **`SchemaValidator`**: Enforces strict column name, data type, and structure matching for training and inference DataFrames.
- **`TargetValidator`**: Validates target column existence, cardinality, problem type compatibility (binary vs multiclass vs continuous regression), and label formatting.
- **`LeakageValidator`**: Re-evaluates post-split features to guarantee no future temporal or target information bleeds into training folds.
- **`DataQualityValidator`**: Executes rules against missing values, zero variance columns, and constant value columns.

### Epic 3 — Developer Experience Framework (`kiteml.exceptions`, `kiteml.warnings`, `kiteml.suggestions`, `kiteml.cli`)
The DX Framework transforms raw Python errors into actionable developer guidance:
- **Exceptions (`kiteml.exceptions`)**: Hierarchy of error classes inheriting from `KiteMLError`. Each error carries a distinct catalog code (e.g. `KML-101`), precise contextual attributes, and multi-format renderers (Terminal Rich formatting, Markdown, HTML, JSON).
- **Warnings (`kiteml.warnings`)**: Categorized warning system (`KML-W-XXX`) with configurable escalation policies (`ignore`, `info`, `warn`, `error`).
- **Suggestions Engine (`kiteml.suggestions`)**: Contextual recommendation generator with fuzzy string matching (e.g. suggesting `Price` when user inputs `prcie`) and clear `Why?` explanations.
- **CLI Ecosystem (`kiteml.cli`)**: 14 subcommands built with `rich` and `argparse` providing full command-line capability:
  `train`, `serve`, `predict`, `profile`, `doctor`, `init`, `playground`, `dashboard`, `monitor`, `export`, `benchmark`, `experiment`, `plugins`, `version`.

### Epic 4 — Intelligent ML Pipeline (`kiteml.orchestration`, `kiteml.pipeline`, `kiteml.preprocessing`, `kiteml.feature_engineering`, `kiteml.feature_selection`, `kiteml.serialization`, `kiteml.reporting`)
Epic 4 automates end-to-end data transformation into a reproducible DAG pipeline:
- **Preprocessing Engine (`kiteml.preprocessing`)**: Auto-generates imputation, encoding, and scaling pipelines based on inferred schema.
- **Feature Engineering (`kiteml.feature_engineering`)**: Extracts datetime features (year, month, day, dayofweek, hour), text metrics (length, word count), and numeric interaction terms.
- **Feature Selection (`kiteml.feature_selection`)**: Multi-selector voting system aggregating variance thresholding, correlation filtering, mutual information, and model-based feature importance.
- **DAG Execution Engine (`kiteml.pipeline`)**: Directed Acyclic Graph that manages transformation stage dependencies, state persistence, and diagnostic hooks.
- **Serialization (`kiteml.serialization`)**: Encapsulates pipelines and trained models into native `.kml` binary packages protected by SHA-256 integrity checksums.
- **Reporting (`kiteml.reporting`)**: Generates interactive execution reports, HTML dashboards, JSON logs, and replay timelines.
- **Orchestrator (`kiteml.orchestration.KiteMLPipeline`)**: Top-level API coordinating all pipeline stages in a single unified workflow.

### Epic 5 — Intelligent Training & Deployment (`kiteml.models`, `kiteml.selection`, `kiteml.training`, `kiteml.optimization`, `kiteml.serving`, `kiteml.deployment`, `kiteml.monitoring`, `kiteml.governance`, `kiteml.registry`, `kiteml.plugins`, `kiteml.experiments`, `kiteml.integrations`)
Epic 5 turns validated pipelines into production deployments:
- **Model Training & Selection (`kiteml.models`, `kiteml.selection`, `kiteml.training`)**: Trains multiple algorithm candidates (LightGBM/XGBoost/scikit-learn classifiers & regressors) using stratified k-fold cross-validation, hyperparameter tuning, and automated metric scoring.
- **FastAPI Model Server (`kiteml.serving`)**: Auto-generates production-ready REST APIs with OpenAPI docs (`/docs`), `/predict` endpoints, and health probes (`/health`).
- **Deployment Packaging (`kiteml.deployment`)**:
  - `onnx_export.py`: Converts scikit-learn/KiteML models into optimized ONNX graph representations.
  - `docker_export.py`: Generates containerized `Dockerfile`, `requirements.txt`, and deployment artifacts.
  - `inference_guardrails.py`: Enforces runtime schema validation, missing column imputation, and prediction latency limits.
- **Drift Monitoring (`kiteml.monitoring`)**: Tracks statistical data drift (PSI, KS-test) and concept drift on live inference streams.
- **Governance & MLOps (`kiteml.governance`, `kiteml.registry`, `kiteml.experiments`, `kiteml.integrations`)**: Produces structured Model Cards (`model_card.json`), audit trails, and integration adapters for WandB and MLflow.
- **Plugin SDK (`kiteml.plugins`)**: Extensible framework for developer-authored custom preprocessing and feature engineering stages.

---

## 3. Technology Stack & Dependencies

- **Core Runtime**: Python 3.10, 3.11, 3.12, 3.13
- **Data Engine**: `pandas >= 2.2` (or `>= 3.0` on Python 3.11+), `numpy >= 1.23.0`
- **Machine Learning**: `scikit-learn >= 1.2.0`, `joblib >= 1.2.0`
- **CLI & UI**: `rich >= 13.0.0`
- **Optional Dependencies**: `fastapi`, `uvicorn`, `pydantic` (Serving), `skl2onnx`, `onnxruntime` (ONNX), `wandb` (W&B), `mlflow` (MLflow), `mkdocs-material` (Docs)
