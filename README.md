<p align="center">
  <h1 align="center">🪁 KiteML</h1>
  <p align="center">
    <strong>Train production-grade ML models with a single line of code — intelligent AutoML for everyone.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/kiteml-ai/"><img src="https://img.shields.io/pypi/v/kiteml-ai?color=blue&label=PyPI" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/kiteml-ai/"><img src="https://img.shields.io/pypi/pyversions/kiteml-ai" alt="Python Versions"></a>
    <a href="https://github.com/Priyatham27/kiteML/actions"><img src="https://img.shields.io/github/actions/workflow/status/Priyatham27/kiteML/test.yml?label=tests" alt="Tests"></a>
    <a href="https://github.com/Priyatham27/kiteML/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Priyatham27/kiteML?color=brightgreen" alt="License"></a>
    <a href="https://pepy.tech/project/kiteml-ai"><img src="https://img.shields.io/pepy/dt/kiteml-ai?color=orange" alt="Downloads"></a>
  </p>
</p>

---

KiteML is a full-stack, production-grade **Intelligent AutoML Ecosystem** that automates the complete machine learning lifecycle — from raw dataset ingestion, data profiling, and leakage detection to automated feature engineering, DAG pipeline execution, model selection, REST API serving, ONNX/Docker deployment, and production drift monitoring.

---

## 🚀 Key Features Matrix

| Subsystem | Capabilities |
| :--- | :--- |
| **Core AutoML** | Single-line training (`train()`), auto problem inference (classification & regression), cross-validated model selection, metric evaluation. |
| **Intelligence Layer** | Dataset profiling, automatic data leakage detection, cardinality analysis, severe class imbalance detection, SHAP explainability. |
| **Data Validation** | Schema validation contracts, target variable sanity checks, data quality guards, leak-prevention pre-flight checks. |
| **Intelligent ML Pipeline** | Automated imputation, scaling, and categorical encoding; feature engineering; voting feature selection; DAG execution engine; `.kml` package serialization with SHA-256 integrity checksums. |
| **Developer Experience (DX)** | Structured error catalog (`KML-XXX`), configurable warning policies (`KML-W-XXX`), context-aware typo suggestions (`match_column_name`), and execution diagnostic boxes. |
| **Production Deployment** | Automated FastAPI REST server (`kiteml serve`), optimized ONNX export, Docker container packaging, realtime & batch inference guardrails. |
| **CLI Ecosystem** | 14 rich subcommands: `train`, `serve`, `predict`, `profile`, `doctor`, `init`, `playground`, `dashboard`, `monitor`, `export`, `benchmark`, `experiment`, `plugins`, `version`. |
| **MLOps & Governance** | Automated Model Card generation (`model_card.json`), audit logging, population stability drift monitoring (PSI/KS-test), WandB & MLflow experiment tracking adapters. |

---

## 🏛️ System Architecture

```
                                    🪁 KiteML Ecosystem Architecture
                                    
  ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────────────┐
  │  Epic 1: Intelligence│   │   Epic 2: Validation │   │  Epic 3: Developer Experience    │
  │  • Data Profiling    │   │  • Schema Checks     │   │  • Structured Errors (KML-XXX)   │
  │  • Leakage Detection │   │  • Target Sanity     │   │  • Warning Engine (KML-W-XXX)    │
  │  • Imbalance Scans   │   │  • Quality Rules     │   │  • Typo Suggestions Engine       │
  │  • SHAP & Importance │   │  • Leak Guards       │   │  • 14-Command CLI Suite          │
  └──────────┬───────────┘   └──────────┬───────────┘   └────────────────┬─────────────────┘
             │                          │                                │
             └──────────────────────────┼────────────────────────────────┘
                                        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                         Epic 4: Intelligent ML Pipeline                                │
  │  • Preprocessing Engine   • Automated Feature Creation   • Voting Feature Selection    │
  │  • DAG Execution Pipeline • Native .kml Serialization    • Execution Replay Timeline   │
  │  • Unified KiteMLPipeline Orchestrator                                                 │
  └─────────────────────────────────────┬──────────────────────────────────────────────────┘
                                        │
                                        ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                     Epic 5: Intelligent Training & Deployment                          │
  │  • Cross-Validated Model Selection    • FastAPI REST Serving     • ONNX Exporter         │
  │  • Docker Container Packager          • Inference Guardrails     • Production Drift Monitor│
  │  • Model Cards & Compliance Logs      • WandB & MLflow Adapters  • Extensible Plugin SDK   │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Requirements

- **Python**: `3.10`, `3.11`, `3.12`, or `3.13` (officially supported)

---

## 📦 Installation

```bash
pip install kiteml-ai
```

> **Import Note**: The package name on PyPI is `kiteml-ai`. The Python import remains `import kiteml`.

### Optional Extras

```bash
pip install kiteml-ai[serving]   # FastAPI REST model server & OpenAPI docs
pip install kiteml-ai[onnx]      # ONNX model graph conversion & ONNX Runtime
pip install kiteml-ai[wandb]     # Weights & Biases experiment tracking
pip install kiteml-ai[mlflow]    # MLflow tracking & model registry
pip install kiteml-ai[docs]      # MkDocs documentation builder
pip install kiteml-ai[all]       # Complete ecosystem dependencies
```

---

## ⚡ Quick Start

### 1. Simple AutoML Training (`train()`)

```python
from kiteml import train, load

# Train classification model with automatic preprocessing & model selection
result = train("customer_churn.csv", target="Exited")

# Print summary & execution diagnostics
print(result.summary())
print(result.diagnostics())

# Make predictions on new dataset
predictions = result.predict(new_df)

# Save best model artifact
result.save_model("churn_model.pkl")

# Reload saved model
saved_result = load("churn_model.pkl")
```

### 2. End-to-End Pipeline Orchestration (`KiteMLPipeline`)

For fine-grained control over preprocessing, feature engineering, feature selection, and `.kml` package serialization:

```python
from kiteml import KiteMLPipeline
import pandas as pd

df = pd.read_csv("housing.csv")

# Initialize and fit unified pipeline
pipeline = KiteMLPipeline()
build_result = pipeline.build(df, target="price")

# Inspect pipeline summary and replay timeline
print(build_result.report.summary())

# Transform new dataset through trained DAG
transformed_df = pipeline.transform(new_df)

# Save as production-ready .kml package with SHA-256 integrity verification
pipeline.save("housing_pipeline.kml")

# Load pipeline for deployment
loaded_pipeline = KiteMLPipeline.load("housing_pipeline.kml")
```

### 3. Command Line Interface (CLI)

```bash
# 1. Train a model with automatic reporting
kiteml train data.csv --target label --save model.pkl

# 2. Profile a dataset for leakage, outliers, and quality issues
kiteml profile data.csv --target label

# 3. Serve a model via production REST API (FastAPI)
kiteml serve model.pkl --port 8000

# 4. Make batch predictions
kiteml predict model.pkl new_data.csv --output predictions.csv

# 5. Run environment diagnostics
kiteml doctor
```

---

## 🖥️ CLI Commands Ecosystem

KiteML includes 14 subcommands out-of-the-box:

```
kiteml
├── train       Train models with automatic preprocessing & cross-validation
├── serve       Start a FastAPI REST server with OpenAPI documentation
├── predict     Generate predictions from a trained model or .kml package
├── profile     Analyze dataset quality, imbalance, and leakage risks
├── doctor      Validate Python environment, dependencies, and GPU drivers
├── init        Scaffold a new production ML project template
├── playground  Download sample datasets (churn, housing, iris)
├── dashboard   Launch local interactive HTML performance dashboard
├── monitor     Check statistical data drift (PSI) on production data streams
├── export      Export models to ONNX graph format or Docker containers
├── benchmark   Run execution benchmarks across model algorithms
├── experiment  List and inspect local experiment tracking logs
├── plugins     List, install, and manage custom pipeline plugins
└── version     Display KiteML ecosystem versions and build info
```

---

## 🚀 Production Deployment & MLOps

### REST API Serving
Deploy models with auto-generated OpenAPI documentation (`http://localhost:8000/docs`):

```bash
kiteml serve model.pkl --port 8000 --workers 4
```

### ONNX & Docker Export
```bash
# Export model to optimized ONNX graph format
kiteml export model.pkl --format onnx --output model.onnx

# Generate production Dockerfile & container package
kiteml export model.pkl --format docker --output ./docker_deploy/
```

### Model Governance & Drift Monitoring
Every trained model automatically generates a structured `model_card.json` containing lineage, evaluation metrics, and feature importance. Monitor live predictions for statistical distribution shift:

```python
from kiteml.monitoring import DriftMonitor

monitor = DriftMonitor(reference_data=train_df)
drift_report = monitor.detect_drift(current_inference_df)
print(drift_report.summary())
```

---

## 📁 Repository Structure

```
kiteml/
├── core.py              # Main train() function & execution entrypoint
├── orchestration/       # KiteMLPipeline — unified AutoML orchestrator
├── pipeline/            # DAG transformation engine & DX pipeline
├── preprocessing/       # Auto cleaning, encoding, and scaling
├── feature_engineering/ # Automatic feature discovery & creation engine
├── feature_selection/   # Voting-based multi-selector feature engine
├── serialization/       # Native .kml pipeline packaging & SHA-256 check
├── reporting/           # Execution reports, timeline replay, HTML dashboards
├── intelligence/        # Data profiler, leakage detector, SHAP explainability
├── validation/          # Schema, target, and quality validation rules
├── exceptions/          # Structured error framework (KML-XXX)
├── warnings/            # Structured warning policies (KML-W-XXX)
├── suggestions/         # Context-aware typo matcher & recommendations
├── models/              # Classifier & regressor wrappers
├── evaluation/          # Evaluation metrics, ROC-AUC, residual analysis
├── serving/             # FastAPI production server engine
├── deployment/          # ONNX conversion, Docker export, guardrails
├── monitoring/          # Production data & concept drift monitor
├── experiments/         # Experiment tracking & run logger
├── governance/          # Model Cards (model_card.json) & audit log
├── plugins/             # Plugin SDK for custom transformation stages
└── cli/                 # 14-command Rich CLI ecosystem
```

---

## 📖 Documentation & Community

Comprehensive documentation is available at [https://priyatham27.github.io/kiteML/](https://priyatham27.github.io/kiteML/):

- [Getting Started Guide](https://priyatham27.github.io/kiteML/getting_started/)
- [User Guides (Epics 1–5)](https://priyatham27.github.io/kiteML/user_guides/intelligence/profiling_leakage/)
- [CLI Reference](https://priyatham27.github.io/kiteML/getting_started/quickstart_cli/)
- [API Reference](https://priyatham27.github.io/kiteML/api/core/)
- [Architecture Specifications](https://priyatham27.github.io/kiteML/architecture/overview/)

---

## 🛠️ Contributing

We welcome community contributions! Please read our [Contributing Guide](community/CONTRIBUTING.md) to get started:

```bash
# Clone repository
git clone https://github.com/Priyatham27/kiteML.git
cd kiteML

# Create virtual environment and install in editable mode
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"

# Run unit tests
pytest tests/
```

---

## 📄 License

KiteML is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with care by the KiteML Team</sub>
</p>
