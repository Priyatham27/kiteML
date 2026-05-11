<p align="center">
  <h1 align="center">🪁 KiteML</h1>
  <p align="center">
    <strong>Train production-grade ML models with a single line of code.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/kiteml/"><img src="https://img.shields.io/pypi/v/kiteml?color=blue&label=PyPI" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/kiteml/"><img src="https://img.shields.io/pypi/pyversions/kiteml" alt="Python Versions"></a>
    <a href="https://github.com/kiteml/kiteml/actions"><img src="https://img.shields.io/github/actions/workflow/status/kiteml/kiteml/test.yml?label=tests" alt="Tests"></a>
    <a href="https://codecov.io/gh/kiteml/kiteml"><img src="https://img.shields.io/codecov/c/github/kiteml/kiteml?color=green" alt="Coverage"></a>
    <a href="https://github.com/kiteml/kiteml/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kiteml/kiteml?color=brightgreen" alt="License"></a>
    <a href="https://pepy.tech/project/kiteml"><img src="https://img.shields.io/pepy/dt/kiteml?color=orange" alt="Downloads"></a>
  </p>
</p>

---

KiteML is an intelligent AutoML framework that automates the entire ML pipeline — from raw data to production-ready models. It handles preprocessing, feature engineering, model selection, training, evaluation, serving, and deployment, all through a clean Python API and powerful CLI.

---

## Features

| Category | Capabilities |
|---|---|
| **Core ML** | Auto preprocessing, model selection, cross-validated training, evaluation reports |
| **Intelligence** | Explainability (SHAP/feature importance), imbalance detection, data profiling |
| **Production** | FastAPI serving, ONNX export, Docker packaging, batch & real-time inference |
| **CLI** | 14 subcommands — `train`, `serve`, `predict`, `profile`, `doctor`, and more |
| **Integrations** | WandB, MLflow, plugin SDK for custom extensions |
| **Governance** | Model cards, audit logging, experiment tracking |
| **I/O Formats** | CSV, Excel, JSON, Parquet |

---

## Installation

```bash
pip install kiteml
```

### Extras

```bash
pip install kiteml[serving]   # FastAPI model server
pip install kiteml[onnx]      # ONNX export support
pip install kiteml[wandb]     # Weights & Biases tracking
pip install kiteml[mlflow]    # MLflow experiment tracking
pip install kiteml[all]       # Everything
```

---

## Quick Start

### Python API

```python
from kiteml import train

# Classification
result = train("data.csv", target="label")
print(result.summary())
result.save_model("my_model.pkl")

# Regression
result = train("housing.csv", target="price", task_type="regression")
print(result.summary())

# Make predictions
predictions = result.predict(new_data)
```

### CLI

```bash
# Train a model
kiteml train data.csv --target label

# Train with options
kiteml train data.csv --target price --type regression --save model.pkl

# Serve a model
kiteml serve model.pkl --port 8000

# Profile your dataset
kiteml profile data.csv

# Run diagnostics
kiteml doctor
```

---

## Architecture

```
kiteml/
├── core.py              # Main train() function
├── preprocessing/       # Auto cleaning, encoding, scaling
├── models/              # Model selection & training
├── evaluation/          # Metrics & reporting
├── intelligence/        # Explainability, profiling, imbalance detection
├── serving/             # FastAPI production server
├── deployment/          # ONNX, Docker, packaging
├── monitoring/          # Drift detection & performance tracking
├── experiments/         # Experiment tracking & logging
├── plugins/             # Extensible plugin SDK
├── governance/          # Model cards & audit logging
└── cli/                 # 14-command CLI ecosystem
```

---

## Documentation

Full documentation is available at [https://kiteml.github.io/kiteml](https://kiteml.github.io/kiteml).

- [Getting Started](https://kiteml.github.io/kiteml/getting_started/)
- [Usage Guide](https://kiteml.github.io/kiteml/usage/)
- [CLI Reference](https://kiteml.github.io/kiteml/cli/)
- [API Reference](https://kiteml.github.io/kiteml/api/)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/kiteml/kiteml.git
cd kiteml
pip install -e ".[dev]"
pytest tests/
```

---

## License

KiteML is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with care by the KiteML Team</sub>
</p>
