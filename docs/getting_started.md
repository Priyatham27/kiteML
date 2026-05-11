# Getting Started with KiteML

Welcome to KiteML! This guide will take you from installation to your first production-ready deployment.

## Installation

KiteML requires Python 3.8+.

```bash
pip install kiteml
```

For all optional features (serving, ONNX export, advanced models):
```bash
pip install kiteml[all]
```

## Your First Project

Let's use the CLI to scaffold a new project:

```bash
kiteml init churn_predictor
cd churn_predictor
```

Now, download a sample dataset:
```bash
kiteml playground customer_churn
```

Analyze the data quality:
```bash
kiteml profile data/churn.csv --target Exited
```

Train a model and export it as a deployable bundle:
```bash
kiteml train data/churn.csv --target Exited --export model.kiteml
```

Serve the model locally:
```bash
kiteml serve model.kiteml
```

Visit `http://localhost:8000/docs` to test your new REST API!
