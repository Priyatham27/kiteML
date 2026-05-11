# 🪁 KiteML

**The Intelligent Machine Learning Ecosystem.**

KiteML is a full-stack, end-to-end framework that takes raw data and turns it into a deployed, monitored production model.

## Why KiteML?

- **Intelligent by Default**: Built-in data profiling, leakage detection, and intelligent handling of imbalanced datasets.
- **Production-Ready**: Export models directly to REST APIs, Docker containers, or ONNX format.
- **Developer First**: World-class CLI, typed APIs, and an extensible plugin system.
- **Ecosystem Scale**: Built-in MLOps tracking, lineage, versioning, and community integrations (MLflow, W&B, Airflow).

## Quick Start

```bash
pip install kiteml

kiteml init my_project
cd my_project

# Auto-train and generate a dashboard
kiteml train data.csv --target churn --dashboard

# Serve immediately via REST API
kiteml serve model.kiteml
```

## Community

KiteML is open source and community-driven. Read our [Contribution Guide](community/CONTRIBUTING.md) to get involved!
