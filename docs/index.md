# 🪁 KiteML

**Train production-grade ML models with a single line of code — intelligent AutoML for everyone.**

KiteML is a full-stack, end-to-end framework that takes raw data and turns it into a deployed, monitored production model across 5 completed architecture epics.

---

## Why KiteML?

- **Intelligent by Default**: Built-in data profiling, automated data leakage detection, and imbalance analysis.
- **Intelligent ML Pipeline**: DAG transformation engine, automated feature engineering, voting feature selection, and `.kml` binary package serialization with SHA-256 integrity checksums.
- **Production-Ready**: Serve models via FastAPI REST APIs (`kiteml serve`), ONNX graphs, or Docker containers with live drift monitoring.
- **Developer First**: World-class 14-command CLI, structured exceptions (`KML-XXX`), warning policies (`KML-W-XXX`), and context-aware suggestions engine.

---

## Quick Start

```bash
pip install kiteml-ai

kiteml init my_project
cd my_project

# Auto-train and generate model
kiteml train data.csv --target churn --save model.pkl

# Serve immediately via REST API
kiteml serve model.pkl --port 8000
```

---

## Community

KiteML is open source and community-driven. Read our [Contribution Guide](community/contributing.md) to get involved!
