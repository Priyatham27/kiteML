# Getting Started Overview

Welcome to **KiteML**! Whether you are a data scientist looking to rapidly prototype models or an MLOps engineer building automated DAG pipelines and REST API microservices, KiteML provides intuitive tools for every stage of the machine learning lifecycle.

---

## 🧭 Navigation Guide

Choose your path to get up and running with KiteML:

<div class="grid cards" markdown>

-   :material-download: **[Installation Guide](installation.md)**

    ---

    System requirements, Python version prerequisites (3.10+), PyPI installation (`kiteml-ai`), optional extra dependencies, and troubleshooting.

-   :material-language-python: **[Python API Quick Start](quickstart_python.md)**

    ---

    Build your first AutoML model in 3 lines of Python using `kiteml.train()`, explore `Result` metrics, and run unified `KiteMLPipeline` DAG workflows.

-   :material-console: **[CLI Ecosystem Quick Start](quickstart_cli.md)**

    ---

    Scaffold projects with `kiteml init`, profile data quality with `kiteml profile`, train models, and serve production REST APIs with `kiteml serve`.

-   :material-lightbulb-on: **[Core Concepts](concepts.md)**

    ---

    Understand KiteML's 5 architecture epics, DAG pipeline execution, `.kml` binary packages, and developer experience diagnostic engine.

</div>

---

## ⚡ 60-Second Quick Start Example

```python
from kiteml import train

# 1. Train an optimal model with automated preprocessing & model selection
result = train("dataset.csv", target="target_column")

# 2. View execution diagnostics and evaluation metrics
print(result.summary())
print(result.diagnostics())

# 3. Make predictions on unseen data
predictions = result.predict(new_dataframe)

# 4. Save model artifact for deployment
result.save_model("my_model.pkl")
```
