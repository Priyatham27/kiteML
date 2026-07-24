# CLI Ecosystem Quick Start

KiteML features a 14-command terminal ecosystem for scaffolding projects, profiling datasets, training models, serving REST APIs, and monitoring production deployments.

---

## 1. Scaffold a New Project (`kiteml init`)

Create a standardized machine learning project layout:

```bash
kiteml init churn_predictor
cd churn_predictor
```

Generated Directory Structure:
```text
churn_predictor/
├── data/
├── models/
├── configs/
└── main.py
```

---

## 2. Download Sample Playgrounds (`kiteml playground`)

Download built-in benchmark datasets for testing:

```bash
kiteml playground customer_churn
```

---

## 3. Profile Dataset Quality (`kiteml profile`)

Analyze dataset quality, identify missing values, detect high cardinality, and check for target leakage prior to training:

```bash
kiteml profile data/customer_churn.csv --target Exited
```

---

## 4. Train a Model (`kiteml train`)

Train a model directly from the command line:

=== "Classification"

    ```bash
    kiteml train data/customer_churn.csv --target Exited --save models/churn.pkl
    ```

=== "Regression"

    ```bash
    kiteml train data/housing.csv --target price --type regression --save models/housing.pkl
    ```

---

## 5. Serve REST API (`kiteml serve`)

Launch a production-grade FastAPI web server to serve real-time predictions:

```bash
kiteml serve models/churn.pkl --port 8000
```

Open `http://localhost:8000/docs` in your browser to inspect interactive OpenAPI documentation!

---

## 6. Make Batch Predictions (`kiteml predict`)

Generate predictions on a new CSV file:

```bash
kiteml predict models/churn.pkl data/new_customers.csv --output predictions.csv
```

---

## 7. Run Environment Diagnostics (`kiteml doctor`)

Validate your installation, hardware drivers, and installed optional dependencies:

```bash
kiteml doctor
```

---

## Summary of All 14 CLI Subcommands

| Command | Description |
| :--- | :--- |
| `kiteml train` | Train AutoML models with cross-validation and diagnostics. |
| `kiteml serve` | Launch production FastAPI REST API server with OpenAPI docs. |
| `kiteml predict` | Generate batch predictions from trained models or `.kml` pipelines. |
| `kiteml profile` | Profile datasets for data leakage, imbalance, and missingness. |
| `kiteml doctor` | Run system diagnostics and verify dependencies. |
| `kiteml init` | Scaffold standardized ML project directory structure. |
| `kiteml playground` | Download built-in sample datasets (churn, housing, iris). |
| `kiteml dashboard` | Launch local interactive HTML performance dashboard. |
| `kiteml monitor` | Monitor production inference streams for data drift (PSI). |
| `kiteml export` | Export models to ONNX graph format or Docker containers. |
| `kiteml benchmark` | Benchmark training algorithms on custom datasets. |
| `kiteml experiment` | List and inspect experiment tracking logs. |
| `kiteml plugins` | Manage custom pipeline plugins and extensions. |
| `kiteml version` | Display KiteML ecosystem versions and build info. |
