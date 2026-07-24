# Installation Guide

Welcome to KiteML! This guide provides complete instructions for installing KiteML, setting up virtual environments, and managing optional dependencies.

---

## 📋 Prerequisites

KiteML requires Python **3.10** or higher.

| Requirement | Minimum Version | Recommended |
| :--- | :--- | :--- |
| **Python** | `>= 3.10` | `3.10`, `3.11`, `3.12`, `3.13` |
| **pip** | `>= 23.0` | Latest |
| **OS** | Windows / macOS / Linux | OS Independent |

---

## 📦 PyPI Package Installation

Install the official package from PyPI:

```bash
pip install kiteml-ai
```

!!! note "Import Namespace"
    The PyPI package name is `kiteml-ai`. In Python code, import the package as `import kiteml`:

    ```python
    import kiteml
    from kiteml import train, KiteMLPipeline
    ```

---

## 🧩 Optional Dependency Bundles

KiteML provides optional dependency extras for serving, ONNX, and experiment tracking:

=== "FastAPI Serving"
    ```bash
    pip install kiteml-ai[serving]
    ```

=== "ONNX Export"
    ```bash
    pip install kiteml-ai[onnx]
    ```

=== "Weights & Biases"
    ```bash
    pip install kiteml-ai[wandb]
    ```

=== "MLflow"
    ```bash
    pip install kiteml-ai[mlflow]
    ```

=== "All Extras"
    ```bash
    pip install kiteml-ai[all]
    ```

---

## 🔍 Verification

Verify your environment by running:

```bash
kiteml doctor
```
