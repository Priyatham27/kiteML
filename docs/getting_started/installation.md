# Installation Guide

This guide covers system prerequisites, package installation options, optional extra dependencies, and virtual environment setup for **KiteML**.

---

## 📋 System Requirements

KiteML is compatible with Windows, macOS, and Linux operating systems.

| Prerequisite | Minimum Version | Supported / Recommended |
| :--- | :--- | :--- |
| **Python** | `>= 3.10` | Python `3.10`, `3.11`, `3.12`, `3.13` |
| **pip** | `>= 23.0` | Latest `pip` version |
| **RAM** | `4 GB` | `8 GB+` recommended for large datasets |

!!! note "Python Version Requirement"
    KiteML strictly requires **Python 3.10 or higher**. If you are using Python 3.8 or 3.9, please upgrade your Python runtime environment prior to installation.

---

## 📦 PyPI Package Installation

Install the core KiteML package from PyPI using `pip`:

```bash
pip install kiteml-ai
```

!!! important "PyPI Package vs Python Import"
    The official distribution package on PyPI is named **`kiteml-ai`**. However, the Python import in your code remains **`import kiteml`** (similar to `scikit-learn` -> `import sklearn`).

    ```python
    import kiteml
    from kiteml import train, KiteMLPipeline
    ```

---

## 🧩 Optional Extra Dependencies

KiteML offers modular extras so you only install dependencies needed for your specific deployment target:

=== "FastAPI Serving"

    Install dependencies required to run the production REST API server (`kiteml serve`):
    ```bash
    pip install kiteml-ai[serving]
    ```
    *Includes: `fastapi`, `uvicorn[standard]`, `pydantic`*

=== "ONNX Export"

    Install dependencies to export models to optimized ONNX graph representations:
    ```bash
    pip install kiteml-ai[onnx]
    ```
    *Includes: `skl2onnx`, `onnxruntime`*

=== "WandB Tracking"

    Install Weights & Biases integration adapter:
    ```bash
    pip install kiteml-ai[wandb]
    ```
    *Includes: `wandb`*

=== "MLflow Tracking"

    Install MLflow experiment tracking adapter:
    ```bash
    pip install kiteml-ai[mlflow]
    ```
    *Includes: `mlflow`*

=== "Complete Ecosystem"

    Install all optional dependencies at once:
    ```bash
    pip install kiteml-ai[all]
    ```

---

## 🛠️ Setting Up a Virtual Environment

It is best practice to install KiteML inside an isolated Python virtual environment:

=== "Linux / macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install kiteml-ai[all]
    ```

=== "Windows PowerShell"

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install kiteml-ai[all]
    ```

---

## 🔍 Verifying Installation

Verify that KiteML is installed correctly by running the `kiteml doctor` command in your terminal:

```bash
kiteml doctor
```

Expected Output:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🪁 KiteML Environment Doctor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python Version      3.10.12 (OK)
  KiteML Version      1.0.2 (OK)
  Core Dependencies   pandas (OK), scikit-learn (OK), rich (OK)
  Serving Stack       FastAPI (OK), Uvicorn (OK)
  ONNX Runtime        Available (OK)
  Status              Environment healthy and ready!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
