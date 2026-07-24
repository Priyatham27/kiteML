# Frequently Asked Questions (FAQ)

### 1. What Python versions are supported?
KiteML requires Python **3.10** or higher (officially supporting Python `3.10`, `3.11`, `3.12`, and `3.13`).

### 2. Why is the PyPI package name `kiteml-ai` while import is `import kiteml`?
The distribution package on PyPI is named `kiteml-ai` to distinguish it on PyPI registry, while the Python code namespace remains `import kiteml` for developer convenience (identical to `scikit-learn` -> `import sklearn`).

### 3. How does KiteML prevent data leakage?
KiteML's `LeakageDetector` scans dataset features against target variables to identify near 1.0 mutual information scores and target correlation proxies prior to splitting. Furthermore, transformation parameters are fit exclusively on training folds during cross-validation.

### 4. What is a `.kml` package?
A `.kml` package is a serialized binary archive containing transformation DAG states, feature encoders, scalers, and model weights, secured by a SHA-256 integrity checksum.

### 5. How do I serve a model as a REST API?
Run `kiteml serve model.pkl --port 8000` from your terminal to launch a FastAPI REST server with interactive OpenAPI documentation.
