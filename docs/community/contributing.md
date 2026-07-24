# Contributing to KiteML

Thank you for your interest in contributing to **KiteML**! We welcome bug reports, feature requests, documentation improvements, and pull requests from developers of all skill levels.

---

## 🚀 Quick Setup for Developers

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kiteML.git
   cd kiteML
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Editable Package & Dev Extras**:
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Run Unit Tests**:
   ```bash
   pytest tests/
   ```

---

## 🧪 Code Quality & Style Guidelines

- **Linting & Formatting**: We use `ruff` for linting and `black` for formatting.
  ```bash
  ruff check kiteml/
  black --check kiteml/
  ```
- **Type Checking**: We use `mypy` for static type checking.
  ```bash
  mypy kiteml/
  ```
- **Docstring Style**: Write all docstrings in **Google Python Style**.

---

## 📝 Pull Request Checklist

Before submitting your PR, ensure:
- [x] All pytest tests pass (`pytest tests/`).
- [x] Code passes linting (`ruff check`) and formatting (`black`).
- [x] New features or bug fixes include corresponding unit tests in `tests/`.
- [x] Documentation is updated if public APIs or CLI commands changed.
