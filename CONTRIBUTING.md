# Contributing to KiteML

Thank you for your interest in contributing to KiteML! 🪁

We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code contributions.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kiteml.git
   cd kiteml
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- `pip` and `virtualenv` (or `venv`)

### Install in Development Mode

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=kiteml --cov-report=html

# Run specific test markers
pytest -m smoke       # Quick smoke tests only
pytest -m "not slow"  # Skip slow tests
```

### Linting & Formatting

```bash
# Format code
black kiteml/ tests/

# Lint
ruff check kiteml/ tests/

# Type check
mypy kiteml/ --ignore-missing-imports
```

---

## Making Changes

### Branch Naming

| Type    | Format                       | Example                       |
| ------- | ---------------------------- | ----------------------------- |
| Feature | `feature/description`        | `feature/onnx-quantization`   |
| Bugfix  | `fix/description`            | `fix/null-handling-crash`     |
| Docs    | `docs/description`           | `docs/api-reference`          |
| Refactor| `refactor/description`       | `refactor/pipeline-cleanup`   |

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ONNX quantization support
fix: handle null values in categorical encoder
docs: update CLI reference for predict command
test: add integration tests for batch inference
chore: update CI workflow to Python 3.13
```

---

## Pull Request Process

1. **Ensure all tests pass** before submitting
2. **Update documentation** if you're adding or changing features
3. **Add tests** for any new functionality
4. **Keep PRs focused** — one logical change per PR
5. **Fill out the PR template** completely
6. **Request review** from at least one maintainer

### PR Checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code is formatted (`black --check kiteml/ tests/`)
- [ ] Linting passes (`ruff check kiteml/`)
- [ ] Type checks pass (`mypy kiteml/`)
- [ ] Documentation is updated (if applicable)
- [ ] Changelog entry added (for user-facing changes)

---

## Coding Standards

### Python Style

- **Formatter:** Black (line length 120)
- **Linter:** Ruff
- **Type checker:** MyPy
- **Target Python:** 3.9+

### Design Principles

- Keep functions focused and single-purpose
- Use type hints on all public APIs
- Write docstrings for all public classes and functions
- Prefer composition over inheritance
- Keep dependencies minimal

### Documentation

- All public APIs must have docstrings (Google style)
- Include usage examples in docstrings for complex functions
- Update the user-facing docs for any feature changes

---

## Reporting Issues

### Bug Reports

Please include:
- Python version and OS
- KiteML version (`kiteml --version`)
- Minimal reproducible example
- Full error traceback
- Expected vs actual behavior

### Feature Requests

Please include:
- Use case description
- Proposed API (if applicable)
- Alternatives considered

---

## 🙏 Thank You!

Every contribution, no matter how small, helps make KiteML better. We appreciate your time and effort!
