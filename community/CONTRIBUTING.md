# Contributing to KiteML

We'd love your help to make KiteML better! Please follow these guidelines.

## Development Setup

1. Fork and clone the repository.
2. Install in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```
3. Run the test suite:
```bash
pytest tests/
```

## Submitting Pull Requests

- Keep PRs focused and single-purpose.
- Ensure all tests pass.
- Add typing (`mypy --strict`) and format with Black.
- Update documentation if you are adding new features.

## Code of Conduct

We expect all contributors to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).
