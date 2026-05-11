# KiteML Usage Guide

## Installation

```bash
pip install kiteml
```

Or install from source:

```bash
git clone https://github.com/yourname/kiteml.git
cd kiteml
pip install -e .
```

## Quick Start

```python
from kiteml import train

# Train on a CSV file
result = train("data.csv", target="label")

# Or pass a DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
result = train(df, target="label")
```

## Parameters

| Parameter      | Type              | Default          | Description                              |
|----------------|-------------------|------------------|------------------------------------------|
| `data`         | `str / DataFrame` | —                | Path to CSV or a pandas DataFrame        |
| `target`       | `str`             | —                | Name of the target column                |
| `problem_type` | `str / None`      | `None` (auto)    | `'classification'` or `'regression'`     |
| `test_size`    | `float`           | `0.2`            | Fraction of data for testing             |
| `scale`        | `bool`            | `True`           | Whether to apply feature scaling         |
| `random_state` | `int`             | `42`             | Random seed for reproducibility          |
| `verbose`      | `bool`            | `True`           | Print progress messages                  |

## Result Object

The `train()` function returns a `Result` object:

```python
result.model          # The trained best model
result.metrics        # Dict of evaluation metrics
result.report         # Human-readable report string
result.problem_type   # 'classification' or 'regression'
result.all_results    # CV scores for all models tried
result.summary()      # Short summary string
result.save_model()   # Save model to disk
Result.load_model()   # Load model from disk
```

## CLI Usage

```bash
kiteml train data.csv --target label
kiteml train data.csv --target price --type regression --save model.pkl
```

## Supported File Formats

- `.csv`
- `.xls` / `.xlsx`
- `.json`
- `.parquet`
