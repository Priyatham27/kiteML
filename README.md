# KiteML 🪁

**Train ML models with a single line of code.**

KiteML is a lightweight AutoML library that automates data cleaning, feature encoding, model selection, training, and evaluation — all in one function call.

## ✨ Features

- 🧹 **Auto preprocessing** — handles nulls, encodes categoricals, scales features
- 🤖 **Auto model selection** — tries multiple models and picks the best via cross-validation
- 📊 **Evaluation reports** — accuracy, F1, R², RMSE, confusion matrix, and more
- 💾 **Save & load models** — export trained models with one call
- 🖥️ **CLI support** — train from the command line
- 📁 **Multi-format** — CSV, Excel, JSON, Parquet

## 🚀 Quick Start

```python
from kiteml import train

result = train("data.csv", target="label")
print(result.summary())
result.save_model("my_model.pkl")
```

## 📦 Installation

```bash
pip install kiteml
```

## 🖥️ CLI

```bash
kiteml train data.csv --target label
kiteml train data.csv --target price --type regression --save model.pkl
```

## 📖 Documentation

See [docs/usage.md](docs/usage.md) for full documentation.

## 📜 License

MIT License
