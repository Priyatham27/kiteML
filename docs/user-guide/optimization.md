# Hyperparameter Optimization

KiteML includes hyperparameter search strategies to fine-tune candidate models.

---

## 1. Search Strategies

- **Randomized Search**: Samples hyperparameter distributions efficiently.
- **Optuna Integration**: Uses Bayesian optimization algorithms to maximize metric performance.

```python
from kiteml.optimization import HyperparameterTuner

tuner = HyperparameterTuner(strategy="optuna", n_trials=50)
best_params = tuner.tune(model, X, y)
```
