# Model Selection Engine

The `selection` module runs a tournament across candidate algorithms to select the model with the highest cross-validation score.

---

## 1. Selection Mechanics

```python
from kiteml.selection import ModelSelector

selector = ModelSelector(metric="f1")
best_model = selector.evaluate_candidates(candidates, X_train, y_train)
```
