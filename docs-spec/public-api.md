# KiteML Public API Specification (`docs-spec/public-api.md`)

## 1. Package Exports Overview (`import kiteml`)

All primary user interactions with KiteML occur through top-level function exports and primary orchestrator classes defined in `kiteml/__init__.py`.

```python
from kiteml import (
    # Core High-Level API
    train,
    load,
    validate,
    # Primary Result Classes
    Result,
    TrainingResult,
    ClassificationMetrics,
    RegressionMetrics,
    TrainingTimes,
    # Epic 4 Pipeline Orchestration
    KiteMLPipeline,
    PipelineBuildResult,
    # Constants
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CV_FOLDS,
)
```

---

## 2. Top-Level Functions

### 2.1 `train()`
Main AutoML entrypoint that automates preprocessing, feature selection, model training, evaluation, and report generation.

```python
def train(
    data: str | pd.DataFrame,
    target: str,
    problem_type: str | None = None,
    test_size: float = 0.2,
    scale: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Result:
```

#### Parameters:
- `data`: Path to a CSV/JSON/Parquet/Excel file OR a pandas DataFrame instance.
- `target`: Target column name to predict.
- `problem_type`: `'classification'`, `'regression'`, or `None` (auto-detected).
- `test_size`: Fraction of dataset reserved for validation split (default `0.2`).
- `scale`: Whether to apply standard feature scaling (default `True`).
- `random_state`: Integer seed for reproducible random splits (default `42`).
- `verbose`: Whether to render CLI progress output and diagnostics (default `True`).

#### Returns:
- `Result`: Container object holding best trained model, metrics, diagnostic report, and summary helper methods.

---

### 2.2 `load()`
Loads a saved model artifact or `.kml` pipeline package from disk.

```python
def load(filepath: str | Path) -> Result | KiteMLPipeline:
```

#### Parameters:
- `filepath`: Path to `.pkl`, `.joblib`, or `.kml` binary package.

#### Returns:
- `Result` or `KiteMLPipeline` instance restored to memory.

---

### 2.3 `validate()`
Runs Epic 2 validation checks against a dataset before model training.

```python
def validate(
    data: str | pd.DataFrame,
    target: str | None = None,
    schema: dict | None = None,
) -> ValidationReport:
```

---

## 3. Top-Level Classes

### 3.1 `KiteMLPipeline`
Unified Epic 4 pipeline orchestrator that coordinates automated preprocessing, feature engineering, voting feature selection, and DAG transformation execution.

```python
class KiteMLPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None: ...
    def fit(self, df: pd.DataFrame, target: str) -> KiteMLPipeline: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame: ...
    def build(self, df: pd.DataFrame, target: str) -> PipelineBuildResult: ...
    def save(self, filepath: str | Path) -> None: ...
    @classmethod
    def load(cls, filepath: str | Path) -> KiteMLPipeline: ...
```

---

### 3.2 `Result`
Holds output from `train()`.

```python
class Result:
    model: Any                      # Best trained scikit-learn/LightGBM model
    metrics: dict[str, float]       # Dictionary of evaluation metrics
    report: str                     # Text report of model evaluation
    problem_type: str               # 'classification' or 'regression'
    all_results: list[dict]         # CV scores for all evaluated models
    
    def summary(self) -> str: ...
    def diagnostics(self) -> str: ...
    def predict(self, new_data: pd.DataFrame) -> np.ndarray: ...
    def save_model(self, filepath: str) -> None: ...
    @classmethod
    def load_model(cls, filepath: str) -> Result: ...
```

---

## 4. Package Constants

- `DEFAULT_TEST_SIZE: float = 0.2`
- `DEFAULT_RANDOM_STATE: int = 42`
- `DEFAULT_CV_FOLDS: int = 5`
- `__version__: str = "1.0.2"`
