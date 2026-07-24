# Voting Feature Selection Engine

KiteML implements a **multi-selector voting system** (`kiteml.feature_selection`) to identify and preserve highly predictive features while eliminating uninformative or noisy columns.

---

## 1. Multi-Selector Voting Mechanics

Instead of relying on a single feature selection heuristic, KiteML aggregates four independent selector algorithms:

```mermaid
graph TD
    Features[Generated Feature Set] --> S1[Variance Threshold Selector]
    Features --> S2[Correlation Filter Selector]
    Features --> S3[Mutual Information Selector]
    Features --> S4[Model Importance Selector]

    S1 --> Voting[Multi-Selector Voting Aggregator]
    S2 --> Voting
    S3 --> Voting
    S4 --> Voting

    Voting --> Final[Selected Optimal Features]
```

### Selector Algorithms:
1. **Variance Threshold**: Eliminates constant and near-zero variance features (`variance < 0.01`).
2. **Correlation Filter**: Removes highly collinear redundant features (`|r| > 0.95`).
3. **Mutual Information**: Measures non-linear dependency with target variable.
4. **Tree Importance**: Evaluates split gains using a fast preliminary Random Forest / LightGBM model.

---

## 2. Voting Thresholds

Each feature receives a vote (0 to 4) from the selectors. Features meeting the consensus threshold (default `>= 2` votes) are retained in the final DAG pipeline.

```python
from kiteml.feature_selection import VotingFeatureSelector

selector = VotingFeatureSelector(min_votes=2, max_features=50)
selected_df = selector.fit_transform(X, y)

print(f"Features reduced from {X.shape[1]} to {selected_df.shape[1]}")
```
