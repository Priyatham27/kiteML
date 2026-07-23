# KiteML Context-Aware Suggestions Guide

The Suggestion Engine analyzes execution context and provides ranked, explainable recommendations.

## Explainable Suggestions

Every suggestion object includes:
- `title`: Short recommendation summary.
- `description`: Detailed explanation.
- `confidence`: Confidence score (0.0 to 1.0).
- `category`: Domain category (`Column`, `Target`, `Schema`, etc.).
- `action`: Concrete code snippet.
- `why`: List of explainability bullet points detailing *why* the suggestion was generated.

## Providers
- `ColumnSuggestionProvider`: Matches typos in column names (`prcie` -> `Price`).
- `TargetSuggestionProvider`: Recommends optimal targets for classification or regression.
- `SchemaSuggestionProvider`: Suggests removing constant or identifier features.
- `ValidationSuggestionProvider`: Suggests missing value imputation strategies.
- `TrainingSuggestionProvider`: Recommends model choices based on sample size.
- `DeploymentSuggestionProvider`: Recommends ONNX export.
- `PerformanceSuggestionProvider`: Recommends multiprocessing for large datasets.

## Public API Usage

```python
# From an exception
try:
    kiteml.train(df, target="prcie")
except KiteMLError as error:
    print(error.suggestions())

# From a training result
result = kiteml.train(df, target="price")
print(result.suggestions())
```
