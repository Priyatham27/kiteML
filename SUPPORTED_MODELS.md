# Supported Machine Learning Models

KiteML includes a comprehensive suite of classification and regression algorithm candidate wrappers built on top of `scikit-learn`, `LightGBM`, and `XGBoost`.

---

## 📊 Classification Models

| Model Algorithm | Internal Identifier | Submodule | Default Parameters | Recommended Task |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM Classifier** | `lightgbm` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1`, `max_depth=-1` | Tabular data with high sample count |
| **XGBoost Classifier** | `xgboost` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6` | Complex non-linear feature spaces |
| **Random Forest Classifier** | `rf` / `random_forest` | `kiteml.models` | `n_estimators=100`, `max_depth=None`, `random_state=42` | Robust baseline, robust to noise |
| **Gradient Boosting Classifier** | `gb` / `gradient_boosting` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1` | General tabular classification |
| **Extra Trees Classifier** | `et` / `extra_trees` | `kiteml.models` | `n_estimators=100`, `random_state=42` | Reduced variance tree ensemble |
| **Logistic Regression** | `lr` / `logistic_regression` | `kiteml.models` | `C=1.0`, `solver='lbfgs'`, `max_iter=1000` | Linear baseline, highly interpretable |
| **HistGradientBoosting Classifier** | `hist_gb` | `kiteml.models` | `max_iter=100`, `learning_rate=0.1` | Large datasets with missing values |
| **Decision Tree Classifier** | `dt` / `decision_tree` | `kiteml.models` | `max_depth=None` | Fast interpretable baseline |
| **AdaBoost Classifier** | `adaboost` | `kiteml.models` | `n_estimators=50`, `learning_rate=1.0` | Boosting ensemble baseline |
| **Gaussian Naive Bayes** | `nb` / `naive_bayes` | `kiteml.models` | `var_smoothing=1e-9` | Fast probabilistic classification |

---

## 📈 Regression Models

| Model Algorithm | Internal Identifier | Submodule | Default Parameters | Recommended Task |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM Regressor** | `lightgbm` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1` | High-dimensional regression |
| **XGBoost Regressor** | `xgboost` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1` | Non-linear continuous target prediction |
| **Random Forest Regressor** | `rf` / `random_forest` | `kiteml.models` | `n_estimators=100`, `random_state=42` | Robust non-parametric regression |
| **Gradient Boosting Regressor** | `gb` / `gradient_boosting` | `kiteml.models` | `n_estimators=100`, `learning_rate=0.1` | Precise continuous estimation |
| **Extra Trees Regressor** | `et` / `extra_trees` | `kiteml.models` | `n_estimators=100`, `random_state=42` | Fast ensemble regressor |
| **Ridge Regression** | `ridge` | `kiteml.models` | `alpha=1.0`, `solver='auto'` | Linear L2 regularization |
| **Lasso Regression** | `lasso` | `kiteml.models` | `alpha=1.0` | Sparse feature estimation |
| **ElasticNet Regression** | `elasticnet` | `kiteml.models` | `alpha=1.0`, `l1_ratio=0.5` | Combined L1 + L2 regularization |
| **Linear Regression** | `linear` | `kiteml.models` | Ordinary Least Squares | Standard linear baseline |
| **HistGradientBoosting Regressor** | `hist_gb` | `kiteml.models` | `max_iter=100` | Large-scale continuous targets |
