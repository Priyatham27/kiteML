"""
pipeline.py — Unified preprocessing pipeline for KiteML.

Handles numerical imputation, categorical encoding, and feature scaling
in a single object that follows the sklearn fit/transform contract.

Design
------
* Internally builds a scikit-learn Pipeline object for each branch
  (numerical and categorical) so the entire transform graph is a first-class
  sklearn object — safe to serialize, inspect, and compose.
* fit_transform() is called on training data only — no leakage.
* transform() reuses the fitted components to process test/inference data.
* feature_names is populated after fit so callers always know the output
  column order (essential for feature importance mapping).

sklearn Pipeline benefits
--------------------------
* Safer deployment — one object encapsulates the whole transform chain.
* Cleaner serialization — joblib.dump(preprocessor) captures everything.
* Avoids transform mismatch — fitted params are coupled with the transform
  steps, making it impossible to accidentally apply a fresh scaler on test
  data.
* Composable — the .sklearn_pipeline property exposes the raw Pipeline for
  advanced users who want to plug it directly into sklearn GridSearchCV.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    """
    Unified, leakage-free preprocessing pipeline backed by sklearn internals.

    Applies in order:
    1. Numerical imputation  (mean strategy)
    2. Categorical imputation (most-frequent strategy)
    3. One-Hot Encoding of categoricals  (unknown categories → all zeros)
    4. StandardScaler across all features

    The transform graph is implemented as a scikit-learn ColumnTransformer
    fed into a Pipeline, giving all the benefits of the sklearn ecosystem
    (pickling, cloning, parameter inspection) for free.

    Attributes
    ----------
    num_cols : List[str]
        Numeric columns detected at fit time.
    cat_cols : List[str]
        Categorical columns detected at fit time.
    feature_names : List[str]
        Ordered list of output feature names after encoding.
        Numeric columns come first, then OHE-expanded categorical columns.
    is_fitted : bool
        True after ``fit_transform`` has been called.
    sklearn_pipeline : sklearn.pipeline.Pipeline
        The raw sklearn Pipeline backing this Preprocessor.
        Useful for advanced users (GridSearchCV, Pipeline composition, etc.).
    """

    def __init__(self) -> None:
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

        # Populated in fit_transform() once column types are known
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # 🔹 Internal builder
    # ------------------------------------------------------------------

    def _build_pipeline(self, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
        """
        Construct the sklearn Pipeline for the detected column sets.

        The pipeline has two stages:
        1. ``col_transform`` — ColumnTransformer with separate sub-pipelines
           for numerical and categorical columns.
        2. ``scaler``        — StandardScaler applied to the combined output.

        Parameters
        ----------
        num_cols : list of str
            Names of numeric columns.
        cat_cols : list of str
            Names of categorical columns.

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        transformers = []

        if num_cols:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                ]
            )
            transformers.append(("num", num_pipeline, num_cols))

        if cat_cols:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))

        col_transform = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # safely ignore any surprise columns
            verbose_feature_names_out=False,  # clean feature names without prefix
        )

        full_pipeline = Pipeline(
            [
                ("col_transform", col_transform),
                ("scaler", StandardScaler()),
            ]
        )

        return full_pipeline

    # ------------------------------------------------------------------
    # 🔹 Feature name extraction
    # ------------------------------------------------------------------

    def _extract_feature_names(self) -> List[str]:
        """
        Read the ordered output feature names from the fitted ColumnTransformer.

        Numeric features come first (in the order they appeared in the
        DataFrame), followed by OHE-expanded categorical features.

        Returns
        -------
        list of str
        """
        col_transform = self._pipeline.named_steps["col_transform"]
        names: List[str] = []

        for name, transformer, cols in col_transform.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder: OneHotEncoder = transformer.named_steps["encoder"]
                names.extend(encoder.get_feature_names_out(cols).tolist())

        return names

    # ------------------------------------------------------------------
    # 🔹 FIT + TRANSFORM  (training data only)
    # ------------------------------------------------------------------

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the pipeline on ``X`` and return the processed array.

        Must be called only on **training** data to avoid leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame (no target column).

        Returns
        -------
        np.ndarray
            Processed feature array, shape ``(n_samples, n_features_out)``.
        """
        X = X.copy()

        # ── Detect column types ──────────────────────────────────────────
        self.num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()

        # ── Build and fit the sklearn Pipeline ───────────────────────────
        self._pipeline = self._build_pipeline(self.num_cols, self.cat_cols)
        X_processed = self._pipeline.fit_transform(X)

        # ── Capture output feature names ─────────────────────────────────
        self.feature_names = self._extract_feature_names()
        self.is_fitted = True

        return X_processed

    # ------------------------------------------------------------------
    # 🔹 TRANSFORM  (test / inference data)
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted pipeline to new data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame with the same columns seen during training.

        Returns
        -------
        np.ndarray
            Processed feature array, shape ``(n_samples, n_features_out)``.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform() on training data first.")
        return self._pipeline.transform(X.copy())

    # ------------------------------------------------------------------
    # 🔹 sklearn Pipeline access
    # ------------------------------------------------------------------

    @property
    def sklearn_pipeline(self) -> Optional[Pipeline]:
        """
        The raw fitted sklearn Pipeline backing this Preprocessor.

        Useful for advanced workflows such as GridSearchCV composition or
        direct inspection of individual transformer parameters.

        Returns ``None`` before ``fit_transform`` has been called.
        """
        return self._pipeline

    # ------------------------------------------------------------------
    # 🔹 Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"<Preprocessor [{status}] | "
            f"num={len(self.num_cols)} | cat={len(self.cat_cols)} | "
            f"features_out={len(self.feature_names)}>"
        )
