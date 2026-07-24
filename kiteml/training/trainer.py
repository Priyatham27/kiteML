"""
trainer.py — ModelTrainer for cross-validation fitting and model evaluation in KiteML.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, r2_score

from kiteml.training.cross_validation import CrossValidationEngine


class ModelTrainer:
    """
    Executes cross-validated model training and baseline fitting.
    """

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str,
        n_splits: int = 5,
        random_state: int = 42,
        model: Any = None,
    ) -> tuple[Any, list[float]]:
        """
        Train baseline model across CV folds.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        task_type : str
            ML task type.
        n_splits : int
            Number of folds.
        random_state : int
            Random seed.
        model : Any, optional
            Scikit-learn estimator instance.

        Returns
        -------
        tuple[Any, list[float]]
            Fitted baseline model and list of CV fold scores.
        """
        if model is None:
            if "classification" in task_type:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(n_estimators=50, random_state=random_state)
            else:
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(n_estimators=50, random_state=random_state)

        cv = CrossValidationEngine().get_cv(task_type=task_type, n_splits=n_splits, random_state=random_state)
        scores: list[float] = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)

            if "classification" in task_type:
                if not pd.api.types.is_integer_dtype(preds) and not pd.api.types.is_bool_dtype(preds):
                    preds = np.round(preds).astype(int)
                score = float(accuracy_score(y_val, preds))
            else:
                score = float(r2_score(y_val, preds))
            scores.append(score)

        # Fit final model on full training set
        model.fit(X_train, y_train)
        return model, scores


def train_model(model: Any, X_train: Any, y_train: Any) -> tuple[Any, float]:
    """
    Train a single model instance on X_train and y_train and return fitted model with elapsed time.

    Parameters
    ----------
    model : Any
        Estimator instance.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    tuple[Any, float]
        Fitted model and training time in seconds.
    """
    import time

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    return model, elapsed
