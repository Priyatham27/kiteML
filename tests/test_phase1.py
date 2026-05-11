"""
test_phase1.py — End-to-end and unit tests for KiteML Phase 1 Core ML Engine.

Coverage
--------
- Model registry (classification + regression)
- Model selector (structured returns, best-model selection)
- Training engine (trainer.train_model)
- Evaluation engine (metrics for both problem types)
- Report generator (classification + regression formats)
- Result object (summary, ranking, leaderboard, predict, save/load)
- Full pipeline via kiteml.train() (classification + regression)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_data():
    """Small binary classification dataset as a DataFrame."""
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


@pytest.fixture
def reg_data():
    """Small regression dataset as a DataFrame."""
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df


@pytest.fixture
def clf_arrays():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture
def reg_arrays():
    X, y = make_regression(n_samples=300, n_features=4, noise=0.1, random_state=0)
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.2, random_state=0)


# ===========================================================================
# 1. Model Registry
# ===========================================================================


class TestModelRegistry:

    def test_classification_models_returns_dict(self):
        from kiteml.models.registry import get_classification_models

        models = get_classification_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_regression_models_returns_dict(self):
        from kiteml.models.registry import get_regression_models

        models = get_regression_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_classification_required_models_present(self):
        from kiteml.models.registry import get_classification_models

        models = get_classification_models()
        for name in ("LogisticRegression", "RandomForest", "DecisionTree"):
            assert name in models, f"{name} missing from classification registry"

    def test_regression_required_models_present(self):
        from kiteml.models.registry import get_regression_models

        models = get_regression_models()
        for name in ("LinearRegression", "RandomForest", "DecisionTree"):
            assert name in models, f"{name} missing from regression registry"

    def test_get_clf_models_returns_fresh_copies(self):
        """Two calls should return independent objects (no shared state)."""
        from kiteml.models.registry import get_classification_models

        m1 = get_classification_models()
        m2 = get_classification_models()
        assert m1 is not m2
        for k in m1:
            assert m1[k] is not m2[k]

    def test_global_registry_extensible(self):
        """Adding a model to CLASSIFICATION_MODELS is reflected in the dict."""
        from sklearn.naive_bayes import GaussianNB

        from kiteml.models import registry

        registry.CLASSIFICATION_MODELS["_TestNB"] = GaussianNB()
        assert "_TestNB" in registry.CLASSIFICATION_MODELS
        # Clean up
        del registry.CLASSIFICATION_MODELS["_TestNB"]


# ===========================================================================
# 2. Model Selector
# ===========================================================================


class TestModelSelector:

    def test_classification_returns_tuple(self, clf_arrays):
        from kiteml.models.selector import select_best_model

        X_train, X_test, y_train, y_test = clf_arrays
        result = select_best_model(X_train, y_train, problem_type="classification", cv=3)
        assert isinstance(result, tuple) and len(result) == 2

    def test_regression_returns_tuple(self, reg_arrays):
        from kiteml.models.selector import select_best_model

        X_train, X_test, y_train, y_test = reg_arrays
        result = select_best_model(X_train, y_train, problem_type="regression", cv=3)
        assert isinstance(result, tuple) and len(result) == 2

    def test_all_results_structured_format(self, clf_arrays):
        """all_results must be {name: {"score", "rank", "error"}} dicts."""
        from kiteml.models.selector import select_best_model

        X_train, _, y_train, _ = clf_arrays
        _, all_results = select_best_model(X_train, y_train, problem_type="classification", cv=3)
        for name, info in all_results.items():
            assert isinstance(info, dict), f"{name} result is not a dict"
            assert "score" in info
            assert "rank" in info
            assert "error" in info

    def test_winner_has_rank_1(self, clf_arrays):
        from kiteml.models.selector import select_best_model

        X_train, _, y_train, _ = clf_arrays
        best_model, all_results = select_best_model(X_train, y_train, problem_type="classification", cv=3)
        type(best_model).__name__
        # Map display name to class name (registry may use short names)
        rank_1_models = [name for name, info in all_results.items() if info.get("rank") == 1]
        assert len(rank_1_models) == 1

    def test_scores_are_floats(self, clf_arrays):
        from kiteml.models.selector import select_best_model

        X_train, _, y_train, _ = clf_arrays
        _, all_results = select_best_model(X_train, y_train, problem_type="classification", cv=3)
        for name, info in all_results.items():
            if info["error"] is None:
                assert isinstance(info["score"], float), f"{name} score is not float"

    def test_invalid_problem_type_raises(self, clf_arrays):
        from kiteml.models.selector import select_best_model

        X_train, _, y_train, _ = clf_arrays
        with pytest.raises(ValueError, match="Unknown problem_type"):
            select_best_model(X_train, y_train, problem_type="clustering")


# ===========================================================================
# 3. Training Engine
# ===========================================================================


class TestTrainer:

    def test_train_model_returns_fitted_model(self, clf_arrays):
        from kiteml.training.trainer import train_model

        X_train, _, y_train, _ = clf_arrays
        model = LogisticRegression(max_iter=1000)
        fitted, t = train_model(model, X_train, y_train)
        assert fitted is model  # same object
        assert hasattr(fitted, "coef_")  # sklearn fitted attr
        assert isinstance(t, float) and t > 0

    def test_train_model_regression(self, reg_arrays):
        from kiteml.training.trainer import train_model

        X_train, _, y_train, _ = reg_arrays
        model = LinearRegression()
        fitted, t = train_model(model, X_train, y_train)
        assert hasattr(fitted, "coef_")

    def test_train_model_can_predict_after(self, clf_arrays):
        from kiteml.training.trainer import train_model

        X_train, X_test, y_train, _ = clf_arrays
        model = LogisticRegression(max_iter=1000)
        fitted, _ = train_model(model, X_train, y_train)
        preds = fitted.predict(X_test)
        assert len(preds) == len(X_test)


# ===========================================================================
# 4. Evaluation Engine
# ===========================================================================


class TestMetrics:

    def test_classification_metric_keys(self, clf_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = clf_arrays
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type="classification")
        for key in ("accuracy", "precision", "recall", "f1_score", "confusion_matrix", "classification_report"):
            assert key in metrics, f"Missing key: {key}"

    def test_regression_metric_keys(self, reg_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = reg_arrays
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type="regression")
        for key in ("r2_score", "mse", "rmse", "mae"):
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_in_valid_range(self, clf_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = clf_arrays
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type="classification")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_rmse_nonnegative(self, reg_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = reg_arrays
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type="regression")
        assert metrics["rmse"] >= 0.0

    def test_rmse_equals_sqrt_mse(self, reg_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = reg_arrays
        model = LinearRegression().fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type="regression")
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-9

    def test_invalid_problem_type_raises(self, clf_arrays):
        from kiteml.evaluation.metrics import evaluate_model

        X_train, X_test, y_train, y_test = clf_arrays
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        with pytest.raises(ValueError):
            evaluate_model(model, X_test, y_test, problem_type="clustering")


# ===========================================================================
# 5. Report Generator
# ===========================================================================


class TestReport:

    def test_classification_report_contains_branding(self):
        from kiteml.evaluation.report import generate_report

        metrics = {
            "accuracy": 0.91,
            "precision": 0.89,
            "recall": 0.92,
            "f1_score": 0.90,
            "confusion_matrix": [[45, 5], [3, 47]],
            "classification_report": "              precision    recall  f1-score",
        }
        report = generate_report(metrics, problem_type="classification", model_name="RandomForest")
        assert "KiteML" in report
        assert "RandomForest" in report
        assert "Accuracy" in report
        assert "F1" in report

    def test_regression_report_contains_key_metrics(self):
        from kiteml.evaluation.report import generate_report

        metrics = {"r2_score": 0.88, "mse": 4.2, "rmse": 2.05, "mae": 1.8}
        report = generate_report(metrics, problem_type="regression", model_name="LinearRegression")
        assert "R²" in report or "R2" in report or "r2" in report.lower()
        assert "RMSE" in report

    def test_report_with_leaderboard(self):
        from kiteml.evaluation.report import generate_report

        metrics = {
            "accuracy": 0.91,
            "precision": 0.89,
            "recall": 0.92,
            "f1_score": 0.90,
            "confusion_matrix": [[45, 5], [3, 47]],
            "classification_report": "",
        }
        all_results = {
            "RandomForest": {"score": 0.91, "rank": 1, "error": None},
            "LogisticRegression": {"score": 0.84, "rank": 2, "error": None},
        }
        report = generate_report(
            metrics, problem_type="classification", model_name="RandomForest", all_results=all_results
        )
        assert "Leaderboard" in report
        assert "#1" in report
        assert "#2" in report

    def test_report_returns_string(self):
        from kiteml.evaluation.report import generate_report

        metrics = {"r2_score": 0.9, "mse": 1.0, "rmse": 1.0, "mae": 0.8}
        report = generate_report(metrics, problem_type="regression")
        assert isinstance(report, str)


# ===========================================================================
# 6. Full Pipeline — kiteml.train()
# ===========================================================================


class TestFullPipeline:

    def test_train_classification(self, clf_data, tmp_path):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result is not None
        assert result.problem_type == "classification"
        assert result.model is not None
        assert result.accuracy is not None
        assert 0.0 <= result.accuracy <= 1.0

    def test_train_regression(self, reg_data, tmp_path):
        import kiteml

        result = kiteml.train(reg_data, target="target", problem_type="regression")
        assert result is not None
        assert result.problem_type == "regression"
        assert result.rmse is not None
        assert result.rmse >= 0.0

    def test_result_all_results_structured(self, clf_data):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        for _name, info in result.all_results.items():
            assert isinstance(info, dict)
            assert "score" in info
            assert "rank" in info

    def test_result_summary_runs(self, clf_data, capsys):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        result.summary()
        captured = capsys.readouterr()
        assert "KiteML" in captured.out

    def test_result_ranking_runs(self, clf_data, capsys):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        result.ranking()
        captured = capsys.readouterr()
        assert "#1" in captured.out

    def test_result_leaderboard_is_dataframe(self, clf_data):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        lb = result.leaderboard()
        assert lb is not None
        assert isinstance(lb, pd.DataFrame)
        assert "Rank" in lb.columns
        assert "Model" in lb.columns
        assert "CV Score" in lb.columns

    def test_result_predict_works(self, clf_data):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        X_new = clf_data.drop(columns=["target"]).head(10)
        preds = result.predict(X_new)
        assert len(preds) == 10

    def test_result_report_contains_branding(self, clf_data):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        report = result.report()
        assert "KiteML" in report

    def test_result_save_load(self, clf_data, tmp_path):
        import kiteml

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        path = str(tmp_path / "model.pkl")
        result.save(path)
        bundle = kiteml.output.result.Result.load(path)
        assert "model" in bundle
        assert "preprocessor" in bundle

    def test_invalid_target_raises(self, clf_data):
        import kiteml

        with pytest.raises(ValueError, match="not found"):
            kiteml.train(clf_data, target="nonexistent_column")

    def test_auto_detects_problem_type(self, clf_data):
        import kiteml

        result = kiteml.train(clf_data, target="target")
        assert result.problem_type in ("classification", "regression")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
