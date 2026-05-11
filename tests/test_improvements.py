"""
test_improvements.py — Tests for the 5 Phase 1 improvements.

Covers:
1. Config centralization (DEFAULT_RANDOM_STATE, etc.)
2. sklearn Pipeline internals in Preprocessor
3. Typed Metrics dataclasses (ClassificationMetrics, RegressionMetrics)
4. Time tracking (TrainingTimes with total + training breakdown)
5. predict_proba() fallback handling
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df


@pytest.fixture
def clf_arrays():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return train_test_split(X, y, test_size=0.2, random_state=0)


# ===========================================================================
# 1. Config Centralization
# ===========================================================================

class TestConfig:

    def test_config_module_exists(self):
        import kiteml.config as cfg
        assert hasattr(cfg, "DEFAULT_RANDOM_STATE")
        assert hasattr(cfg, "DEFAULT_TEST_SIZE")
        assert hasattr(cfg, "DEFAULT_CV_FOLDS")
        assert hasattr(cfg, "DEFAULT_N_JOBS")
        assert hasattr(cfg, "DEFAULT_VERBOSE")

    def test_default_random_state_is_int(self):
        from kiteml.config import DEFAULT_RANDOM_STATE
        assert isinstance(DEFAULT_RANDOM_STATE, int)

    def test_default_test_size_in_range(self):
        from kiteml.config import DEFAULT_TEST_SIZE
        assert 0 < DEFAULT_TEST_SIZE < 1

    def test_default_cv_folds_positive(self):
        from kiteml.config import DEFAULT_CV_FOLDS
        assert DEFAULT_CV_FOLDS >= 2

    def test_registry_uses_config_seed(self):
        """Registry models must use DEFAULT_RANDOM_STATE, not a bare literal."""
        from kiteml.config import DEFAULT_RANDOM_STATE
        from kiteml.models.registry import CLASSIFICATION_MODELS
        lr = CLASSIFICATION_MODELS["LogisticRegression"]
        assert lr.random_state == DEFAULT_RANDOM_STATE

    def test_selector_defaults_match_config(self):
        """select_best_model default args must equal config values."""
        import inspect
        from kiteml.models.selector import select_best_model
        from kiteml.config import DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS
        sig = inspect.signature(select_best_model)
        assert sig.parameters["random_state"].default == DEFAULT_RANDOM_STATE
        assert sig.parameters["cv"].default == DEFAULT_CV_FOLDS

    def test_core_train_defaults_match_config(self):
        """kiteml.train() default args must equal config values."""
        import inspect
        from kiteml.core import train
        from kiteml.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS
        sig = inspect.signature(train)
        assert sig.parameters["random_state"].default == DEFAULT_RANDOM_STATE
        assert sig.parameters["test_size"].default == DEFAULT_TEST_SIZE
        assert sig.parameters["cv"].default == DEFAULT_CV_FOLDS

    def test_top_level_exports(self):
        import kiteml
        assert hasattr(kiteml, "DEFAULT_RANDOM_STATE")
        assert hasattr(kiteml, "DEFAULT_TEST_SIZE")
        assert hasattr(kiteml, "DEFAULT_CV_FOLDS")


# ===========================================================================
# 2. sklearn Pipeline Internals
# ===========================================================================

class TestSklearnPipeline:

    def test_preprocessor_builds_sklearn_pipeline(self):
        from kiteml.preprocessing.pipeline import Preprocessor
        from sklearn.pipeline import Pipeline
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        p = Preprocessor()
        p.fit_transform(X)
        assert p.sklearn_pipeline is not None
        assert isinstance(p.sklearn_pipeline, Pipeline)

    def test_sklearn_pipeline_none_before_fit(self):
        from kiteml.preprocessing.pipeline import Preprocessor
        p = Preprocessor()
        assert p.sklearn_pipeline is None

    def test_pipeline_has_col_transform_step(self):
        from kiteml.preprocessing.pipeline import Preprocessor
        from sklearn.compose import ColumnTransformer
        X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        p = Preprocessor()
        p.fit_transform(X)
        assert "col_transform" in p.sklearn_pipeline.named_steps
        assert isinstance(p.sklearn_pipeline.named_steps["col_transform"], ColumnTransformer)

    def test_pipeline_has_scaler_step(self):
        from kiteml.preprocessing.pipeline import Preprocessor
        from sklearn.preprocessing import StandardScaler
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        p = Preprocessor()
        p.fit_transform(X)
        assert "scaler" in p.sklearn_pipeline.named_steps
        assert isinstance(p.sklearn_pipeline.named_steps["scaler"], StandardScaler)

    def test_pipeline_output_matches_transform(self):
        """fit_transform and transform must produce same-shape arrays."""
        from kiteml.preprocessing.pipeline import Preprocessor
        X = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"]
        })
        p = Preprocessor()
        out_fit = p.fit_transform(X)
        out_transform = p.transform(X)
        assert out_fit.shape == out_transform.shape

    def test_sklearn_pipeline_serializable(self, tmp_path):
        """The sklearn pipeline inside Preprocessor must survive joblib round-trip."""
        import joblib
        from kiteml.preprocessing.pipeline import Preprocessor
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        p = Preprocessor()
        original_out = p.fit_transform(X)
        path = str(tmp_path / "pipe.pkl")
        joblib.dump(p, path)
        p2 = joblib.load(path)
        reloaded_out = p2.transform(X)
        # Shape and values must be identical after a full joblib round-trip
        assert reloaded_out.shape == original_out.shape
        assert np.allclose(reloaded_out, original_out, atol=1e-9)


# ===========================================================================
# 3. Typed Metrics Dataclasses
# ===========================================================================

class TestTypedMetrics:

    def test_classification_metrics_dataclass(self):
        from kiteml.output.result import ClassificationMetrics
        m = ClassificationMetrics(
            accuracy=0.91, precision=0.89, recall=0.92, f1_score=0.90,
            confusion_matrix=[[45, 5], [3, 47]], classification_report="..."
        )
        assert m.accuracy == 0.91
        assert m.f1_score == 0.90

    def test_regression_metrics_dataclass(self):
        from kiteml.output.result import RegressionMetrics
        m = RegressionMetrics(r2_score=0.88, mse=4.2, rmse=2.05, mae=1.8)
        assert m.rmse == 2.05
        assert m.r2_score == 0.88

    def test_metrics_to_dict(self):
        from kiteml.output.result import ClassificationMetrics
        m = ClassificationMetrics(accuracy=0.9, precision=0.9, recall=0.9,
                                   f1_score=0.9, confusion_matrix=[], classification_report="")
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "accuracy" in d

    def test_result_has_typed_metrics(self, clf_data):
        import kiteml
        from kiteml.output.result import ClassificationMetrics
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert isinstance(result.metrics, ClassificationMetrics)

    def test_result_regression_typed_metrics(self, reg_data):
        import kiteml
        from kiteml.output.result import RegressionMetrics
        result = kiteml.train(reg_data, target="target", problem_type="regression")
        assert isinstance(result.metrics, RegressionMetrics)

    def test_result_metrics_dict_coercion(self):
        """Passing a plain dict to Result must produce the correct typed object."""
        from kiteml.output.result import Result, ClassificationMetrics
        from sklearn.linear_model import LogisticRegression
        raw = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9,
               "f1_score": 0.9, "confusion_matrix": [], "classification_report": ""}
        model = LogisticRegression()
        r = Result(model=model, metrics=raw, report="", problem_type="classification")
        assert isinstance(r.metrics, ClassificationMetrics)
        assert r.metrics.accuracy == 0.9

    def test_result_accessors_still_work(self, clf_data):
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result.accuracy == result.metrics.accuracy
        assert result.f1 == result.metrics.f1_score


# ===========================================================================
# 4. Time Tracking
# ===========================================================================

class TestTimeTracking:

    def test_training_times_dataclass(self):
        from kiteml.output.result import TrainingTimes
        t = TrainingTimes(total=12.4, training=3.1)
        assert t.total == 12.4
        assert t.training == 3.1

    def test_training_times_str(self):
        from kiteml.output.result import TrainingTimes
        t = TrainingTimes(total=12.4, training=3.1)
        s = str(t)
        assert "12.40" in s
        assert "3.10" in s

    def test_result_has_times_object(self, clf_data):
        import kiteml
        from kiteml.output.result import TrainingTimes
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert hasattr(result, "times")
        assert isinstance(result.times, TrainingTimes)

    def test_times_total_positive(self, clf_data):
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result.times.total > 0

    def test_times_training_positive(self, clf_data):
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result.times.training > 0

    def test_times_training_less_than_total(self, clf_data):
        """model.fit() time must be a subset of total wall-clock time."""
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result.times.training <= result.times.total

    def test_elapsed_time_backward_compat(self, clf_data):
        """result.elapsed_time must still equal result.times.total."""
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        assert result.elapsed_time == result.times.total

    def test_trainer_returns_tuple(self, clf_arrays):
        from kiteml.training.trainer import train_model
        X_train, _, y_train, _ = clf_arrays
        model = LogisticRegression(max_iter=1000)
        result = train_model(model, X_train, y_train)
        assert isinstance(result, tuple) and len(result) == 2
        fitted, t = result
        assert hasattr(fitted, "coef_")
        assert isinstance(t, float) and t > 0

    def test_summary_shows_times(self, clf_data, capsys):
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        result.summary()
        out = capsys.readouterr().out
        assert "Total Time" in out
        assert "Training Time" in out


# ===========================================================================
# 5. predict_proba() Fallback
# ===========================================================================

class TestPredictProbaFallback:

    def _make_result_with_model(self, model, clf_data):
        """Helper: train a result but swap in a custom model."""
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        result.model = model
        result.model_name = type(model).__name__
        return result

    def test_predict_proba_works_for_proba_model(self, clf_data):
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        # Ensure the winner supports predict_proba
        if hasattr(result.model, "predict_proba"):
            X_new = clf_data.drop(columns=["target"]).head(5)
            proba = result.predict_proba(X_new)
            assert proba.shape[0] == 5
            assert proba.shape[1] >= 2

    def test_predict_proba_raises_for_no_proba_model(self, clf_data):
        """SVC without probability=True must raise AttributeError by default."""
        import kiteml
        from kiteml.preprocessing.pipeline import Preprocessor
        from kiteml.output.result import Result

        result = kiteml.train(clf_data, target="target", problem_type="classification")
        # Replace model with bare SVC (no predict_proba)
        X = clf_data.drop(columns=["target"])
        y = clf_data["target"]
        svc = SVC(random_state=42, probability=False)  # no proba
        X_proc = result.preprocessor.transform(X)
        svc.fit(X_proc, y)
        result.model = svc
        result.model_name = "SVC"

        X_new = clf_data.drop(columns=["target"]).head(5)
        with pytest.raises(AttributeError, match="predict_proba"):
            result.predict_proba(X_new)

    def test_predict_proba_fallback_returns_array(self, clf_data):
        """With fallback_to_predict=True, a one-hot array is returned."""
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        X = clf_data.drop(columns=["target"])
        y = clf_data["target"]
        svc = SVC(random_state=42, probability=False)
        X_proc = result.preprocessor.transform(X)
        svc.fit(X_proc, y)
        result.model = svc
        result.model_name = "SVC"

        X_new = clf_data.drop(columns=["target"]).head(10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = result.predict_proba(X_new, fallback_to_predict=True)
            assert len(caught) == 1
            assert issubclass(caught[0].category, UserWarning)
            assert "fallback" in str(caught[0].message).lower() or "predict_proba" in str(caught[0].message)

        assert out.shape[0] == 10
        # One-hot: each row sums to 1
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_predict_proba_fallback_warning_message(self, clf_data):
        """Warning message must mention the model name and recommend alternatives."""
        import kiteml
        result = kiteml.train(clf_data, target="target", problem_type="classification")
        X = clf_data.drop(columns=["target"])
        y = clf_data["target"]
        svc = SVC(random_state=42, probability=False)
        X_proc = result.preprocessor.transform(X)
        svc.fit(X_proc, y)
        result.model = svc
        result.model_name = "SVC"

        X_new = clf_data.drop(columns=["target"]).head(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.predict_proba(X_new, fallback_to_predict=True)
        assert "SVC" in str(caught[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
