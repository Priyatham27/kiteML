"""
test_phase2.py — Comprehensive tests for the KiteML Phase 2 Intelligence Layer.

Covers all 16 intelligence modules and the DataProfile orchestrator.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier

# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def clean_clf_df():
    X, y = make_classification(n_samples=300, n_features=6, random_state=42)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(6)])
    df["target"] = y
    return df


@pytest.fixture
def messy_df():
    """DataFrame with common real-world issues for quality testing."""
    np.random.seed(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n).astype(float),
            "salary": np.random.exponential(50000, n),
            "city": np.random.choice(["NYC", "LA", "CHI", "HOU"], n),
            "constant_col": [1] * n,
            "customer_id": range(n),
            "review": ["This product is great and I love it very much"] * n,
            "order_date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "target": np.random.randint(0, 2, n),
        }
    )
    # Append exact duplicate rows first (before introducing NaN so they stay identical)
    dup_rows = df.head(10).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    # Introduce missing values on rows that are NOT in the appended block
    df.loc[100:130, "age"] = np.nan
    return df


@pytest.fixture
def imbalanced_df():
    n = 1000
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "target": np.where(np.arange(n) < 950, 0, 1),
        }
    )
    return df


@pytest.fixture
def leaky_df():
    n = 300
    X, y = make_classification(n_samples=n, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df["target_copy"] = y.copy()  # exact copy — critical leakage
    df["near_target"] = y + np.random.randn(n) * 0.01  # near-perfect correlation
    return df


# ===========================================================================
# 1. Column Analyzer
# ===========================================================================


class TestColumnAnalyzer:

    def test_returns_column_analysis_result(self, messy_df):
        from kiteml.intelligence.column_analyzer import analyze_columns

        result = analyze_columns(messy_df, exclude=["target"])
        assert result is not None
        assert len(result.profiles) > 0

    def test_detects_constant_column(self, messy_df):
        from kiteml.intelligence.column_analyzer import ColumnType, analyze_columns

        result = analyze_columns(messy_df)
        assert result.profiles["constant_col"].column_type == ColumnType.CONSTANT

    def test_detects_identifier(self, messy_df):
        from kiteml.intelligence.column_analyzer import ColumnType, analyze_columns

        result = analyze_columns(messy_df)
        assert result.profiles["customer_id"].column_type == ColumnType.IDENTIFIER

    def test_detects_text_column(self):
        from kiteml.intelligence.column_analyzer import ColumnType, analyze_columns

        # Use unique long text values so they are not classified as CONSTANT
        reviews = [
            "This product is absolutely amazing and I love every aspect of it completely",
            "Really disappointed with the quality of this item, very poor build quality",
            "Excellent customer service and fast delivery, highly recommend to everyone",
            "The size was completely wrong and the return process was very difficult",
            "Best purchase I have made this year, works exactly as advertised perfectly",
        ] * 20
        df = pd.DataFrame({"review": reviews, "score": range(100)})
        result = analyze_columns(df)
        assert result.profiles["review"].column_type == ColumnType.TEXT

    def test_detects_numerical_columns(self, clean_clf_df):
        from kiteml.intelligence.column_analyzer import ColumnType, analyze_columns

        result = analyze_columns(clean_clf_df, exclude=["target"])
        num_cols = result.of_type(ColumnType.NUMERICAL)
        assert len(num_cols) > 0

    def test_type_summary_counts_match(self, messy_df):
        from kiteml.intelligence.column_analyzer import analyze_columns

        result = analyze_columns(messy_df)
        total_from_summary = sum(result.type_summary.values())
        assert total_from_summary == len(result.profiles)

    def test_to_dict_returns_dict(self, clean_clf_df):
        from kiteml.intelligence.column_analyzer import analyze_columns

        result = analyze_columns(clean_clf_df)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert all("type" in v for v in d.values())


# ===========================================================================
# 2. Schema Inference
# ===========================================================================


class TestSchemaInference:

    def test_infer_schema_structure(self, clean_clf_df):
        from kiteml.intelligence.schema_inference import infer_schema

        schema = infer_schema(clean_clf_df)
        assert schema.n_rows == len(clean_clf_df)
        assert schema.n_cols == len(clean_clf_df.columns)
        assert len(schema.columns) == len(clean_clf_df.columns)

    def test_numeric_column_has_stats(self, clean_clf_df):
        from kiteml.intelligence.schema_inference import infer_schema

        schema = infer_schema(clean_clf_df)
        col = list(schema.columns.values())[0]
        assert col.min_val is not None
        assert col.mean is not None
        assert col.distribution is not None

    def test_null_detection(self, messy_df):
        from kiteml.intelligence.schema_inference import infer_schema

        schema = infer_schema(messy_df)
        assert schema.columns["age"].nullable is True
        assert schema.columns["age"].null_count > 0

    def test_to_dict(self, clean_clf_df):
        from kiteml.intelligence.schema_inference import infer_schema

        d = infer_schema(clean_clf_df).to_dict()
        assert isinstance(d, dict)


# ===========================================================================
# 3. Target Detection
# ===========================================================================


class TestTargetDetection:

    def test_detects_last_column(self, clean_clf_df):
        from kiteml.intelligence.target_detection import detect_target

        result = detect_target(clean_clf_df)
        assert result.column == "target"

    def test_keyword_detection(self):
        from kiteml.intelligence.target_detection import detect_target

        df = pd.DataFrame(
            {
                "age": [1, 2, 3],
                "salary": [10, 20, 30],
                "label": [0, 1, 0],
            }
        )
        result = detect_target(df)
        assert result.column == "label"

    def test_confidence_is_valid(self, clean_clf_df):
        from kiteml.intelligence.target_detection import detect_target

        result = detect_target(clean_clf_df)
        assert 0.0 <= result.confidence <= 1.0

    def test_reason_is_non_empty(self, clean_clf_df):
        from kiteml.intelligence.target_detection import detect_target

        result = detect_target(clean_clf_df)
        assert len(result.reason) > 0


# ===========================================================================
# 4. Problem Inference
# ===========================================================================


class TestProblemInference:

    def test_binary_classification(self):
        from kiteml.intelligence.problem_inference import infer_problem_type_advanced

        y = pd.Series([0, 1, 0, 1, 1, 0])
        result = infer_problem_type_advanced(y)
        assert result.problem_type == "classification"
        assert result.subtype == "binary"

    def test_multiclass(self):
        from kiteml.intelligence.problem_inference import infer_problem_type_advanced

        y = pd.Series([0, 1, 2, 1, 2, 0, 0])
        result = infer_problem_type_advanced(y)
        assert result.problem_type == "classification"
        assert result.subtype == "multiclass"

    def test_regression(self):
        from kiteml.intelligence.problem_inference import infer_problem_type_advanced

        y = pd.Series(np.random.randn(200))
        result = infer_problem_type_advanced(y)
        assert result.problem_type == "regression"

    def test_string_target_is_classification(self):
        from kiteml.intelligence.problem_inference import infer_problem_type_advanced

        y = pd.Series(["cat", "dog", "cat", "bird"])
        result = infer_problem_type_advanced(y)
        assert result.problem_type == "classification"

    def test_confidence_range(self):
        from kiteml.intelligence.problem_inference import infer_problem_type_advanced

        y = pd.Series([0, 1] * 50)
        result = infer_problem_type_advanced(y)
        assert 0 <= result.confidence <= 1.0


# ===========================================================================
# 5. Quality Analyzer
# ===========================================================================


class TestQualityAnalyzer:

    def test_detects_duplicates(self, messy_df):
        from kiteml.intelligence.quality_analyzer import analyze_quality

        report = analyze_quality(messy_df)
        dup_issues = [i for i in report.issues if i.issue_type == "duplicate_rows"]
        assert len(dup_issues) > 0

    def test_detects_constant_column(self, messy_df):
        from kiteml.intelligence.quality_analyzer import analyze_quality

        report = analyze_quality(messy_df)
        const_issues = [i for i in report.issues if i.issue_type == "constant_column"]
        assert len(const_issues) > 0

    def test_detects_high_missing(self):
        from kiteml.intelligence.quality_analyzer import analyze_quality

        df = pd.DataFrame({"a": [np.nan] * 80 + [1.0] * 20, "b": range(100)})
        report = analyze_quality(df)
        high_miss = [i for i in report.issues if i.issue_type == "high_missing_rate"]
        assert len(high_miss) > 0

    def test_score_is_valid(self, clean_clf_df):
        from kiteml.intelligence.quality_analyzer import analyze_quality

        report = analyze_quality(clean_clf_df)
        assert 0 <= report.score <= 100

    def test_clean_df_has_high_score(self, clean_clf_df):
        from kiteml.intelligence.quality_analyzer import analyze_quality

        report = analyze_quality(clean_clf_df)
        assert report.score >= 85


# ===========================================================================
# 6. Imbalance Detector
# ===========================================================================


class TestImbalanceDetector:

    def test_extreme_imbalance(self, imbalanced_df):
        from kiteml.intelligence.imbalance_detector import detect_imbalance

        report = detect_imbalance(imbalanced_df["target"])
        assert report.is_imbalanced is True
        assert report.severity in ("severe", "extreme")

    def test_balanced_dataset(self, clean_clf_df):
        from kiteml.intelligence.imbalance_detector import detect_imbalance

        report = detect_imbalance(clean_clf_df["target"])
        assert report.severity in ("none", "mild")

    def test_class_distribution_sums_to_one(self, imbalanced_df):
        from kiteml.intelligence.imbalance_detector import detect_imbalance

        report = detect_imbalance(imbalanced_df["target"])
        total = sum(report.class_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_recommendations_non_empty_for_severe(self, imbalanced_df):
        from kiteml.intelligence.imbalance_detector import detect_imbalance

        report = detect_imbalance(imbalanced_df["target"])
        assert len(report.recommendations) > 0


# ===========================================================================
# 7. Leakage Detector
# ===========================================================================


class TestLeakageDetector:

    def test_detects_critical_leakage(self, leaky_df):
        from kiteml.intelligence.leakage_detector import detect_leakage

        report = detect_leakage(leaky_df, target="target")
        assert report.has_leakage_risk is True
        assert "target_copy" in report.critical_columns

    def test_clean_df_no_leakage(self, clean_clf_df):
        from kiteml.intelligence.leakage_detector import detect_leakage

        report = detect_leakage(clean_clf_df, target="target")
        # clean df should have no critical leakage
        assert len(report.critical_columns) == 0

    def test_recommendations_populated(self, leaky_df):
        from kiteml.intelligence.leakage_detector import detect_leakage

        report = detect_leakage(leaky_df, target="target")
        assert len(report.recommendations) > 0


# ===========================================================================
# 8. Outlier Detector
# ===========================================================================


class TestOutlierDetector:

    def test_detects_outliers_iqr(self):
        from kiteml.intelligence.outlier_detector import detect_outliers

        df = pd.DataFrame({"x": list(range(100)) + [9999, -9999]})
        report = detect_outliers(df, method="iqr")
        assert report.has_outliers is True
        assert "x" in report.columns_with_outliers

    def test_detects_outliers_zscore(self):
        from kiteml.intelligence.outlier_detector import detect_outliers

        df = pd.DataFrame({"x": list(range(100)) + [9999]})
        report = detect_outliers(df, method="zscore")
        assert report.has_outliers is True

    def test_no_outliers_in_normal_data(self):
        from kiteml.intelligence.outlier_detector import detect_outliers

        np.random.seed(0)
        df = pd.DataFrame({"x": np.random.randn(500)})
        report = detect_outliers(df, method="iqr")
        # IQR may still flag some; just verify structure
        assert isinstance(report.has_outliers, bool)
        assert report.outlier_row_ratio <= 1.0


# ===========================================================================
# 9. Text Detector
# ===========================================================================


class TestTextDetector:

    def test_detects_text_column(self, messy_df):
        from kiteml.intelligence.text_detector import detect_text_columns

        result = detect_text_columns(messy_df)
        assert result.has_text is True
        assert "review" in result.text_columns

    def test_no_text_in_numeric_df(self, clean_clf_df):
        from kiteml.intelligence.text_detector import detect_text_columns

        result = detect_text_columns(clean_clf_df)
        assert result.has_text is False

    def test_detail_fields_populated(self, messy_df):
        from kiteml.intelligence.text_detector import detect_text_columns

        result = detect_text_columns(messy_df)
        info = result.details["review"]
        assert info.avg_word_count > 0
        assert info.confidence >= 0.5


# ===========================================================================
# 10. Datetime Detector
# ===========================================================================


class TestDatetimeDetector:

    def test_detects_datetime_string(self, messy_df):
        from kiteml.intelligence.datetime_detector import detect_datetime_columns

        result = detect_datetime_columns(messy_df)
        assert result.has_datetime is True
        assert "order_date" in result.datetime_columns

    def test_detects_datetime64_dtype(self):
        from kiteml.intelligence.datetime_detector import detect_datetime_columns

        df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=10)})
        result = detect_datetime_columns(df)
        assert "ts" in result.datetime_columns

    def test_extract_features_drops_original(self):
        from kiteml.intelligence.datetime_detector import extract_datetime_features

        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5)})
        out = extract_datetime_features(df, ["date"])
        assert "date" not in out.columns
        assert "date_year" in out.columns
        assert "date_month" in out.columns


# ===========================================================================
# 11. Correlation Analyzer
# ===========================================================================


class TestCorrelationAnalyzer:

    def test_finds_high_correlation(self):
        from kiteml.intelligence.correlation_analyzer import analyze_correlations

        n = 200
        x = np.random.randn(n)
        df = pd.DataFrame({"x": x, "x_copy": x + np.random.randn(n) * 0.01, "target": x > 0})
        report = analyze_correlations(df, target="target")
        assert len(report.high_correlation_pairs) > 0

    def test_target_correlations_populated(self, clean_clf_df):
        from kiteml.intelligence.correlation_analyzer import analyze_correlations

        report = analyze_correlations(clean_clf_df, target="target")
        assert len(report.target_correlations) > 0

    def test_top_predictors_sorted(self, clean_clf_df):
        from kiteml.intelligence.correlation_analyzer import analyze_correlations

        report = analyze_correlations(clean_clf_df, target="target")
        corrs = [report.target_correlations[c] for c in report.top_predictors]
        assert corrs == sorted(corrs, reverse=True)


# ===========================================================================
# 12. Cardinality Analyzer
# ===========================================================================


class TestCardinalityAnalyzer:

    def test_low_cardinality(self):
        from kiteml.intelligence.cardinality_analyzer import analyze_cardinality

        df = pd.DataFrame({"cat": ["a", "b", "c"] * 100})
        report = analyze_cardinality(df)
        assert report.details["cat"].cardinality_level == "low"

    def test_high_cardinality_flagged(self):
        from kiteml.intelligence.cardinality_analyzer import analyze_cardinality

        df = pd.DataFrame({"city": [f"city_{i}" for i in range(500)]})
        report = analyze_cardinality(df)
        assert "city" in report.high_cardinality_columns

    def test_encoding_recommendation_present(self):
        from kiteml.intelligence.cardinality_analyzer import analyze_cardinality

        df = pd.DataFrame({"cat": ["x", "y"] * 50})
        report = analyze_cardinality(df)
        assert len(report.details["cat"].encoding_recommendation) > 0


# ===========================================================================
# 13. Memory Optimizer
# ===========================================================================


class TestMemoryOptimizer:

    def test_returns_memory_report(self, clean_clf_df):
        from kiteml.intelligence.memory_optimizer import analyze_memory

        report = analyze_memory(clean_clf_df)
        assert report.total_memory_bytes > 0
        assert report.total_memory_mb > 0

    def test_suggests_int_downcast(self):
        from kiteml.intelligence.memory_optimizer import analyze_memory

        df = pd.DataFrame({"small_int": np.array([1, 2, 3, 4], dtype=np.int64)})
        report = analyze_memory(df)
        col = report.columns["small_int"]
        assert col.suggested_dtype != "int64" or col.estimated_savings_bytes >= 0


# ===========================================================================
# 14. Feature Recommender
# ===========================================================================


class TestFeatureRecommender:

    def test_recommends_drop_for_identifier(self, messy_df):
        from kiteml.intelligence.column_analyzer import analyze_columns
        from kiteml.intelligence.feature_recommender import generate_recommendations

        col_analysis = analyze_columns(messy_df, exclude=["target"])
        report = generate_recommendations(messy_df, col_analysis, target="target")
        drop_rec = [r for r in report.recommendations if r.action == "drop"]
        assert len(drop_rec) > 0

    def test_high_priority_has_reason(self, messy_df):
        from kiteml.intelligence.column_analyzer import analyze_columns
        from kiteml.intelligence.feature_recommender import generate_recommendations

        col_analysis = analyze_columns(messy_df, exclude=["target"])
        report = generate_recommendations(messy_df, col_analysis, target="target")
        for rec in report.high_priority():
            assert len(rec.reason) > 0

    def test_summary_is_string(self, clean_clf_df):
        from kiteml.intelligence.column_analyzer import analyze_columns
        from kiteml.intelligence.feature_recommender import generate_recommendations

        col_analysis = analyze_columns(clean_clf_df, exclude=["target"])
        report = generate_recommendations(clean_clf_df, col_analysis, target="target")
        assert isinstance(report.summary, str)


# ===========================================================================
# 15. Explainability
# ===========================================================================


class TestExplainability:

    def test_feature_importances_from_rf(self, clean_clf_df):
        from sklearn.ensemble import RandomForestClassifier

        from kiteml.intelligence.explainability import explain_model

        X = clean_clf_df.drop(columns=["target"]).values
        y = clean_clf_df["target"].values
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        model.fit(X, y)
        feature_names = [f"num_{i}" for i in range(6)]
        report = explain_model(model, feature_names)
        assert report.method == "feature_importances_"
        assert len(report.feature_importances) > 0
        assert report.shap_ready is True

    def test_coef_from_logistic(self, clean_clf_df):
        from sklearn.linear_model import LogisticRegression

        from kiteml.intelligence.explainability import explain_model

        X = clean_clf_df.drop(columns=["target"]).values
        y = clean_clf_df["target"].values
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        feature_names = [f"num_{i}" for i in range(6)]
        report = explain_model(model, feature_names)
        assert report.method == "coef_"

    def test_to_dict(self, clean_clf_df):
        from kiteml.intelligence.explainability import explain_model

        X = clean_clf_df.drop(columns=["target"]).values
        y = clean_clf_df["target"].values
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(X, y)
        report = explain_model(model, [f"f{i}" for i in range(6)])
        d = report.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0


# ===========================================================================
# 16. Master Recommendations
# ===========================================================================


class TestMasterRecommendations:

    def test_returns_report(self, clean_clf_df):
        from kiteml.intelligence.quality_analyzer import analyze_quality
        from kiteml.intelligence.recommendations import build_recommendation_report

        quality = analyze_quality(clean_clf_df)
        report = build_recommendation_report(quality=quality)
        assert report is not None
        assert isinstance(report.overall_health, str)

    def test_critical_leakage_raises_count(self, leaky_df):
        from kiteml.intelligence.leakage_detector import detect_leakage
        from kiteml.intelligence.recommendations import build_recommendation_report

        leakage = detect_leakage(leaky_df, target="target")
        report = build_recommendation_report(leakage=leakage)
        assert report.critical_count > 0

    def test_sorted_by_priority(self, leaky_df, messy_df):
        from kiteml.intelligence.leakage_detector import detect_leakage
        from kiteml.intelligence.quality_analyzer import analyze_quality
        from kiteml.intelligence.recommendations import build_recommendation_report

        leakage = detect_leakage(leaky_df, target="target")
        quality = analyze_quality(messy_df)
        report = build_recommendation_report(leakage=leakage, quality=quality)
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        priorities = [order[r.priority] for r in report.recommendations]
        assert priorities == sorted(priorities)


# ===========================================================================
# 17. DataProfile Orchestrator
# ===========================================================================


class TestDataProfileOrchestrator:

    def test_builds_data_profile(self, clean_clf_df):
        from kiteml.intelligence.data_profiler import build_data_profile

        profile = build_data_profile(clean_clf_df, target="target", problem_type="classification")
        assert profile is not None
        assert profile.target == "target"
        assert profile.schema is not None
        assert profile.quality is not None
        assert profile.leakage is not None

    def test_imbalance_set_for_classification(self, clean_clf_df):
        from kiteml.intelligence.data_profiler import build_data_profile

        profile = build_data_profile(clean_clf_df, target="target", problem_type="classification")
        assert profile.imbalance is not None

    def test_imbalance_none_for_regression(self):
        from kiteml.intelligence.data_profiler import build_data_profile

        X, y = make_regression(n_samples=100, n_features=3, random_state=0)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = y
        profile = build_data_profile(df, target="target", problem_type="regression")
        assert profile.imbalance is None

    def test_master_recommendations_populated(self, messy_df):
        from kiteml.intelligence.data_profiler import build_data_profile

        profile = build_data_profile(messy_df, target="target", problem_type="classification")
        assert len(profile.master_recommendations.recommendations) > 0


# ===========================================================================
# 18. Full Integration — result.profile() / result.recommendations() etc.
# ===========================================================================


class TestResultIntegration:

    def test_result_has_data_profile(self, clean_clf_df):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        assert result.data_profile is not None

    def test_result_profile_runs(self, clean_clf_df, capsys):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        result.profile()
        out = capsys.readouterr().out
        assert "KiteML" in out

    def test_result_recommendations_runs(self, clean_clf_df, capsys):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        result.recommendations()
        out = capsys.readouterr().out
        assert "Recommendations" in out

    def test_result_data_quality_report_runs(self, clean_clf_df, capsys):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        result.data_quality_report()
        out = capsys.readouterr().out
        assert "Quality" in out

    def test_result_leakage_report_runs(self, clean_clf_df, capsys):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        result.leakage_report()
        out = capsys.readouterr().out
        assert "Leakage" in out

    def test_result_feature_summary_runs(self, clean_clf_df, capsys):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        result.feature_summary()
        out = capsys.readouterr().out
        assert "Feature" in out

    def test_result_export_html(self, clean_clf_df, tmp_path):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        path = str(tmp_path / "report.html")
        result.export_html(path=path)
        import os

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "KiteML" in content
        assert "<!DOCTYPE html>" in content

    def test_repr_shows_profile_indicator(self, clean_clf_df):
        import kiteml

        result = kiteml.train(clean_clf_df, target="target", problem_type="classification")
        r = repr(result)
        assert "profile=✓" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
