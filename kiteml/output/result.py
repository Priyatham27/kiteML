"""
result.py - Defines the Result object returned by kiteml.train().

Encapsulates:
- Trained model + fitted Preprocessor (backed by sklearn Pipeline internals)
- Ordered feature names (including OHE-expanded columns)
- Typed Metrics dataclass — better autocomplete, IDE support, maintainability
- Evaluation metrics and formatted report
- Prediction API — preprocessing is applied automatically
- predict_proba() with graceful fallback for models that don't support it
- Save / load of the complete artifact bundle
- Feature importance extraction
- Time tracking: total elapsed and per-phase breakdown
"""

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# ===========================================================================
# 🔹 Typed Metrics Dataclasses
# ===========================================================================


@dataclass
class ClassificationMetrics:
    """
    Typed container for classification evaluation metrics.

    Using a dataclass instead of a plain dict gives:
    - IDE autocomplete on every attribute
    - Type checking (mypy / pyright friendly)
    - Easy serialization via dataclasses.asdict()
    - Immutable-feel API — users see exactly what fields exist

    Attributes
    ----------
    accuracy : float
        Overall fraction of correctly classified samples.
    precision : float
        Weighted average precision across all classes.
    recall : float
        Weighted average recall across all classes.
    f1_score : float
        Weighted average F1 score across all classes.
    confusion_matrix : list of list of int
        Raw confusion matrix as a nested list.
    classification_report : str
        Full text report from sklearn.metrics.classification_report.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    classification_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


@dataclass
class RegressionMetrics:
    """
    Typed container for regression evaluation metrics.

    Attributes
    ----------
    r2_score : float
        Coefficient of determination (1.0 = perfect fit).
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error (primary sort metric, lower = better).
    mae : float
        Mean absolute error.
    """

    r2_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


@dataclass
class TrainingTimes:
    """
    Per-phase time breakdown for a KiteML training run.

    Attributes
    ----------
    total : float
        Wall-clock seconds for the full kiteml.train() call.
    training : float
        Seconds spent in model.fit() on the full training set.
    """

    total: float = 0.0
    training: float = 0.0

    def __str__(self) -> str:
        return f"Total: {self.total:.2f}s  |  " f"Training: {self.training:.2f}s"


# ===========================================================================
# 🔹 Result
# ===========================================================================


class Result:
    """
    The face of KiteML — what users interact with after training.

    Attributes
    ----------
    model : Any
        Fitted scikit-learn compatible model.
    model_name : str
        Human-readable class name of the best model.
    metrics : ClassificationMetrics or RegressionMetrics
        Typed metrics object — full IDE autocomplete support.
    report_text : str
        Formatted text report generated after evaluation.
    problem_type : str
        ``'classification'`` or ``'regression'``.
    all_results : Dict[str, Any]
        Cross-validation scores for every candidate model that was tried.
    preprocessor : Any
        The fitted :class:`~kiteml.preprocessing.pipeline.Preprocessor`
        instance (backed by an sklearn Pipeline internally).
    feature_names : List[str]
        Ordered list of feature names after encoding (includes OHE columns).
    feature_importances : Dict[str, float] or None
        Per-feature importance values mapped to ``feature_names``.
    times : TrainingTimes
        Per-phase timing breakdown (total wall-clock + model.fit() time).
    elapsed_time : float
        Alias for ``times.total`` — kept for backward compatibility.
    """

    def __init__(
        self,
        model: Any,
        metrics: Union[Dict[str, Any], ClassificationMetrics, RegressionMetrics],
        report: str,
        problem_type: str,
        all_results: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        preprocessor: Optional[Any] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
        elapsed_time: float = 0.0,
        training_time: float = 0.0,
        data_profile: Optional[Any] = None,
    ):
        self.model = model
        self.model_name = model_name or type(model).__name__
        self.report_text = report
        self.problem_type = problem_type
        self.all_results = all_results or {}
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.feature_importances = feature_importances
        self.data_profile = data_profile  # Phase 2: DataProfile object

        # ── Time tracking ─────────────────────────────────────────────────
        self.times = TrainingTimes(total=elapsed_time, training=training_time)
        self.elapsed_time = elapsed_time  # backward compat alias

        # ── Typed metrics ─────────────────────────────────────────────────
        self.metrics = self._coerce_metrics(metrics, problem_type)

    # ------------------------------------------------------------------
    # 🔹 Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_metrics(
        metrics: Union[Dict[str, Any], ClassificationMetrics, RegressionMetrics],
        problem_type: str,
    ) -> Union[ClassificationMetrics, RegressionMetrics]:
        """
        Accept either a raw dict (legacy) or a typed dataclass.

        This ensures backward compatibility — existing code that passes
        a plain dict still works, and new code gets the typed object.
        """
        if isinstance(metrics, (ClassificationMetrics, RegressionMetrics)):
            return metrics

        # Convert raw dict → typed dataclass
        if problem_type == "classification":
            return ClassificationMetrics(
                accuracy=metrics.get("accuracy", 0.0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                confusion_matrix=metrics.get("confusion_matrix", []),
                classification_report=metrics.get("classification_report", ""),
            )
        else:
            return RegressionMetrics(
                r2_score=metrics.get("r2_score", 0.0),
                mse=metrics.get("mse", 0.0),
                rmse=metrics.get("rmse", 0.0),
                mae=metrics.get("mae", 0.0),
            )

    # ------------------------------------------------------------------
    # 🔹 Pretty Metric Accessors
    # ------------------------------------------------------------------

    @property
    def accuracy(self) -> Optional[float]:
        """Classification accuracy (``None`` for regression tasks)."""
        if isinstance(self.metrics, ClassificationMetrics):
            return self.metrics.accuracy
        return None

    @property
    def f1(self) -> Optional[float]:
        """Weighted F1 score (``None`` for regression tasks)."""
        if isinstance(self.metrics, ClassificationMetrics):
            return self.metrics.f1_score
        return None

    @property
    def r2(self) -> Optional[float]:
        """R² score (``None`` for classification tasks)."""
        if isinstance(self.metrics, RegressionMetrics):
            return self.metrics.r2_score
        return None

    @property
    def rmse(self) -> Optional[float]:
        """Root Mean Squared Error (``None`` for classification tasks)."""
        if isinstance(self.metrics, RegressionMetrics):
            return self.metrics.rmse
        return None

    @property
    def mae(self) -> Optional[float]:
        """Mean Absolute Error (``None`` for classification tasks)."""
        if isinstance(self.metrics, RegressionMetrics):
            return self.metrics.mae
        return None

    @property
    def score(self) -> Optional[float]:
        """
        Primary score for the task type.

        - Classification → accuracy
        - Regression      → R²
        """
        if self.problem_type == "classification":
            return self.accuracy
        return self.r2

    # ------------------------------------------------------------------
    # 🔹 Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return the full formatted evaluation report."""
        return self.report_text

    def summary(self) -> None:
        """Print a clean, human-readable training summary."""
        sep = "─" * 46
        print("\n🪁 KiteML Training Summary")
        print(sep)
        print(f"  Problem Type  : {self.problem_type}")
        print(f"  Best Model    : {self.model_name}")
        print(f"  Total Time    : {self.times.total:.2f}s")
        print(f"  Training Time : {self.times.training:.2f}s")
        print(sep)

        if self.problem_type == "classification":
            if self.accuracy is not None:
                print(f"  Accuracy      : {self.accuracy:.4f}")
            if self.f1 is not None:
                print(f"  F1 Score      : {self.f1:.4f}")
        else:
            if self.r2 is not None:
                print(f"  R² Score      : {self.r2:.4f}")
            if self.rmse is not None:
                print(f"  RMSE          : {self.rmse:.4f}")
            if self.mae is not None:
                print(f"  MAE           : {self.mae:.4f}")

        # Top-5 feature importances when available
        if self.feature_importances:
            top = sorted(
                self.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            print(sep)
            print("  Top Features:")
            for feat, imp in top:
                bar = "█" * int(abs(imp) * 20)
                print(f"    {feat:<20s} {imp:+.4f}  {bar}")

        # Model leaderboard
        if self.all_results:
            scored = {
                k: v["score"] for k, v in self.all_results.items() if isinstance(v, dict) and v.get("score") is not None
            }
            if scored:
                print(sep)
                print("  Model Leaderboard:")
                ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
                for idx, (name, sc) in enumerate(ranked, start=1):
                    marker = " ✓" if name == self.model_name else ""
                    print(f"    #{idx:<3} {name:<26} {sc:.4f}{marker}")

        print(sep)

    # ------------------------------------------------------------------
    # 🔹 Model Persistence  (save / load full artifact bundle)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the complete artifact bundle to disk.

        The bundle contains the model, the fitted Preprocessor (which holds
        the sklearn Pipeline internally), feature names, typed metrics, and
        time tracking — everything needed for inference with no extra code.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'model.pkl'``).
        """
        bundle = {
            "model": self.model,
            "model_name": self.model_name,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "problem_type": self.problem_type,
            "metrics": self.metrics,  # typed dataclass
            "metrics_dict": self.metrics.to_dict(),  # plain dict for interop
            "feature_importances": self.feature_importances,
            "times": self.times,
        }
        joblib.dump(bundle, path)
        print(f"💾 Bundle saved → {path}")

    # Keep old name for backward compatibility
    def save_model(self, path: str = "kiteml_model.pkl") -> None:
        """Alias for :meth:`save`."""
        self.save(path)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Load a saved artifact bundle from disk.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: ``model``, ``preprocessor``,
            ``feature_names``, ``problem_type``, ``metrics``,
            ``feature_importances``, ``times``.
        """
        bundle = joblib.load(path)
        print(f"📂 Bundle loaded ← {path}")
        return bundle

    # Keep old name for backward compatibility
    @staticmethod
    def load_model(path: str = "kiteml_model.pkl") -> Any:
        """Alias for :meth:`load` — returns the model object only."""
        bundle = joblib.load(path)
        print(f"📂 Model loaded ← {path}")
        return bundle.get("model", bundle)

    # ------------------------------------------------------------------
    # 🔹 Prediction API  (preprocessing is handled automatically)
    # ------------------------------------------------------------------

    def _preprocess_input(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted Preprocessor to raw input data.

        Unknown categories are handled gracefully by the OHE step inside
        the sklearn Pipeline (``handle_unknown='ignore'`` → all-zeros vector).

        Raises
        ------
        RuntimeError
            If no preprocessor is attached (e.g. Result was built manually).
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "No preprocessor attached to this Result. " "Re-run kiteml.train() to get a fully wired Result."
            )
        return self.preprocessor.transform(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new raw data.

        The Preprocessor (imputation → encoding → scaling, all backed by
        sklearn Pipeline) is applied automatically.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (unprocessed) feature DataFrame.

        Returns
        -------
        np.ndarray
            Predicted labels (classification) or values (regression).
        """
        X_processed = self._preprocess_input(X)
        return self.model.predict(X_processed)

    def predict_proba(
        self,
        X: pd.DataFrame,
        fallback_to_predict: bool = False,
    ) -> np.ndarray:
        """
        Return class probabilities (classification only).

        Parameters
        ----------
        X : pd.DataFrame
            Raw (unprocessed) feature DataFrame.
        fallback_to_predict : bool
            When ``True`` and the model does not support ``predict_proba``
            (e.g. plain SVM), fall back silently to ``predict()`` and return
            a one-hot encoded array of shape ``(n_samples, n_classes)``.
            When ``False`` (default), raise ``AttributeError`` instead so
            the caller is made aware of the limitation.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, n_classes)`` containing class
            probabilities, or one-hot predictions when using the fallback.

        Raises
        ------
        AttributeError
            If the model does not support ``predict_proba`` and
            ``fallback_to_predict=False``.
        """
        X_processed = self._preprocess_input(X)

        # ── Check for predict_proba support ──────────────────────────────
        has_proba = hasattr(self.model, "predict_proba") and callable(getattr(self.model, "predict_proba", None))

        if has_proba:
            return self.model.predict_proba(X_processed)

        # ── Fallback handling ─────────────────────────────────────────────
        if fallback_to_predict:
            warnings.warn(
                f"{self.model_name} does not support predict_proba. "
                "Falling back to predict() — returning one-hot encoded predictions. "
                "Consider using a probabilistic model (RandomForest, LogisticRegression).",
                UserWarning,
                stacklevel=2,
            )
            preds = self.model.predict(X_processed)
            classes = np.unique(preds)
            one_hot = (preds[:, None] == classes[None, :]).astype(float)
            return one_hot

        raise AttributeError(
            f"{self.model_name} does not support predict_proba. "
            "Use fallback_to_predict=True to get one-hot encoded predictions instead, "
            "or choose a probabilistic model such as RandomForest or LogisticRegression."
        )

    # ------------------------------------------------------------------
    # 🔹 Feature Importance
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Return feature importance values, sorted by absolute magnitude.

        Uses ``feature_importances_`` (tree ensembles) or ``coef_``
        (linear models). Returns ``None`` when neither is available.

        Parameters
        ----------
        top_n : int, optional
            Limit to the top N features. Returns all when ``None``.

        Returns
        -------
        Dict[str, float] or None
        """
        if self.feature_importances is None:
            print("⚠️  Feature importance not available for this model.")
            return None

        sorted_fi = dict(
            sorted(
                self.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )
        if top_n is not None:
            sorted_fi = dict(list(sorted_fi.items())[:top_n])
        return sorted_fi

    # ------------------------------------------------------------------
    # 🔹 Leaderboard helpers
    # ------------------------------------------------------------------

    def leaderboard(self) -> Optional[pd.DataFrame]:
        """
        Return a sorted DataFrame of all model cross-validation scores.

        Returns
        -------
        pd.DataFrame or None
        """
        if not self.all_results:
            return None

        rows = []
        for name, info in self.all_results.items():
            if isinstance(info, dict):
                score = info.get("score")
                rank = info.get("rank")
                error = info.get("error")
            else:
                # backward-compat: plain float values
                score = info if isinstance(info, float) else None
                rank = None
                error = None if score is not None else str(info)

            rows.append(
                {
                    "Rank": rank,
                    "Model": name,
                    "CV Score": score,
                    "Best": name == self.model_name,
                    "Error": error,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("Rank", ascending=True, na_position="last").reset_index(drop=True)
        return df

    def ranking(self) -> None:
        """
        Print a clean ranked leaderboard to the terminal.

        Only models that completed cross-validation are shown, sorted
        from best to worst.  The winning model is marked with ✓.
        """
        df = self.leaderboard()
        if df is None or df.empty:
            print("⚠️  No leaderboard data available.")
            return

        W = 52
        print("\n" + "═" * W)
        print("  🪁  KiteML — Model Ranking")
        print("═" * W)
        print(f"  {'Rank':<5} {'Model':<26} {'CV Score':>8}")
        print("─" * W)

        for _, row in df.iterrows():
            if pd.isna(row["CV Score"]):
                print(f"  {'ERR':<5} {row['Model']:<26} {'FAILED':>8}")
            else:
                rank_str = f"#{int(row['Rank'])}"
                marker = " ✓" if row["Best"] else ""
                print(f"  {rank_str:<5} {row['Model']:<26} {row['CV Score']:>8.4f}{marker}")

        print("═" * W)

    # ------------------------------------------------------------------
    # 🔹 Phase 2 — Intelligence Layer Methods
    # ------------------------------------------------------------------

    def _require_profile(self, method_name: str):
        """Raise a helpful error if no DataProfile is attached."""
        if self.data_profile is None:
            raise RuntimeError(
                f"result.{method_name}() requires a DataProfile. " "Re-run kiteml.train() — profiling is automatic."
            )

    def profile(self, output: str = "terminal") -> None:
        """
        Display the full dataset intelligence report.

        Parameters
        ----------
        output : str
            ``'terminal'`` (default) prints to stdout.
            ``'html'`` saves to ``kiteml_report.html``.
        """
        self._require_profile("profile")
        if output == "html":
            from kiteml.profiling.html_export import export_html

            export_html(self.data_profile)
        else:
            from kiteml.profiling.report_generator import generate_profile_report

            print(generate_profile_report(self.data_profile))

    def recommendations(self) -> None:
        """Print all prioritized intelligence recommendations."""
        self._require_profile("recommendations")
        self.data_profile.master_recommendations.print_report()

    def data_quality_report(self) -> None:
        """Print the data quality report (issues, score, summary)."""
        self._require_profile("data_quality_report")
        q = self.data_profile.quality
        W = 54
        print("\n" + "═" * W)
        print("  🪁  KiteML — Data Quality Report")
        print("═" * W)
        print(f"  Quality Score  : {q.score}/100")
        print(f"  Summary        : {q.summary}")
        print("─" * W)
        if not q.issues:
            print("  ✅ No issues detected.")
        else:
            for issue in q.issues:
                icon = "🔴" if issue.severity.value == "error" else "🟡" if issue.severity.value == "warning" else "ℹ️ "
                print(f"  {icon} [{issue.issue_type}]")
                print(f"     {issue.description}")
                print(f"     → {issue.recommendation}")
        print("═" * W)

    def leakage_report(self) -> None:
        """Print the leakage detection report."""
        self._require_profile("leakage_report")
        lk = self.data_profile.leakage
        W = 54
        print("\n" + "═" * W)
        print("  🪁  KiteML — Leakage Detection Report")
        print("═" * W)
        if not lk.has_leakage_risk:
            print("  ✅ No leakage risks detected.")
        else:
            for risk in lk.risks:
                icon = "🚨" if risk.risk_level == "critical" else "⚠️ "
                print(f"  {icon} [{risk.risk_level.upper()}] '{risk.column}'")
                print(f"     Correlation: {risk.correlation_with_target:.4f}")
                print(f"     Reason: {risk.reason}")
        print("─" * W)
        for msg in lk.recommendations:
            print(f"  {msg}")
        print("═" * W)

    def feature_summary(self, top_n: int = 10) -> None:
        """
        Print a comprehensive feature intelligence summary.

        Covers column types, recommendations, and top importances.
        """
        self._require_profile("feature_summary")
        W = 56
        print("\n" + "═" * W)
        print("  🪁  KiteML — Feature Intelligence Summary")
        print("═" * W)

        # Column type breakdown
        print("  Column Types:")
        for col_type, count in self.data_profile.column_analysis.type_summary.items():
            print(f"    {col_type:<20} {count}")

        # Feature recommendations
        feat_recs = self.data_profile.feature_recommendations
        if feat_recs.recommendations:
            print("─" * W)
            print("  Feature Recommendations:")
            for rec in feat_recs.recommendations[:8]:
                icon = "⚠️ " if rec.priority == "high" else "💡"
                print(f"    {icon} [{rec.action.upper()}] {rec.column}")
                print(f"       {rec.reason[:70]}")

        # Feature importances
        if self.feature_importances:
            print("─" * W)
            print(f"  Top {top_n} Feature Importances:")
            sorted_fi = sorted(
                self.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            for feat, imp in sorted_fi:
                bar = "█" * int(abs(imp) * 30)
                print(f"    {feat:<22} {imp:+.4f}  {bar}")

        print("═" * W)

    def export_html(self, path: str = "kiteml_report.html") -> str:
        """
        Export the full dataset intelligence report as a self-contained HTML file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        str
            Path where the HTML file was saved.
        """
        self._require_profile("export_html")
        from kiteml.profiling.html_export import export_html

        return export_html(self.data_profile, path=path)

    # ------------------------------------------------------------------
    # 🔹 Phase 3 — Production & Deployment Layer
    # ------------------------------------------------------------------

    def package(
        self,
        path: str = "model.kiteml",
        target_column: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Export a self-contained deployable .kiteml bundle.

        Bundle contains: model, preprocessor, schema, metrics,
        manifest, environment snapshot, and feature importances.

        Parameters
        ----------
        path : str
            Output directory path. Default ``'model.kiteml'``.
        target_column : str, optional
            Target column name for metadata.
        overwrite : bool
            Overwrite an existing bundle. Default False.

        Returns
        -------
        PackageResult
        """
        from kiteml.deployment.packaging import package as _package

        return _package(self, path=path, target_column=target_column, overwrite=overwrite)

    def batch_predict(
        self,
        data,
        chunk_size: int = 1000,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Run memory-safe batch inference on a large dataset.

        Parameters
        ----------
        data : str or pd.DataFrame
            CSV/Parquet file path or DataFrame.
        chunk_size : int
            Rows per inference chunk. Default 1000.
        output_path : str, optional
            Save predictions CSV here.
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        BatchResult
        """
        from kiteml.deployment.batch_inference import batch_predict as _bp

        return _bp(self, data=data, chunk_size=chunk_size, output_path=output_path, verbose=verbose)

    def monitor_drift(
        self,
        current_data,
        reference_data=None,
        feature_names=None,
    ):
        """
        Detect data drift between reference (training) and current data.

        Parameters
        ----------
        current_data : pd.DataFrame
            New production data to check for drift.
        reference_data : pd.DataFrame, optional
            Baseline data. If None, uses feature distributions from the
            training DataProfile if available.
        feature_names : list, optional
            Columns to monitor. Defaults to all numeric features.

        Returns
        -------
        DriftReport
        """
        from kiteml.monitoring.drift_monitor import check_drift

        if reference_data is None:
            raise ValueError(
                "reference_data is required for drift monitoring. " "Pass your training DataFrame as reference_data."
            )
        report = check_drift(
            reference_df=reference_data,
            current_df=current_data,
            feature_names=feature_names or self.feature_names,
        )
        print(report.summary())
        for rec in report.recommendations:
            print(f"  {rec}")
        return report

    def export_onnx(
        self,
        path: str = "model.onnx",
        opset_version: int = 17,
    ):
        """
        Export the trained model to ONNX format.

        Requires: ``pip install skl2onnx onnx``

        Parameters
        ----------
        path : str
            Output file path. Default ``'model.onnx'``.
        opset_version : int
            ONNX opset version. Default 17.

        Returns
        -------
        OnnxExportResult
        """
        from kiteml.deployment.onnx_export import export_onnx as _export

        return _export(
            model=self.model,
            feature_names=self.feature_names or [],
            path=path,
            opset_version=opset_version,
        )

    def export_docker(
        self,
        output_dir: str = "docker_deploy",
        port: int = 8000,
    ):
        """
        Generate a complete Docker deployment package.

        Generates: Dockerfile, requirements.txt, serve.py (FastAPI),
        docker-compose.yml, and .dockerignore.

        Parameters
        ----------
        output_dir : str
            Directory to write Docker files. Default ``'docker_deploy'``.
        port : int
            Server port. Default 8000.

        Returns
        -------
        DockerExportResult
        """
        from kiteml.deployment.docker_export import export_docker as _docker

        return _docker(self, output_dir=output_dir, port=port)

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        background: bool = False,
        open_browser: bool = False,
    ) -> None:
        """
        Launch a FastAPI REST inference server.

        Requires: ``pip install fastapi uvicorn``

        Parameters
        ----------
        host : str
            Server host. Default ``'0.0.0.0'``.
        port : int
            Server port. Default 8000.
        background : bool
            Run in background thread (useful for notebooks). Default False.
        open_browser : bool
            Auto-open Swagger UI. Default False.
        """
        from kiteml.deployment.model_server import serve as _serve

        _serve(self, host=host, port=port, background=background, open_browser=open_browser)

    def generate_api(
        self,
        output_dir: str = ".",
        filename: str = "kiteml_api.py",
    ) -> str:
        """
        Generate a standalone FastAPI inference script.

        Parameters
        ----------
        output_dir : str
            Directory to write the file. Default current directory.
        filename : str
            Output filename. Default ``'kiteml_api.py'``.

        Returns
        -------
        str
            Path to the generated file.
        """
        from kiteml.deployment.api_generator import generate_api as _gen

        return _gen(self, output_dir=output_dir, filename=filename)

    def experiment(
        self,
        experiment_name: str = "default",
        dataset=None,
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
    ):
        """
        Track this training run in the KiteML experiment store.

        Parameters
        ----------
        experiment_name : str
            Logical group for this run. Default ``'default'``.
        dataset : pd.DataFrame, optional
            Training dataset (for hash computation).
        tags : dict, optional
            Free-form key-value metadata.
        notes : str
            Human-readable notes.

        Returns
        -------
        ExperimentRun
        """
        from kiteml.experiments.tracker import track

        return track(
            self,
            experiment_name=experiment_name,
            dataset=dataset,
            tags=tags,
            notes=notes,
        )

    def version(
        self,
        version: Optional[str] = None,
        bump: str = "patch",
        notes: str = "",
        bundle_path: Optional[str] = None,
    ):
        """
        Record a semantic version for this model.

        Parameters
        ----------
        version : str, optional
            Explicit version string (e.g. ``'v1.2.0'``). Auto-bumps if None.
        bump : str
            ``'major'``, ``'minor'``, or ``'patch'``. Default ``'patch'``.
        notes : str
            Release notes.
        bundle_path : str, optional
            Associated .kiteml bundle path.

        Returns
        -------
        ModelVersion
        """
        from kiteml.governance.versioning import version_model

        return version_model(
            self,
            version=version,
            bump=bump,
            notes=notes,
            bundle_path=bundle_path,
        )

    def lineage(
        self,
        dataset=None,
        print_tree: bool = True,
    ):
        """
        Build and optionally display the full pipeline lineage.

        Parameters
        ----------
        dataset : pd.DataFrame, optional
        print_tree : bool
            Print the lineage tree to terminal. Default True.

        Returns
        -------
        PipelineLineage
        """
        from kiteml.governance.lineage import build_lineage

        lin = build_lineage(self, dataset=dataset)
        if print_tree:
            lin.print_lineage()
        return lin

    def sign(self):
        """
        Generate a cryptographic fingerprint for integrity verification.

        Returns
        -------
        ModelSignature
        """
        from kiteml.governance.signatures import sign_model

        sig = sign_model(self)
        print(f"🔏 Model signature: {sig.fingerprint}")
        return sig

    def generate_dashboard(
        self,
        path: str = "kiteml_dashboard.html",
        drift_report=None,
    ) -> str:
        """
        Generate a rich production deployment dashboard.

        Parameters
        ----------
        path : str
            Output HTML file path. Default ``'kiteml_dashboard.html'``.
        drift_report : DriftReport, optional
            Include drift status in dashboard.

        Returns
        -------
        str
            Path to saved dashboard.
        """
        from kiteml.profiling.deployment_dashboard import generate_dashboard

        return generate_dashboard(self, path=path, drift_report=drift_report)

    def realtime_engine(self):
        """
        Return a RealtimeInferenceEngine for low-latency single-row inference.

        Returns
        -------
        RealtimeInferenceEngine
        """
        from kiteml.deployment.realtime_inference import RealtimeInferenceEngine

        return RealtimeInferenceEngine(
            model=self.model,
            feature_names=self.feature_names or [],
            problem_type=self.problem_type,
            preprocessor=self.preprocessor,
        )

    def data_contract(
        self,
        version: str = "1.0.0",
        save_path: Optional[str] = None,
    ):
        """
        Generate a formal data contract from this Result.

        Parameters
        ----------
        version : str
        save_path : str, optional
            Save contract JSON here.

        Returns
        -------
        DataContract
        """
        from kiteml.monitoring.data_contracts import DataContract

        contract = DataContract.from_result(self, version=version)
        if save_path:
            contract.save(save_path)
        return contract

    def __repr__(self) -> str:
        score_str = f"{self.score:.4f}" if self.score is not None else "N/A"
        has_profile = "✓" if self.data_profile is not None else "✗"
        return (
            f"<KiteML Result | model={self.model_name} | "
            f"type={self.problem_type} | score={score_str} | "
            f"time={self.times.total:.2f}s | profile={has_profile}>"
        )
