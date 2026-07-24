# KiteML Module Index Specification (`docs-spec/module-index.md`)

This index catalogs all 32 submodules and top-level modules within the `kiteml` package.

| Submodule / File | Architectural Epic | Primary Purpose & Key Classes/Functions |
| :--- | :--- | :--- |
| `kiteml/__init__.py` | Top-Level | Defines public package exports (`train`, `load`, `validate`, `KiteMLPipeline`, `Result`). |
| `kiteml/core.py` | Top-Level | Core implementation of `train()` workflow, pipeline orchestration glue, and validation calls. |
| `kiteml/config.py` | Top-Level | Global constants (`DEFAULT_TEST_SIZE`, `DEFAULT_CV_FOLDS`), default model hyperparameter presets. |
| `kiteml/cli.py` | Top-Level / DX | Legacy CLI entrypoint adapter forwarding to `kiteml.cli.main`. |
| `kiteml/cli/` | Epic 3 (DX) | 14-command CLI ecosystem: `train`, `serve`, `predict`, `profile`, `doctor`, `init`, `playground`, `dashboard`, `monitor`, `export`, `benchmark`, `experiment`, `plugins`, `version`. |
| `kiteml/deployment/` | Epic 5 (Deploy) | Deployment utilities: ONNX exporter (`onnx_export.py`), Docker packager (`docker_export.py`), inference guardrails (`inference_guardrails.py`), FastAPI generator (`api_generator.py`). |
| `kiteml/evaluation/` | Epic 5 (Eval) | Classification & regression evaluation metrics (ROC-AUC, confusion matrix, residual plots). |
| `kiteml/exceptions/` | Epic 3 (DX) | Structured exception framework (`KiteMLError`, `DatasetError`, `SchemaError`, error code catalog `KML-XXXNNN`). |
| `kiteml/experiments/` | Epic 5 (MLOps) | Experiment tracker, run logger, parameter logging, and artifact tracking. |
| `kiteml/feature_engineering/` | Epic 4 (Pipeline) | Automated feature creation engine (`planner.py`, `blueprint.py`, `providers.py`, `importance_predictor.py`). |
| `kiteml/feature_selection/` | Epic 4 (Pipeline) | Voting-based multi-selector feature filtering (variance, correlation, mutual information, model importance). |
| `kiteml/governance/` | Epic 5 (Deploy) | Automated Model Card generation (`model_card.json`), compliance checks, and audit trail logging. |
| `kiteml/integrations/` | Epic 5 (MLOps) | External MLOps service adapters (`wandb.py`, `mlflow.py`). |
| `kiteml/intelligence/` | Epic 1 (Intel) | Profiling & data intelligence: column analyzer, cardinality, correlation, datetime/text detectors, leakage detector, imbalance detector, outlier detector, memory optimizer, SHAP explainability. |
| `kiteml/ml/` | Core ML | High-level ML wrapper providing backward-compatible `train()` and `load()` functions. |
| `kiteml/models/` | Epic 5 (Training) | Model catalog wrappers for scikit-learn, LightGBM, and XGBoost classifiers & regressors. |
| `kiteml/monitoring/` | Epic 5 (Deploy) | Data drift (PSI, KS-test), concept drift, and prediction latency monitoring. |
| `kiteml/optimization/` | Epic 5 (Training) | Hyperparameter tuning algorithms (RandomizedSearch, Optuna integrations). |
| `kiteml/orchestration/` | Epic 4 (Pipeline) | Unified `KiteMLPipeline` orchestrator, workflow engine, lifecycle event hooks, execution context. |
| `kiteml/output/` | Core ML | Output container objects (`Result`, `TrainingResult`, `ClassificationMetrics`, `RegressionMetrics`). |
| `kiteml/pipeline/` | Epic 4 (Pipeline) | Internal DAG engine (`dag.py`, `transformation_pipeline.py`), `DXPipeline`, stages, diagnostic collectors. |
| `kiteml/plugins/` | Epic 5 (Deploy) | Plugin SDK (`plugin_base.py`, `plugin_manager.py`) for custom preprocessing/feature engineering extension stages. |
| `kiteml/prediction/` | Epic 5 (Deploy) | Batch & real-time inference runners with schema validation and post-processing. |
| `kiteml/preprocessing/` | Epic 4 (Pipeline) | Preprocessing planner, blueprint, cleaner, encoder, scaler, and pipeline assembly. |
| `kiteml/profiling/` | Epic 1 (Intel) | Legacy profiling interface and dataset summary report formatting. |
| `kiteml/registry/` | Epic 5 (Deploy) | Local model registry, artifact storage, version tagging, and model loading index. |
| `kiteml/reporting/` | Epic 4 (Pipeline) | Execution report generator, replay timeline, HTML dashboards, JSON log exports. |
| `kiteml/selection/` | Epic 5 (Training) | Model selection engine, cross-validated tournament runner, best model evaluation. |
| `kiteml/serialization/` | Epic 4 (Pipeline) | Native `.kml` pipeline serialization with SHA-256 integrity verification. |
| `kiteml/serving/` | Epic 5 (Deploy) | Production FastAPI server (`model_server.py`), Pydantic schema generation, OpenAPI documentation router. |
| `kiteml/suggestions/` | Epic 3 (DX) | Context-aware suggestions engine (`suggestion_manager.py`), fuzzy column name matching, domain recommendations. |
| `kiteml/training/` | Epic 5 (Training) | Model trainer execution loop, progress callbacks, early stopping. |
| `kiteml/utils/` | Utilities | Helper functions for IO, string formatting, DataFrame manipulation, logging. |
| `kiteml/validation/` | Epic 2 (Valid) | Dataset, schema, target, leakage, and data quality validation modules. |
| `kiteml/warnings/` | Epic 3 (DX) | Structured warning taxonomy (`KiteMLWarning`), catalog (`KML-W-XXX`), warning policy escalation manager. |
