# KiteML Technical Terminology Glossary (`docs-spec/terminology.md`)

To ensure consistent terminology across all documentation, guides, API references, and error messages, use the canonical definitions below.

---

## Technical Glossary

### 1. Architectural Concepts
- **AutoML Framework**: An automated machine learning system that executes data preprocessing, feature engineering, model selection, hyperparameter tuning, evaluation, and deployment without requiring manual user intervention.
- **Epic**: A completed architectural phase in KiteML development (Epic 1: Intelligence, Epic 2: Validation, Epic 3: DX, Epic 4: Intelligent Pipeline, Epic 5: Training & Deployment).
- **DAG Execution Engine**: The Directed Acyclic Graph (`dag.py`) that organizes pipeline transformation stages into an ordered dependency tree and executes them without cycles.
- **KiteMLPipeline**: The unified Python orchestrator class (`kiteml.orchestration.KiteMLPipeline`) that integrates preprocessing, engineering, selection, and serialization in one call.

### 2. File & Artifact Formats
- **`.kml` Package**: The native, serialized KiteML binary archive format containing transformation DAG state, column encoders, scalers, model weights, metadata, and a SHA-256 integrity checksum.
- **Model Card (`model_card.json`)**: A standardized JSON metadata document containing model lineage, training dataset metrics, evaluation performance, and fair-use boundaries.
- **PyPI Package (`kiteml-ai`)**: The official distribution package name on PyPI.
- **Python Import (`import kiteml`)**: The canonical Python import module name.

### 3. Developer Experience (DX) Framework
- **DX Pipeline**: The diagnostic pipeline (`DXPipeline`) that intercepts errors, captures execution context, attaches suggestions, and renders human-readable outputs.
- **KML Error Code**: A standardized structured error identifier formatted as `KML-XXX` (e.g. `KML-101` for target column missing).
- **KML Warning Code**: A standardized warning identifier formatted as `KML-W-XXX` (e.g. `KML-W-201` for severe class imbalance).
- **Warning Policy**: Configurable warning escalation rule (`ignore`, `info`, `warn`, `error`).
- **Fuzzy Matcher**: The string distance algorithm (`match_column_name`) used to detect column name typos.

### 4. Data Intelligence & Pipeline Terms
- **Data Leakage**: The presence of features in the dataset that directly contain information about the target variable not available at real-time inference.
- **Imbalance Ratio**: The proportion of majority-class samples to minority-class samples in classification targets.
- **Voting Feature Selection**: A multi-selector algorithm combining variance thresholding, correlation filtering, mutual information, and model-based feature importance into a consolidated ranking score.
- **Replay Timeline**: An interactive or text-rendered step-by-step audit log detailing every transformation applied to a dataset during pipeline execution.

### 5. Serving & Deployment Terms
- **Inference Guardrails**: Runtime checks enforced prior to model prediction to handle missing columns, out-of-range values, schema shifts, and latency bounds.
- **Data Drift**: Statistical shift in feature distributions between training and production inference datasets measured via Population Stability Index (PSI) or Kolmogorov-Smirnov (KS) tests.
- **Concept Drift**: A change in the relationship between input features and target labels over time in production environments.
