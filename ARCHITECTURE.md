# KiteML Technical Architecture Specification

**KiteML** is an intelligent, full-stack AutoML framework designed to automate dataset profiling, data quality validation, DAG transformation pipeline building, cross-validated model training, REST API serving, containerized deployment, and production drift monitoring.

---

## 🏛️ Ecosystem Architecture Overview

```mermaid
graph TD
    RawData[Raw Dataset / Input DataFrame] --> Intel[Epic 1: Intelligence Layer]
    Intel --> Valid[Epic 2: Validation Layer]
    Valid --> DX[Epic 3: Developer Experience Framework]
    DX --> Pipeline[Epic 4: Intelligent ML Pipeline DAG]
    Pipeline --> Deploy[Epic 5: Intelligent Training & Deployment]

    subgraph "Epic 1: Intelligence Layer"
        Intel --> Profiler[Data Profiler]
        Intel --> Leakage[Leakage Detector]
        Intel --> Imbalance[Imbalance Detector]
        Intel --> SHAP[SHAP Explainability Engine]
    end

    subgraph "Epic 2: Validation Layer"
        Valid --> SchemaVal[SchemaValidator]
        Valid --> TargetVal[TargetValidator]
        Valid --> LeakVal[LeakageValidator]
    end

    subgraph "Epic 3: Developer Experience"
        DX --> Exceptions[KML-XXX Error Catalog]
        DX --> Warnings[KML-W-XXX Warning Policy Engine]
        DX --> Suggestions[Context-Aware Typo Matcher]
    end

    subgraph "Epic 4: Intelligent ML Pipeline"
        Pipeline --> Preproc[Auto Preprocessing Engine]
        Pipeline --> FE[Feature Engineering Engine]
        Pipeline --> FS[Voting Feature Selection Engine]
        Pipeline --> Serial[.kml Binary Serialization]
    end

    subgraph "Epic 5: Intelligent Deployment"
        Deploy --> Serving[FastAPI REST Model Server]
        Deploy --> ONNX[ONNX Graph Converter]
        Deploy --> Docker[Docker Packager]
        Deploy --> Drift[PSI Production Drift Monitor]
    end
```

---

## 🔄 Epic 4 — DAG Transformation Execution Engine

All data transformations in KiteML execute as nodes inside a Directed Acyclic Graph (`dag.py`):

```mermaid
graph LR
    Input[Raw DataFrame] --> ImputerStage[Imputation Stage]
    ImputerStage --> EncoderStage[Categorical Encoder Stage]
    EncoderStage --> ScalerStage[Standard Scaler Stage]
    ScalerStage --> FEStage[Feature Engineering Stage]
    FEStage --> SelectionStage[Voting Feature Selection Stage]
    SelectionStage --> Output[Processed DataFrame / Model]
```

### Key Architectural Guarantee: Zero Data Leakage
Transformation parameters (scaler means/variances, target encoding maps, imputation medians) are computed exclusively on training folds during cross-validation. The fitted DAG is then deterministically applied to test and inference sets.

---

## 📦 `.kml` Serialization & SHA-256 Checksum Validation

Pipelines and trained models are stored as native binary `.kml` packages containing:
- `manifest.json`: Metadata, package version, target column, SHA-256 hash.
- `pipeline_dag.pkl`: Serialized DAG transformation state.
- `model_weights.joblib`: Trained model weights.
- `model_card.json`: Lineage and evaluation metrics.

When `KiteMLPipeline.load("model.kml")` is called, the SHA-256 checksum is verified prior to deserialization to guarantee zero tampering or disk corruption.
