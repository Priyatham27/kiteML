# System Architecture Overview

KiteML is architected as a modular, 5-epic AutoML ecosystem that processes raw tabular datasets through data intelligence scanning, pre-flight quality validation, DAG pipeline execution, cross-validated training, REST API serving, and drift monitoring.

---

## 🏛️ Ecosystem Architecture Diagram

```mermaid
graph TD
    RawData[Raw Dataset] --> Intel[Epic 1: Intelligence Layer]
    Intel --> Valid[Epic 2: Validation Layer]
    Valid --> DX[Epic 3: DX Framework]
    DX --> Pipeline[Epic 4: Intelligent ML Pipeline DAG]
    Pipeline --> Deploy[Epic 5: Training & Deployment]

    subgraph "Epic 1: Intelligence Layer"
        Intel --> Profiler[Data Profiler]
        Intel --> Leakage[Leakage Detector]
        Intel --> Imbalance[Imbalance Detector]
        Intel --> SHAP[SHAP Explainability]
    end

    subgraph "Epic 4: Intelligent ML Pipeline"
        Pipeline --> Preproc[Auto Preprocessing]
        Pipeline --> FE[Feature Engineering]
        Pipeline --> FS[Voting Feature Selection]
        Pipeline --> Serial[.kml Serialization]
    end

    subgraph "Epic 5: Intelligent Deployment"
        Deploy --> Serving[FastAPI Model Server]
        Deploy --> ONNX[ONNX Graph Exporter]
        Deploy --> Docker[Docker Packager]
        Deploy --> Drift[PSI Drift Monitor]
    end
```
