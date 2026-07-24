# Master Documentation Generation Roadmap (`docs-spec/documentation-roadmap.md`)

This roadmap defines the sequential execution steps for building, generating, and validating the complete KiteML documentation suite.

---

## Roadmap Progression Matrix

```
Step 1: Audit Repository & Plan           [COMPLETED]
Step 2: Design Architecture & Specs       [COMPLETED]
Step 3: Generate Modern README.md         [NEXT]
Step 4: Generate Getting Started Pages    [PENDING]
Step 5: Generate Comprehensive User Guides [PENDING]
Step 6: Generate Auto API Documentation   [PENDING]
Step 7: Generate Architecture Specs & DAG [PENDING]
Step 8: Generate Interactive Tutorials    [PENDING]
Step 9: Generate Example Projects & Code  [PENDING]
Step 10: Build, Validate & Review Docs    [PENDING]
```

---

## Detailed Generation Specs by Step

### Step 3 — Modern README.md Generation
- **Target File**: `README.md` (root)
- **Specification Source**: `docs-spec/public-api.md`, `docs-spec/architecture-summary.md`
- **Output Requirements**:
  - Hero header with badges (PyPI version, Python versions, test workflow, license, downloads).
  - High-level overview and feature summary matrix (Core ML, Intelligence, Production, CLI, MLOps, Governance).
  - Quickstart code blocks (Python API & CLI).
  - Epic 4 & Epic 5 feature spotlight (`KiteMLPipeline`, `.kml` package format).
  - Directory structure architecture tree.
  - Links to full documentation site, PyPI, and GitHub.

### Step 4 — Getting Started Section Generation
- **Target Files**:
  - `docs/getting_started/index.md`
  - `docs/getting_started/installation.md`
  - `docs/getting_started/quickstart_python.md`
  - `docs/getting_started/quickstart_cli.md`
  - `docs/getting_started/concepts.md`
- **Specification Source**: `docs-spec/public-api.md`, `docs-spec/terminology.md`

### Step 5 — User Guides Generation (Epics 1–5)
- **Target Files**:
  - `docs/user_guides/intelligence/profiling_leakage.md`
  - `docs/user_guides/intelligence/imbalance_outliers.md`
  - `docs/user_guides/intelligence/explainability.md`
  - `docs/user_guides/validation/schemas.md`
  - `docs/user_guides/dx/diagnostics.md`
  - `docs/user_guides/pipeline/preprocessing.md`
  - `docs/user_guides/pipeline/feature_engineering.md`
  - `docs/user_guides/pipeline/feature_selection.md`
  - `docs/user_guides/pipeline/dag_orchestration.md`
  - `docs/user_guides/pipeline/serialization.md`
  - `docs/user_guides/deployment/fastapi_serving.md`
  - `docs/user_guides/deployment/onnx_export.md`
  - `docs/user_guides/deployment/docker_packaging.md`
  - `docs/user_guides/deployment/drift_monitoring.md`
  - `docs/user_guides/deployment/governance.md`
- **Specification Source**: `docs-spec/architecture-summary.md`, `docs-spec/module-index.md`

### Step 6 — API Reference Documentation Generation
- **Target Files**: `docs/api/*.md` (10 sub-pages configured for `mkdocstrings`)
- **Specification Source**: `docs-spec/public-api.md`, `docs-spec/module-index.md`

### Step 7 — Architecture & Specs Documentation
- **Target Files**: `docs/architecture/*.md` (Mermaid flowcharts, DAG specs, `.kml` checksum validation)

### Step 8 — Tutorials & Jupyter Notebooks Generation
- **Target Files**: `docs/tutorials/*.md` & `.ipynb`

### Step 9 — Example Projects Generation
- **Target Files**: `examples/` runnable Python scripts for Classification, Regression, FastAPIServing, and Plugins.

### Step 10 — Review, Build & Validation
- **Commands**:
  ```bash
  python scripts/build_docs.py
  mkdocs build --strict
  ```
