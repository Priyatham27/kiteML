"""
lineage.py — Full training pipeline lineage tracking for KiteML.

Captures the complete ML pipeline lineage: dataset → preprocessing →
model selection → training → evaluation → deployment, with timestamps
and artifact checksums at each stage.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LineageStep:
    """A single step in the ML pipeline lineage."""
    step_name: str
    step_type: str     # "data", "preprocessing", "selection", "training", "evaluation", "deployment"
    timestamp: str
    duration_s: Optional[float]
    artifacts: Dict[str, str]    # artifact_name → checksum or path
    metadata: Dict[str, Any]


@dataclass
class PipelineLineage:
    """Complete lineage of a KiteML training run."""
    lineage_id: str
    model_name: str
    problem_type: str
    steps: List[LineageStep]
    created_at: str
    total_duration_s: float

    def to_dict(self) -> dict:
        return {
            "lineage_id": self.lineage_id,
            "model_name": self.model_name,
            "problem_type": self.problem_type,
            "created_at": self.created_at,
            "total_duration_s": self.total_duration_s,
            "steps": [s.__dict__ for s in self.steps],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def print_lineage(self) -> None:
        W = 58
        print("\n" + "═" * W)
        print("  🔗  KiteML — Pipeline Lineage")
        print("═" * W)
        for i, step in enumerate(self.steps):
            connector = "└─" if i == len(self.steps) - 1 else "├─"
            dur = f" ({step.duration_s:.2f}s)" if step.duration_s else ""
            print(f"  {connector} [{step.step_type.upper():<14}] {step.step_name}{dur}")
        print("─" * W)
        print(f"  Total: {self.total_duration_s:.2f}s | ID: {self.lineage_id}")
        print("═" * W)


def build_lineage(result: Any, dataset: Optional[Any] = None) -> PipelineLineage:
    """
    Build a PipelineLineage from a KiteML Result.

    Parameters
    ----------
    result : Result
    dataset : pd.DataFrame, optional
        Training dataset for hash computation.

    Returns
    -------
    PipelineLineage
    """
    import uuid

    lineage_id = str(uuid.uuid4())[:8]
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    steps: List[LineageStep] = []

    # ── Step 1: Dataset ───────────────────────────────────────────────────
    dataset_meta: Dict[str, Any] = {}
    dataset_artifacts: Dict[str, str] = {}
    if dataset is not None:
        try:
            import pandas as pd
            n_rows, n_cols = dataset.shape
            ds_hash = hashlib.md5(
                pd.util.hash_pandas_object(dataset).values.tobytes()
            ).hexdigest()[:12]
            dataset_meta = {"n_rows": n_rows, "n_cols": n_cols}
            dataset_artifacts = {"dataset_hash": ds_hash}
        except Exception:
            pass

    if dataset is not None or result.data_profile:
        steps.append(LineageStep(
            step_name="Load & Profile Dataset",
            step_type="data",
            timestamp=created_at,
            duration_s=None,
            artifacts=dataset_artifacts,
            metadata=dataset_meta,
        ))

    # ── Step 2: Preprocessing ─────────────────────────────────────────────
    steps.append(LineageStep(
        step_name="Preprocess Features",
        step_type="preprocessing",
        timestamp=created_at,
        duration_s=None,
        artifacts={"preprocessor": "preprocessor.joblib"},
        metadata={
            "n_features": len(result.feature_names or []),
            "feature_names": list(result.feature_names or []),
        },
    ))

    # ── Step 3: Model Selection ───────────────────────────────────────────
    steps.append(LineageStep(
        step_name="Select Best Model (CV)",
        step_type="selection",
        timestamp=created_at,
        duration_s=None,
        artifacts={},
        metadata={
            "candidates_evaluated": len(result.all_results or {}),
            "winner": result.model_name,
            "all_scores": {
                k: round(float(v["score"] if isinstance(v, dict) else v), 4)
                for k, v in (result.all_results or {}).items()
            },
        },
    ))

    # ── Step 4: Training ──────────────────────────────────────────────────
    steps.append(LineageStep(
        step_name=f"Train {result.model_name}",
        step_type="training",
        timestamp=created_at,
        duration_s=result.times.training if result.times else None,
        artifacts={"model": "model.joblib"},
        metadata={"model_class": result.model_name},
    ))

    # ── Step 5: Evaluation ────────────────────────────────────────────────
    score_val = None
    try:
        score_val = round(float(result.score), 4)
    except Exception:
        pass

    steps.append(LineageStep(
        step_name="Evaluate Model",
        step_type="evaluation",
        timestamp=created_at,
        duration_s=None,
        artifacts={},
        metadata={
            "score": score_val,
            "problem_type": result.problem_type,
        },
    ))

    total_duration = result.times.total if result.times else 0.0

    return PipelineLineage(
        lineage_id=lineage_id,
        model_name=result.model_name,
        problem_type=result.problem_type,
        steps=steps,
        created_at=created_at,
        total_duration_s=round(total_duration, 3),
    )
