"""serving/response_schema.py — Pydantic response schemas (optional dep)."""

try:
    from typing import Any, Dict, List, Optional

    from pydantic import BaseModel

    class PredictResponse(BaseModel):
        predictions: list[Any]
        probabilities: list[dict[str, float]] | None = None
        n_rows: int
        model: str
        problem_type: str
        latency_ms: float

    class HealthResponse(BaseModel):
        status: str
        uptime_s: float

    class SchemaResponse(BaseModel):
        required_features: list[str]
        problem_type: str
        n_features: int

except ImportError:
    pass
