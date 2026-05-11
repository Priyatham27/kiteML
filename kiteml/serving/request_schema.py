"""serving/request_schema.py — Pydantic request schemas (optional dep)."""

try:
    from typing import Any, Dict, List, Optional

    from pydantic import BaseModel

    class SinglePredictRequest(BaseModel):
        data: dict[str, Any]
        return_probabilities: bool = True

    class BatchPredictRequest(BaseModel):
        data: list[dict[str, Any]]
        return_probabilities: bool = True
        chunk_size: int = 100

except ImportError:
    pass  # Pydantic not required at import time
