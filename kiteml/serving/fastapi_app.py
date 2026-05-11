"""
fastapi_app.py — FastAPI application factory for KiteML model serving.

Creates a FastAPI app from a Result object or RealtimeInferenceEngine.
Optional dependency: pip install fastapi uvicorn pydantic
"""

import time
from typing import Any, Optional


def create_app(result: Any):
    """Create FastAPI app from a KiteML Result."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn pydantic")

    import pandas as pd

    from kiteml.deployment.inference_guardrails import InferenceGuardrails

    _start = time.time()
    _counter = [0]
    _guardrails = InferenceGuardrails(result.feature_names or [])

    app = FastAPI(
        title=f"KiteML — {result.model_name}",
        description="Auto-generated KiteML inference API",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        data: list[dict[str, Any]]
        return_probabilities: bool = True

    class PredictResponse(BaseModel):
        predictions: list[Any]
        probabilities: Optional[list[dict[str, float]]] = None
        n_rows: int
        model: str
        problem_type: str
        latency_ms: float

    @app.get("/health")
    def health():
        return {"status": "healthy", "uptime_s": round(time.time() - _start, 1)}

    @app.get("/version")
    def version():
        return {
            "model": result.model_name,
            "problem_type": result.problem_type,
            "n_features": len(result.feature_names or []),
            "score": _safe_float(result.score),
        }

    @app.get("/schema")
    def schema():
        return {
            "required_features": list(result.feature_names or []),
            "problem_type": result.problem_type,
        }

    @app.get("/metrics")
    def metrics():
        return {
            "predictions_served": _counter[0],
            "uptime_s": round(time.time() - _start, 1),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        try:
            df = pd.DataFrame(req.data)
            present = [f for f in (result.feature_names or []) if f in df.columns]
            df = df[present] if present else df

            X = result.preprocessor.transform(df) if result.preprocessor else df.values
            preds = result.model.predict(X).tolist()

            probas = None
            if req.return_probabilities and hasattr(result.model, "predict_proba"):
                raw = result.model.predict_proba(X)
                classes = [str(c) for c in getattr(result.model, "classes_", range(raw.shape[1]))]
                probas = [dict(zip(classes, row.tolist())) for row in raw]

            _counter[0] += len(preds)
            return PredictResponse(
                predictions=preds,
                probabilities=probas,
                n_rows=len(preds),
                model=result.model_name,
                problem_type=result.problem_type,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def create_app_from_engine(engine: Any):
    """Create FastAPI app from a RealtimeInferenceEngine."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn pydantic")

    _start = time.time()
    app = FastAPI(title="KiteML Inference Server", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class PredictRequest(BaseModel):
        data: list[dict[str, Any]]

    @app.get("/health")
    def health():
        return {"status": "healthy", "uptime_s": round(time.time() - _start, 1)}

    @app.get("/schema")
    def schema():
        return {"required_features": engine.feature_names}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            results = [engine.predict(row) for row in req.data]
            return {
                "predictions": [r.prediction for r in results],
                "probabilities": [r.probabilities for r in results],
                "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None
