"""
fastapi.py — FastAPIAdapter for generating FastAPI web application server scaffolds in KiteML.
"""

import pickle
from pathlib import Path
from typing import Any

from kiteml.deployment.adapters import DeploymentAdapter


class FastAPIAdapter(DeploymentAdapter):
    """
    Exports a production-ready FastAPI application server scaffold (`app.py`).
    """

    @property
    def adapter_name(self) -> str:
        return "fastapi"

    def export(
        self,
        model: Any,
        output_dir: Path | str,
        pipeline: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Export FastAPI server code and model artifacts.

        Parameters
        ----------
        model : Any
            Fitted estimator model.
        output_dir : Path | str
            Target output directory.
        pipeline : Any, optional
            Fitted pipeline instance.
        metadata : dict[str, Any], optional
            Export metadata.

        Returns
        -------
        Path
            Directory path containing exported app.py.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        if pipeline is not None:
            pipeline_file = out_path / "pipeline.pkl"
            with open(pipeline_file, "wb") as f:
                pickle.dump(pipeline, f)

        app_code = '''"""
Production FastAPI Model Service generated automatically by KiteML.
"""

import pickle
from pathlib import Path
from typing import Any, List, Dict
from fastapi import FastAPI, HTTPException

app = FastAPI(title="KiteML Production Model API", version="1.0.0")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
PIPELINE_PATH = BASE_DIR / "pipeline.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

pipeline = None
if PIPELINE_PATH.exists():
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "KiteML Model API"}


@app.post("/predict")
def predict(data: List[Dict[str, Any]]):
    import pandas as pd
    from kiteml.prediction import PredictionEngine

    try:
        df = pd.DataFrame(data)
        engine = PredictionEngine()
        res = engine.predict(model=model, dataframe=df, pipeline=pipeline)
        return {"predictions": res.predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''

        app_file = out_path / "app.py"
        with open(app_file, "w", encoding="utf-8") as f:
            f.write(app_code)

        return out_path
