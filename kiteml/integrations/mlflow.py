"""
integrations/mlflow.py — MLflow tracking integration for KiteML.
"""
from typing import Any, Dict
import os

def setup_mlflow(experiment_name: str, tracking_uri: str = None) -> bool:
    """Initialize MLflow integration."""
    try:
        import mlflow
    except ImportError:
        return False

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    return True


def log_run(result: Any, tags: Dict[str, str] = None) -> None:
    """Log a KiteML Result to the active MLflow run."""
    try:
        import mlflow
    except ImportError:
        return

    with mlflow.start_run():
        # Log Params
        mlflow.log_param("model_name", result.model_name)
        mlflow.log_param("problem_type", result.problem_type)
        
        # Log Metrics
        metrics_dict = result.metrics if isinstance(result.metrics, dict) else result.metrics.__dict__
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
                
        # Log Tags
        if tags:
            mlflow.set_tags(tags)
            
        # Log Model artifact
        bundle_path = f"{result.model_name}_mlflow_export.kiteml"
        result.package(bundle_path)
        mlflow.log_artifact(bundle_path)
