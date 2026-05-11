"""
integrations/wandb.py — Weights & Biases integration.
"""

from typing import Any


def setup_wandb(project_name: str, entity: str = None) -> bool:
    """Initialize W&B integration."""
    try:
        import wandb
    except ImportError:
        return False

    wandb.init(project=project_name, entity=entity)
    return True


def log_run_wandb(result: Any, tags: list = None) -> None:
    """Log a KiteML Result to W&B."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    config = {
        "model_name": result.model_name,
        "problem_type": result.problem_type,
    }
    wandb.config.update(config)

    metrics = result.metrics if isinstance(result.metrics, dict) else result.metrics.__dict__
    valid_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    wandb.log(valid_metrics)

    if tags:
        wandb.run.tags = tags + (wandb.run.tags or ())

    bundle_path = f"{result.model_name}_wandb_export.kiteml"
    result.package(bundle_path)

    artifact = wandb.Artifact(name=f"kiteml-{result.model_name}", type="model")
    artifact.add_dir(bundle_path)
    wandb.log_artifact(artifact)
