"""
test_workflow_graph.py — Unit tests for MLWorkflowGraph (Story 5.8 Flagship Feature).
"""

import pandas as pd
import pytest

from kiteml.ml import MLContext, MLWorkflowGraph


def test_ml_workflow_graph_execution():
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "income": [50, 60, 70, 80, 90, 100, 110, 120],
            "target": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )

    ctx = MLContext(dataframe=df, target_column="target")
    graph = MLWorkflowGraph()

    executed_ctx = graph.execute(ctx, policy="balanced")

    assert executed_ctx.problem_type in ("classification", "binary")
    assert executed_ctx.best_model is not None
    assert executed_ctx.pipeline_result is not None
