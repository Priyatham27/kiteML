"""
test_workflow.py — Unit tests for WorkflowGraph (Story 4.7).
"""

import pandas as pd
import pytest

from kiteml.orchestration import OrchestrationContext, WorkflowGraph


def test_workflow_graph_run():
    df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "target": [0, 1, 0]})
    ctx = OrchestrationContext(dataset=df, target_name="target")

    graph = WorkflowGraph()
    res_ctx = graph.run(ctx)

    assert res_ctx.transformed_df is not None
    assert res_ctx.pipeline is not None
    assert res_ctx.report is not None
