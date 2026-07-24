"""
test_dag.py — Unit tests for PipelineDAG (Story 4.4).
"""

import pytest

from kiteml.pipeline import DatetimeStage, MissingValueStage, PipelineDAG, ScalingStage


def test_pipeline_dag_topological_sort():
    dag = PipelineDAG()
    dag.add_stage(ScalingStage(), depends_on=["MissingValueStage"])
    dag.add_stage(MissingValueStage())
    dag.add_stage(DatetimeStage())

    ordered = dag.topological_sort()
    names = [s.name for s in ordered]

    assert names.index("MissingValueStage") < names.index("ScalingStage")
