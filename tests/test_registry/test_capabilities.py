"""
test_capabilities.py — Unit tests for CapabilityAnalyzer (Story 5.2 Flagship Feature).
"""

import pandas as pd
import pytest

from kiteml.registry import CapabilityAnalyzer, model_registry


def test_capability_analyzer_ranking():
    df = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100),
            "target": [0, 1] * 50,
        }
    )

    ranked = model_registry.rank_models_for_dataset(df, target_name="target", task_type="binary_classification")

    assert len(ranked) > 0
    top_provider, top_score = ranked[0]
    assert top_score > 0.0
