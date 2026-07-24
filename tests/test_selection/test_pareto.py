"""
test_pareto.py — Unit tests for ParetoSelector (Story 5.5 Flagship Feature).
"""

import pytest

from kiteml.selection import ParetoSelector


def test_pareto_selector():
    candidates = [
        {
            "name": "RF",
            "composite_score": 90.0,
            "metrics": {"f1": 0.90},
            "benchmark": {"inference_latency_ms": 5.0, "model_size_kb": 100.0},
        },
        {
            "name": "Linear",
            "composite_score": 80.0,
            "metrics": {"f1": 0.80},
            "benchmark": {"inference_latency_ms": 0.5, "model_size_kb": 10.0},
        },
    ]

    p_front = ParetoSelector().find_pareto_frontier(candidates)

    assert p_front["best_overall"] == "RF"
    assert p_front["fastest_inference"] == "Linear"
    assert p_front["most_efficient_memory"] == "Linear"
