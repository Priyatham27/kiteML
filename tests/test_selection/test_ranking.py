"""
test_ranking.py — Unit tests for RankingEngine (Story 5.5).
"""

import pytest

from kiteml.selection import RankingEngine


def test_ranking_engine_sorting():
    candidates = [
        {"name": "M1", "composite_score": 75.0, "benchmark": {"inference_latency_ms": 10.0}},
        {"name": "M2", "composite_score": 95.0, "benchmark": {"inference_latency_ms": 1.0}},
    ]

    ranked = RankingEngine().rank_candidates(candidates, policy="balanced")

    assert ranked[0]["name"] == "M2"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["name"] == "M1"
    assert ranked[1]["rank"] == 2
