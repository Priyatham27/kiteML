"""
test_timeline.py — Unit tests for TransformationTimeline (Story 4.6).
"""

import pytest

from kiteml.reporting import ReplayEvent, TransformationTimeline


def test_transformation_timeline_replay():
    timeline = TransformationTimeline()
    event = ReplayEvent(
        stage_name="ScalingStage",
        duration_ms=2.5,
        input_shape=(10, 5),
        output_shape=(10, 5),
    )
    timeline.add_event(event)

    event_list = timeline.to_list()
    assert len(event_list) == 1
    assert event_list[0]["stage_name"] == "ScalingStage"
    assert event_list[0]["duration_ms"] == 2.5
