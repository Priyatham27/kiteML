"""
test_manifest.py — Unit tests for PipelineManifest (Story 4.5).
"""

import pytest

from kiteml.serialization import PipelineManifest


def test_pipeline_manifest_json_roundtrip():
    manifest = PipelineManifest(
        checksum="abc123sha",
        feature_count=15,
        target_name="price",
        stage_names=["MissingValueStage", "ScalingStage"],
    )

    json_str = manifest.to_json()
    reconstructed = PipelineManifest.from_json(json_str)

    assert reconstructed.checksum == "abc123sha"
    assert reconstructed.feature_count == 15
    assert reconstructed.target_name == "price"
    assert reconstructed.stage_names == ["MissingValueStage", "ScalingStage"]
