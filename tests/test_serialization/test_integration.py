"""
test_integration.py — Integration tests for save_pipeline and load_pipeline (Story 4.5).
"""

from pathlib import Path

import pandas as pd
import pytest

from kiteml.pipeline import TransformationPipeline
from kiteml.serialization import load_pipeline, save_pipeline


def test_save_load_pipeline_identical_transformation(tmp_path: Path):
    df_train = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0],
            "qty": [1, 2, 3, 4],
            "category": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )

    df_test = pd.DataFrame(
        {
            "price": [15.0, 25.0],
            "qty": [2, 3],
            "category": ["A", "B"],
            "target": [0, 1],
        }
    )

    p1 = TransformationPipeline()
    t1 = p1.fit_transform(df_train, target="target")
    assert not t1.empty

    kml_path = tmp_path / "pipeline.kml"
    p1.save(kml_path)

    p2 = TransformationPipeline.load(kml_path)
    t2_test = p2.transform(df_test)
    t1_test = p1.transform(df_test)

    pd.testing.assert_frame_equal(t1_test, t2_test)
