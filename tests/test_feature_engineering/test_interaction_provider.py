"""
test_interaction_provider.py — Unit tests for InteractionFEProvider (Story 4.2).
"""

import pandas as pd
import pytest

from kiteml.feature_engineering import FeatureImportancePredictor, FERuleEngine, InteractionFEProvider


def test_interaction_fe_provider():
    df = pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0],
            "quantity": [1, 2, 3],
        }
    )

    provider = InteractionFEProvider()
    rules = FERuleEngine()
    predictor = FeatureImportancePredictor()

    plans = provider.plan_features(df, None, rules, predictor)

    assert len(plans) == 1
    assert plans[0].generated_name == "price_x_quantity"
    assert "Multiplicative interaction between price and quantity" in plans[0].reasoning[0]
    assert plans[0].star_rating() == "★★★★★"
