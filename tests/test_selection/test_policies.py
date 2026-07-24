"""
test_policies.py — Unit tests for SelectionPolicy (Story 5.5).
"""

import pytest

from kiteml.selection import SelectionPolicy


def test_selection_policy_weights():
    sp = SelectionPolicy()

    bal = sp.get_weights("balanced")
    assert bal["performance"] == 0.50

    acc = sp.get_weights("accuracy")
    assert acc["performance"] == 0.85

    fast = sp.get_weights("fast_inference")
    assert fast["speed"] == 0.50
