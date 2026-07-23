"""
test_lookup.py — Unit tests for lookup.py functional utilities (Story 3.3).
"""

import pytest

from kiteml.exceptions import (
    KML_T002,
    get_error_definition,
    lookup,
    search_errors,
)


def test_get_error_definition():
    defn = get_error_definition(KML_T002)

    assert defn is not None
    assert defn.code == KML_T002
    assert defn.name == "Target Column Not Found"


def test_lookup_by_code():
    res = lookup(code=KML_T002)

    assert len(res) == 1
    assert res[0].code == KML_T002


def test_lookup_by_category_and_severity():
    res = lookup(category="Dataset", severity="ERROR")

    assert len(res) >= 8
    assert all(d.category == "Dataset" for d in res)


def test_search_errors():
    res = search_errors("target")

    assert len(res) > 0
    codes = [d.code for d in res]
    assert KML_T002 in codes
