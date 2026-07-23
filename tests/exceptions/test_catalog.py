"""
test_catalog.py — Unit tests for ErrorCatalog (Story 3.3).
"""

import pytest

from kiteml.exceptions import KML_T002, ErrorCatalog, ErrorDefinition


def test_catalog_get_existing_code():
    defn = ErrorCatalog.get(KML_T002)

    assert defn is not None
    assert isinstance(defn, ErrorDefinition)
    assert defn.code == KML_T002
    assert defn.name == "Target Column Not Found"
    assert defn.category == "Target"
    assert "target" in defn.message_template.lower()
    assert defn.default_suggestion == "Choose one of the available columns."


def test_catalog_get_non_existent():
    assert ErrorCatalog.get("KML-NONEXISTENT") is None


def test_catalog_categories():
    cats = ErrorCatalog.categories()

    assert "Dataset" in cats
    assert "Target" in cats
    assert "Schema" in cats
    assert "Validation" in cats
    assert "Preprocessing" in cats
    assert "Training" in cats
    assert "Prediction" in cats
    assert "Deployment" in cats
    assert "CLI" in cats
    assert "Configuration" in cats


def test_catalog_find():
    target_errors = ErrorCatalog.find(category="Target")
    assert len(target_errors) >= 7

    error_severities = ErrorCatalog.find(severity="ERROR")
    assert len(error_severities) > 0

    search_res = ErrorCatalog.find(search="missing")
    assert len(search_res) > 0
