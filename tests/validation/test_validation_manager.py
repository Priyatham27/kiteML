"""
test_validation_manager.py — Unit tests for ValidationManager (Story 2.6).
"""

import pytest

from kiteml.validation.dataset_validator import DatasetValidator
from kiteml.validation.quality_validator import QualityValidator
from kiteml.validation.schema_validator import SchemaValidator
from kiteml.validation.target_validator import TargetValidator
from kiteml.validation.validation_manager import ValidationManager


def test_default_validators_and_ordering():
    manager = ValidationManager()
    validators = manager.get_validators()

    assert len(validators) == 4
    assert validators[0].name == "DatasetValidator"
    assert validators[1].name == "TargetValidator"
    assert validators[2].name == "SchemaValidator"
    assert validators[3].name == "QualityValidator"


def test_enable_disable_validator():
    manager = ValidationManager()
    manager.disable_validator("QualityValidator")

    validators = manager.get_validators()
    assert len(validators) == 3
    assert not any(v.name == "QualityValidator" for v in validators)

    manager.enable_validator("QualityValidator")
    validators = manager.get_validators()
    assert len(validators) == 4


def test_register_custom_position():
    manager = ValidationManager()
    custom_val = DatasetValidator()
    manager.register_validator(custom_val, position=0)

    validators = manager.get_validators()
    assert validators[0].name == "DatasetValidator"
