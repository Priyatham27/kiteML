"""
validation_manager.py — Validator sequence manager for KiteML.
"""

from typing import Any

from kiteml.validation.dataset_validator import DatasetValidator
from kiteml.validation.quality_validator import QualityValidator
from kiteml.validation.schema_validator import SchemaValidator
from kiteml.validation.target_validator import TargetValidator
from kiteml.validation.validator import BaseValidator


class ValidationManager:
    """
    Manages registration, ordering, enabling, and disabling of validators.
    """

    def __init__(self, validators: list[BaseValidator] | None = None) -> None:
        if validators is None:
            validators = [
                DatasetValidator(),
                TargetValidator(),
                SchemaValidator(),
                QualityValidator(),
            ]
        self._validators: list[BaseValidator] = list(validators)
        self._disabled: set[str] = set()

    def register_validator(self, validator: BaseValidator, position: int | None = None) -> None:
        """
        Register a new validator into the manager.

        Parameters
        ----------
        validator : BaseValidator
            Validator instance to register.
        position : int, optional
            0-indexed position in sequence. Appended if None.
        """
        if not isinstance(validator, BaseValidator):
            raise TypeError(f"Expected subclass of BaseValidator, got {type(validator)}")

        # Replace existing if registered
        self._validators = [v for v in self._validators if v.name != validator.name]

        if position is None or position >= len(self._validators):
            self._validators.append(validator)
        else:
            self._validators.insert(max(0, position), validator)

    def enable_validator(self, name: str) -> None:
        """Enable a previously disabled validator by name."""
        self._disabled.discard(name)

    def disable_validator(self, name: str) -> None:
        """Disable a validator by name."""
        self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        """Return True if validator name is currently enabled."""
        return name not in self._disabled

    def get_validators(self) -> list[BaseValidator]:
        """Return list of all currently enabled validators in execution order."""
        return [v for v in self._validators if v.name not in self._disabled]
