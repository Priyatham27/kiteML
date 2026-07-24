"""
validator.py — PipelineValidator consistency checker for KiteML transformation pipeline.
"""

from typing import Any


class PipelineValidator:
    """
    Validates blueprint consistency and stage dependencies prior to pipeline execution.
    """

    def validate_blueprints(
        self,
        preprocessing_blueprint: Any | None = None,
        engineering_blueprint: Any | None = None,
        selection_blueprint: Any | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate blueprint objects.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_warning_or_error_messages)
        """
        errors: list[str] = []

        if selection_blueprint:
            target = getattr(selection_blueprint, "target_name", None)
            removed = getattr(selection_blueprint, "removed_features", [])
            if target and target in removed:
                errors.append(f"Invalid selection blueprint: Target column '{target}' is marked for removal!")

        return (len(errors) == 0, errors)
