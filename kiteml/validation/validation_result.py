"""
validation_result.py — Standard result container returned by all KiteML validators.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from kiteml.validation.message import ValidationMessage
from kiteml.validation.severity import ValidationSeverity


@dataclass
class ValidationResult:
    """
    Standard outcome returned by a validator or rule execution.

    Attributes
    ----------
    validator_name : str
        Name of the validator that generated this result.
    messages : list[ValidationMessage]
        List of all validation messages (INFO, WARNING, ERROR, CRITICAL).
    statistics : dict[str, Any]
        Arbitrary metrics collected during validation (e.g., shape, memory).
    execution_time : float
        Time taken in seconds to run validation.
    """

    validator_name: str = "Validator"
    messages: list[ValidationMessage] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    @property
    def passed(self) -> bool:
        """True if no ERROR or CRITICAL messages exist."""
        return not self.has_errors()

    def has_errors(self) -> bool:
        """Check if any ERROR or CRITICAL messages were recorded."""
        return any(msg.severity >= ValidationSeverity.ERROR for msg in self.messages)

    def has_warnings(self) -> bool:
        """Check if any WARNING messages were recorded."""
        return any(msg.severity == ValidationSeverity.WARNING for msg in self.messages)

    @property
    def errors(self) -> list[ValidationMessage]:
        """List of ERROR and CRITICAL messages."""
        return [msg for msg in self.messages if msg.severity >= ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationMessage]:
        """List of WARNING messages."""
        return [msg for msg in self.messages if msg.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationMessage]:
        """List of INFO messages."""
        return [msg for msg in self.messages if msg.severity == ValidationSeverity.INFO]

    @property
    def criticals(self) -> list[ValidationMessage]:
        """List of CRITICAL messages."""
        return [msg for msg in self.messages if msg.severity == ValidationSeverity.CRITICAL]

    @property
    def summary(self) -> dict[str, int]:
        """Summary counts of messages grouped by severity."""
        counts = {
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        }
        for msg in self.messages:
            counts[msg.severity.value] = counts.get(msg.severity.value, 0) + 1
        return counts

    def add_message(self, message: ValidationMessage) -> None:
        """Append an existing ValidationMessage."""
        self.messages.append(message)

    def add_info(
        self,
        title: str,
        description: str,
        suggestion: str | None = None,
        rule_id: str | None = None,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add an INFO severity message."""
        self.add_message(
            ValidationMessage(
                severity=ValidationSeverity.INFO,
                title=title,
                description=description,
                suggestion=suggestion,
                rule_id=rule_id,
                code=code,
                context=context or {},
            )
        )

    def add_warning(
        self,
        title: str,
        description: str,
        suggestion: str | None = None,
        rule_id: str | None = None,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a WARNING severity message."""
        self.add_message(
            ValidationMessage(
                severity=ValidationSeverity.WARNING,
                title=title,
                description=description,
                suggestion=suggestion,
                rule_id=rule_id,
                code=code,
                context=context or {},
            )
        )

    def add_error(
        self,
        title: str,
        description: str,
        suggestion: str | None = None,
        rule_id: str | None = None,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add an ERROR severity message."""
        self.add_message(
            ValidationMessage(
                severity=ValidationSeverity.ERROR,
                title=title,
                description=description,
                suggestion=suggestion,
                rule_id=rule_id,
                code=code,
                context=context or {},
            )
        )

    def add_critical(
        self,
        title: str,
        description: str,
        suggestion: str | None = None,
        rule_id: str | None = None,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a CRITICAL severity message."""
        self.add_message(
            ValidationMessage(
                severity=ValidationSeverity.CRITICAL,
                title=title,
                description=description,
                suggestion=suggestion,
                rule_id=rule_id,
                code=code,
                context=context or {},
            )
        )

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """
        Merge another ValidationResult into a new combined ValidationResult.

        Parameters
        ----------
        other : ValidationResult

        Returns
        -------
        ValidationResult
        """
        name = f"{self.validator_name}+{other.validator_name}"
        merged_msgs = list(self.messages) + list(other.messages)
        merged_stats = {**self.statistics, **other.statistics}
        merged_time = round(self.execution_time + other.execution_time, 6)

        return ValidationResult(
            validator_name=name,
            messages=merged_msgs,
            statistics=merged_stats,
            execution_time=merged_time,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a structured dictionary."""
        return {
            "validator_name": self.validator_name,
            "passed": self.passed,
            "summary": self.summary,
            "execution_time": self.execution_time,
            "statistics": self.statistics,
            "messages": [msg.to_dict() for msg in self.messages],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result into a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
