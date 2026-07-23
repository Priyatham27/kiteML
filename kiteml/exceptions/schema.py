"""
schema.py — Schema error exception for KiteML.
"""

from typing import Any

from kiteml.exceptions.base import KiteMLError
from kiteml.exceptions.codes import KML_ERR_SCHEMA
from kiteml.exceptions.context import ErrorContext


class SchemaError(KiteMLError):
    """Exception raised for feature schema, data type, or column structure issues."""

    def __init__(
        self,
        message: str,
        error_code: str = KML_ERR_SCHEMA,
        severity: str = "error",
        suggestion: str | None = None,
        context: ErrorContext | dict[str, Any] | None = None,
        details: str | None = None,
        timestamp: float | None = None,
        help_url: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            suggestion=suggestion,
            context=context,
            details=details,
            timestamp=timestamp,
            help_url=help_url,
        )
