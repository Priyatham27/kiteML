"""
lookup.py — Functional error lookup utilities for KiteML.
"""

from typing import Any

from kiteml.exceptions.catalog import ErrorCatalog
from kiteml.exceptions.metadata import ErrorDefinition


def lookup(
    code: str | None = None,
    category: str | None = None,
    severity: str | None = None,
    search: str | None = None,
) -> list[ErrorDefinition]:
    """
    Search and lookup error definitions in the ErrorCatalog.

    Parameters
    ----------
    code : str, optional
        Exact error code (e.g. 'KML-T002').
    category : str, optional
        Domain category string.
    severity : str, optional
        Severity level string.
    search : str, optional
        Keyword query.

    Returns
    -------
    list of ErrorDefinition
    """
    if code:
        defn = ErrorCatalog.get(code)
        return [defn] if defn else []

    return ErrorCatalog.find(category=category, severity=severity, search=search)


def get_error_definition(code: str) -> ErrorDefinition | None:
    """Retrieve ErrorDefinition by error code."""
    return ErrorCatalog.get(code)


def search_errors(query: str) -> list[ErrorDefinition]:
    """Search catalog error definitions by keyword."""
    return ErrorCatalog.find(search=query)
