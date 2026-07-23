"""
catalog.py — ErrorCatalog static interface for KiteML.
"""

from typing import Any

from kiteml.exceptions.metadata import ErrorDefinition
from kiteml.exceptions.registry import ErrorRegistry, global_error_registry


class ErrorCatalog:
    """
    Centralized Error Catalog interface for querying and retrieving ErrorDefinition metadata.
    """

    _registry: ErrorRegistry = global_error_registry

    @classmethod
    def get(cls, code: str) -> ErrorDefinition | None:
        """
        Retrieve ErrorDefinition by error code string (e.g. 'KML-T002').

        Parameters
        ----------
        code : str
            Error code.

        Returns
        -------
        ErrorDefinition or None
        """
        return cls._registry.get(code)

    @classmethod
    def find(
        cls,
        category: str | None = None,
        severity: str | None = None,
        search: str | None = None,
    ) -> list[ErrorDefinition]:
        """
        Search and filter catalog error definitions by category, severity, or keyword search.

        Parameters
        ----------
        category : str, optional
            Filter by domain category ('Dataset', 'Target', etc.).
        severity : str, optional
            Filter by severity ('ERROR', 'WARNING', etc.).
        search : str, optional
            Keyword search against code, name, template, or tags.

        Returns
        -------
        list of ErrorDefinition
        """
        results = cls._registry.all_definitions()

        if category:
            cat_clean = category.lower().strip()
            results = [d for d in results if d.category.lower() == cat_clean]

        if severity:
            sev_clean = severity.upper().strip()
            results = [d for d in results if d.severity.upper() == sev_clean]

        if search:
            q = search.lower().strip()
            filtered = []
            for d in results:
                match_code = q in d.code.lower()
                match_name = q in d.name.lower()
                match_msg = q in d.message_template.lower()
                match_tags = any(q in t.lower() for t in d.tags)
                if match_code or match_name or match_msg or match_tags:
                    filtered.append(d)
            results = filtered

        return results

    @classmethod
    def categories(cls) -> list[str]:
        """Return list of all registered error categories."""
        return cls._registry.categories()

    @classmethod
    def all_definitions(cls) -> list[ErrorDefinition]:
        """Return all registered ErrorDefinition objects."""
        return cls._registry.all_definitions()
