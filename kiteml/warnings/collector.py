"""
collector.py — WarningCollector container for KiteML workflows.
"""

from typing import Any

from kiteml.warnings.base import KiteMLWarning
from kiteml.warnings.policy import WarningPolicy
from kiteml.warnings.severity import WarningSeverity


class WarningCollector:
    """
    Central collector for non-fatal warnings emitted across KiteML modules.
    """

    def __init__(self, policy: WarningPolicy | None = None) -> None:
        self.policy = policy or WarningPolicy()
        self._warnings: list[KiteMLWarning] = []

    @property
    def warnings(self) -> list[KiteMLWarning]:
        """List of collected warnings."""
        return list(self._warnings)

    def add(self, warning: KiteMLWarning) -> bool:
        """
        Process and add a warning to the collection.

        Returns True if warning was added, False if ignored or duplicate.
        """
        processed = self.policy.process(warning)
        if processed is None:
            return False

        # Deduplication check
        for w in self._warnings:
            if w.code == processed.code and w.message == processed.message:
                return False

        self._warnings.append(processed)
        return True

    def warn(
        self,
        message: str,
        code: str = "KML-W-000",
        severity: WarningSeverity | str = WarningSeverity.MEDIUM,
        recommendation: str | None = None,
        category: str = "General",
        context: dict[str, Any] | None = None,
        source: str | None = None,
        warning_class: type[KiteMLWarning] = KiteMLWarning,
    ) -> None:
        """Helper to create and add a warning."""
        w = warning_class(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            category=category,
            context=context,
            source=source,
        )
        self.add(w)

    def get_by_category(self, category: str) -> list[KiteMLWarning]:
        """Filter warnings by category."""
        cat_clean = category.lower().strip()
        return [w for w in self._warnings if w.category.lower() == cat_clean]

    def get_by_severity(self, severity: WarningSeverity | str) -> list[KiteMLWarning]:
        """Filter warnings by severity level."""
        sev_str = severity.value if isinstance(severity, WarningSeverity) else str(severity).upper()
        return [
            w
            for w in self._warnings
            if (w.severity.value if isinstance(w.severity, WarningSeverity) else str(w.severity).upper()) == sev_str
        ]

    def clear(self) -> None:
        """Clear all collected warnings."""
        self._warnings.clear()

    def __len__(self) -> int:
        return len(self._warnings)
