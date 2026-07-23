"""
warning_manager.py — WarningManager for centralized warning collection and policy handling.
"""

from typing import Any

from kiteml.warnings import (
    KiteMLWarning,
    WarningCollector,
    WarningFormatter,
    WarningPolicy,
    WarningReport,
    WarningSeverity,
)


class WarningManager:
    """
    Central manager for collecting, deduplicating, policy enforcement,
    and reporting non-fatal warnings across KiteML subsystems.
    """

    def __init__(
        self,
        collector: WarningCollector | None = None,
        policy: WarningPolicy | None = None,
        formatter: WarningFormatter | None = None,
    ) -> None:
        self.collector = collector or WarningCollector(policy=policy)
        self.formatter = formatter or WarningFormatter()

    @property
    def warnings(self) -> list[KiteMLWarning]:
        """List of collected warnings."""
        return self.collector.warnings

    def add(self, warning: KiteMLWarning) -> bool:
        """Add a warning through the policy engine."""
        return self.collector.add(warning)

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
        """Issue and record a warning."""
        self.collector.warn(
            message=message,
            code=code,
            severity=severity,
            recommendation=recommendation,
            category=category,
            context=context,
            source=source,
            warning_class=warning_class,
        )

    def get_report(self) -> WarningReport:
        """Generate WarningReport from collected warnings."""
        return WarningReport(warnings=self.warnings)

    def format_report(self, mode: str = "terminal") -> str:
        """Format warning report string."""
        return self.formatter.format(self.get_report(), mode=mode)

    def clear(self) -> None:
        """Clear collected warnings."""
        self.collector.clear()
