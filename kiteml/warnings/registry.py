"""
registry.py — WarningRegistry catalog for KiteML warnings.
"""

from dataclasses import asdict, dataclass
from typing import Any

import kiteml.warnings.categories as cat
from kiteml.warnings.severity import WarningSeverity


@dataclass
class WarningDefinition:
    """Metadata definition for cataloged warnings."""

    code: str
    name: str
    category: str
    severity: WarningSeverity | str = WarningSeverity.MEDIUM
    message_template: str = ""
    default_recommendation: str | None = None
    documentation_slug: str | None = None
    recoverable: bool = True

    def format_message(self, **kwargs: Any) -> str:
        """Format message template with kwargs."""
        if not self.message_template:
            return self.name
        try:
            return self.message_template.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            return self.message_template

    def to_dict(self) -> dict[str, Any]:
        """Serialize WarningDefinition to dictionary."""
        d = asdict(self)
        if isinstance(self.severity, WarningSeverity):
            d["severity"] = self.severity.value
        return d


class WarningRegistry:
    """Catalog repository for standard warning definitions."""

    def __init__(self) -> None:
        self._catalog: dict[str, WarningDefinition] = {}
        self._populate_default_catalog()

    def register(self, definition: WarningDefinition) -> None:
        """Register a warning definition."""
        self._catalog[definition.code] = definition

    def get(self, code: str) -> WarningDefinition | None:
        """Retrieve warning definition by code string."""
        return self._catalog.get(code)

    def all_definitions(self) -> list[WarningDefinition]:
        """Return all registered warning definitions."""
        return list(self._catalog.values())

    def _populate_default_catalog(self) -> None:
        """Populate initial warning catalog."""
        defs = [
            WarningDefinition(
                code="KML-W-D001",
                name="High Missing Values",
                category=cat.DATASET,
                severity=WarningSeverity.MEDIUM,
                message_template="High missing values in column '{col}' ({ratio:.0%}).",
                default_recommendation="Consider imputation or feature removal.",
                documentation_slug="kml-w-d001",
            ),
            WarningDefinition(
                code="KML-W-D002",
                name="Duplicate Rows",
                category=cat.DATASET,
                severity=WarningSeverity.LOW,
                message_template="Dataset contains {count} duplicate rows.",
                default_recommendation="Review and remove duplicate rows if unintentional.",
                documentation_slug="kml-w-d002",
            ),
            WarningDefinition(
                code="KML-W-S001",
                name="Constant Feature",
                category=cat.SCHEMA,
                severity=WarningSeverity.HIGH,
                message_template="Feature '{col}' has 0 variance (constant).",
                default_recommendation="Remove constant feature prior to modeling.",
                documentation_slug="kml-w-s001",
            ),
            WarningDefinition(
                code="KML-W-S002",
                name="High Cardinality",
                category=cat.SCHEMA,
                severity=WarningSeverity.MEDIUM,
                message_template="Feature '{col}' has high cardinality ({count} unique values).",
                default_recommendation="Apply target encoding or frequency encoding.",
                documentation_slug="kml-w-s002",
            ),
            WarningDefinition(
                code="KML-W-V001",
                name="Dataset Health Below Recommendation",
                category=cat.VALIDATION,
                severity=WarningSeverity.HIGH,
                message_template="Dataset health score ({score}/100) is below recommended threshold.",
                default_recommendation="Inspect quality report and address high severity issues.",
                documentation_slug="kml-w-v001",
            ),
            WarningDefinition(
                code="KML-W-M001",
                name="Slow Convergence",
                category=cat.TRAINING,
                severity=WarningSeverity.LOW,
                message_template="Model '{model}' convergence was slow.",
                default_recommendation="Increase max_iter or scale numerical features.",
                documentation_slug="kml-w-m001",
            ),
            WarningDefinition(
                code="KML-W-I001",
                name="Unknown Inference Category",
                category=cat.PREDICTION,
                severity=WarningSeverity.LOW,
                message_template="Unknown category '{val}' in column '{col}' during inference.",
                default_recommendation="Value was imputed with default category.",
                documentation_slug="kml-w-i001",
            ),
            WarningDefinition(
                code="KML-W-DP001",
                name="Large Deployment Bundle",
                category=cat.DEPLOYMENT,
                severity=WarningSeverity.MEDIUM,
                message_template="Deployment bundle size ({size_mb:.1f} MB) is large.",
                default_recommendation="Consider model pruning or compression.",
                documentation_slug="kml-w-dp001",
            ),
        ]
        for d in defs:
            self.register(d)


global_warning_registry = WarningRegistry()
