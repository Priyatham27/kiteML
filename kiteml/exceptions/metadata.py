"""
metadata.py — ErrorDefinition dataclass for catalog entries.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ErrorDefinition:
    """
    Metadata representation for a single entry in the KiteML Error Catalog.

    Attributes
    ----------
    code : str
        Unique error code string (e.g. 'KML-T002').
    name : str
        Human-readable title/name of the error.
    category : str
        Domain category ('Dataset', 'Target', 'Schema', 'Validation', etc.).
    severity : str
        Default severity ('ERROR', 'WARNING', 'CRITICAL', 'INFO').
    message_template : str
        Format string template for the error message.
    default_suggestion : str, optional
        Recommended action or fix suggestion.
    documentation_slug : str, optional
        Documentation URL slug for online reference.
    recoverable : bool
        True if workflow can recover or fallback automatically.
    tags : list of str
        Keywords/tags for search indexing.
    """

    code: str
    name: str
    category: str
    severity: str = "ERROR"
    message_template: str = ""
    default_suggestion: str | None = None
    documentation_slug: str | None = None
    recoverable: bool = False
    tags: list[str] = field(default_factory=list)

    def format_message(self, **kwargs: Any) -> str:
        """Format the message template with supplied kwargs."""
        if not self.message_template:
            return self.name
        try:
            return self.message_template.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            return self.message_template

    def to_dict(self) -> dict[str, Any]:
        """Serialize ErrorDefinition to dictionary."""
        return asdict(self)
