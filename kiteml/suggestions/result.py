"""
result.py — Suggestion object model for KiteML suggestion engine.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Suggestion:
    """
    Structured recommendation returned by the SuggestionEngine.

    Attributes
    ----------
    title : str
        Short title of the recommendation (e.g. 'Use "Price" as target').
    description : str
        Detailed explanation of the suggested action.
    confidence : float
        Confidence score between 0.0 and 1.0 (e.g. 0.98 = 98%).
    category : str
        Category of suggestion ('Column', 'Target', 'Schema', etc.).
    source : str, optional
        Name of the suggestion provider that generated this recommendation.
    action : str, optional
        Concrete code or API action string (e.g. "target='Price'").
    why : list of str
        Explainable rationale bullet points explaining why this suggestion was made.
    metadata : dict
        Additional metadata key-value pairs.
    """

    title: str
    description: str
    confidence: float
    category: str = "General"
    source: str | None = None
    action: str | None = None
    why: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp confidence to [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def confidence_percentage(self) -> int:
        """Return confidence formatted as integer percentage (0–100)."""
        return int(round(self.confidence * 100))

    def to_dict(self) -> dict[str, Any]:
        """Serialize Suggestion object to a plain dictionary."""
        d = asdict(self)
        d["confidence_percentage"] = self.confidence_percentage
        return d

    def __str__(self) -> str:
        out = f"✓ {self.title} ({self.confidence_percentage}% confidence)\n  {self.description}"
        if self.action:
            out += f"\n  Action: {self.action}"
        if self.why:
            out += "\n  Why?"
            for reason in self.why:
                out += f"\n    • {reason}"
        return out
