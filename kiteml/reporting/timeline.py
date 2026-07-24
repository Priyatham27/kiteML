"""
timeline.py — Interactive Execution Replay and TransformationTimeline models for KiteML.
"""

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReplayEvent:
    """
    Event entry in the pipeline execution replay log.
    """

    stage_name: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    duration_ms: float = 0.0
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    transformations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return asdict(self)


@dataclass
class TransformationTimeline:
    """
    Chronological collection of pipeline execution replay events.
    """

    events: list[ReplayEvent] = field(default_factory=list)

    def add_event(self, event: ReplayEvent) -> None:
        """Record a stage execution event."""
        self.events.append(event)

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all events to a list of dictionaries."""
        return [e.to_dict() for e in self.events]
