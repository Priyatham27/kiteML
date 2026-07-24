"""
state.py — PipelineState execution progress model for KiteML.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """
    Tracks pipeline stage execution progress and diagnostic metrics.
    """

    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = True

    def mark_completed(self, stage_name: str) -> None:
        """Record stage as successfully completed."""
        self.completed_stages.append(stage_name)
        self.current_stage = stage_name

    def mark_skipped(self, stage_name: str) -> None:
        """Record stage as skipped."""
        self.skipped_stages.append(stage_name)

    def mark_failed(self, stage_name: str) -> None:
        """Record stage as failed."""
        self.failed_stages.append(stage_name)
        self.success = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "skipped_stages": self.skipped_stages,
            "failed_stages": self.failed_stages,
            "execution_time": self.execution_time,
            "success": self.success,
        }
