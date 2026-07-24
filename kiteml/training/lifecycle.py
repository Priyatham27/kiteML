"""
lifecycle.py — TrainingLifecycle tracker for managing training state transitions in KiteML.
"""

import datetime
from typing import Any

from kiteml.training.state import TrainingState


class TrainingLifecycle:
    """
    Tracks state transitions and execution events during a training run.
    """

    def __init__(self) -> None:
        self.current_state: TrainingState = TrainingState.CREATED
        self.history: list[dict[str, Any]] = []
        self._record(TrainingState.CREATED)

    def transition_to(self, state: TrainingState, details: str | None = None) -> None:
        """Transition lifecycle to a new state."""
        self.current_state = state
        self._record(state, details=details)

    def _record(self, state: TrainingState, details: str | None = None) -> None:
        self.history.append(
            {
                "state": state.value,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "details": details or "",
            }
        )
