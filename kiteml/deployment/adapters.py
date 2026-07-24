"""
adapters.py — DeploymentAdapter abstract base class for KiteML deployment subsystem.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DeploymentAdapter(ABC):
    """
    Abstract base class for all deployment export adapters in KiteML.
    """

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Return unique adapter name identifier."""
        pass

    @abstractmethod
    def export(
        self,
        model: Any,
        output_dir: Path | str,
        pipeline: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export trained solution for target deployment environment."""
        pass
