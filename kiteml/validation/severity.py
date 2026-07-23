"""
severity.py — Severity levels for KiteML validation checks.
"""

from enum import Enum


def _get_rank(val: object) -> int | None:
    if isinstance(val, ValidationSeverity):
        return val.rank
    if isinstance(val, str):
        try:
            return ValidationSeverity(val.lower()).rank
        except ValueError:
            return None
    return None


class ValidationSeverity(str, Enum):
    """
    Severity levels for validation messages and rules.

    Levels in increasing order of severity:
    INFO (1) -> WARNING (2) -> ERROR (3) -> CRITICAL (4)
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank used for ordering and severity threshold filtering."""
        _ranks = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 2,
            ValidationSeverity.ERROR: 3,
            ValidationSeverity.CRITICAL: 4,
        }
        return _ranks[self]

    def __lt__(self, other: object) -> bool:
        r2 = _get_rank(other)
        if r2 is None:
            return NotImplemented
        return self.rank < r2

    def __le__(self, other: object) -> bool:
        r2 = _get_rank(other)
        if r2 is None:
            return NotImplemented
        return self.rank <= r2

    def __gt__(self, other: object) -> bool:
        r2 = _get_rank(other)
        if r2 is None:
            return NotImplemented
        return self.rank > r2

    def __ge__(self, other: object) -> bool:
        r2 = _get_rank(other)
        if r2 is None:
            return NotImplemented
        return self.rank >= r2

    def __eq__(self, other: object) -> bool:
        r2 = _get_rank(other)
        if r2 is None:
            return False
        return self.rank == r2

    def __hash__(self) -> int:
        return hash(self.value)
