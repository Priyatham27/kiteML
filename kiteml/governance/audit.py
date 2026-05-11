"""
audit.py — Append-only audit log for KiteML governance.

Logs every significant operation: predictions, packaging, deployments,
drift alerts, and retraining events to a local JSONL file.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

_DEFAULT_AUDIT_LOG = os.path.join(os.path.expanduser("~"), ".kiteml", "audit.jsonl")


@dataclass
class AuditEntry:
    """A single audit log entry."""

    entry_id: str
    event_type: str  # "prediction" | "packaging" | "deployment" | "drift_alert" | "retrain"
    model_name: str
    timestamp: str
    actor: str  # "system" | "user:<name>"
    details: Dict[str, Any]
    severity: str  # "info" | "warning" | "critical"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AuditLogger:
    """
    Append-only audit logger writing to a JSONL file.

    Parameters
    ----------
    log_path : str
        Path to the audit log file. Defaults to ``~/.kiteml/audit.jsonl``.
    model_name : str
        Model identifier for all log entries.
    actor : str
        Who is performing operations. Default ``'system'``.
    """

    def __init__(
        self,
        model_name: str,
        log_path: Optional[str] = None,
        actor: str = "system",
    ):
        self.model_name = model_name
        self.log_path = log_path or _DEFAULT_AUDIT_LOG
        self.actor = actor
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def _write(self, entry: AuditEntry) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def _log(self, event_type: str, details: Dict, severity: str = "info") -> AuditEntry:
        entry = AuditEntry(
            entry_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            model_name=self.model_name,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            actor=self.actor,
            details=details,
            severity=severity,
        )
        self._write(entry)
        return entry

    def log_prediction(self, n_rows: int, latency_ms: float) -> AuditEntry:
        return self._log("prediction", {"n_rows": n_rows, "latency_ms": latency_ms})

    def log_packaging(self, bundle_path: str, bundle_id: str) -> AuditEntry:
        return self._log("packaging", {"bundle_path": bundle_path, "bundle_id": bundle_id})

    def log_deployment(self, host: str, port: int) -> AuditEntry:
        return self._log("deployment", {"host": host, "port": port})

    def log_drift_alert(self, psi: float, drifted_features: list, severity: str = "warning") -> AuditEntry:
        return self._log(
            "drift_alert",
            {"psi": psi, "drifted_features": drifted_features},
            severity=severity,
        )

    def log_retrain(self, reason: str, new_score: Optional[float] = None) -> AuditEntry:
        return self._log(
            "retrain",
            {"reason": reason, "new_score": new_score},
            severity="info",
        )

    def read_log(self, last_n: Optional[int] = None) -> list:
        """Read and parse the audit log."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries[-last_n:] if last_n else entries
