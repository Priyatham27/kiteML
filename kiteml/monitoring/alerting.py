"""
alerting.py — Rule-based alerting system for KiteML monitoring.

Defines alert rules and fires notifications (console, callback, or file)
when drift, anomaly, or performance thresholds are exceeded.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class AlertRule:
    """A single alert rule definition."""

    name: str
    metric: str  # e.g. "psi", "p99_latency_ms", "anomaly_ratio"
    threshold: float
    operator: str  # ">" | ">=" | "<" | "<="
    severity: str  # "info" | "warning" | "critical"
    message_template: str


@dataclass
class Alert:
    """A fired alert instance."""

    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: str
    message: str
    fired_at: str


class AlertEngine:
    """
    Evaluates alert rules against monitoring metrics and fires notifications.

    Parameters
    ----------
    on_alert : callable, optional
        Custom callback called with each fired Alert.
        Signature: ``fn(alert: Alert) -> None``.
    log_file : str, optional
        If provided, append alert JSON to this file.
    """

    # Built-in default rules
    DEFAULT_RULES: list[AlertRule] = [
        AlertRule(
            "high_drift",
            "psi",
            0.2,
            ">",
            "critical",
            "Data drift PSI={value:.3f} exceeds threshold {threshold}. Consider retraining.",
        ),
        AlertRule(
            "moderate_drift", "psi", 0.1, ">", "warning", "Moderate drift detected PSI={value:.3f}. Monitor closely."
        ),
        AlertRule(
            "high_anomaly_ratio",
            "anomaly_ratio",
            0.05,
            ">",
            "warning",
            "Anomaly ratio {value:.1%} exceeds {threshold:.1%} of production inputs.",
        ),
        AlertRule(
            "high_p99_latency",
            "p99_latency_ms",
            500.0,
            ">",
            "warning",
            "p99 latency {value:.0f}ms exceeds threshold {threshold:.0f}ms.",
        ),
        AlertRule(
            "low_confidence",
            "avg_confidence",
            0.6,
            "<",
            "warning",
            "Average confidence {value:.2%} below threshold {threshold:.2%}.",
        ),
    ]

    def __init__(
        self,
        rules: Optional[list[AlertRule]] = None,
        on_alert: Optional[Callable[[Alert], None]] = None,
        log_file: Optional[str] = None,
    ):
        self.rules = rules if rules is not None else list(self.DEFAULT_RULES)
        self.on_alert = on_alert or self._default_on_alert
        self.log_file = log_file
        self.fired_alerts: list[Alert] = []

    @staticmethod
    def _default_on_alert(alert: Alert) -> None:
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(alert.severity, "⚠️")
        print(f"{icon}  [{alert.severity.upper()}] {alert.message}")

    def _evaluate(self, rule: AlertRule, value: float) -> bool:
        ops = {">": float.__gt__, ">=": float.__ge__, "<": float.__lt__, "<=": float.__le__}
        op = ops.get(rule.operator)
        return op(value, rule.threshold) if op else False

    def check(self, metrics: dict[str, float]) -> list[Alert]:
        """
        Evaluate all rules against the given metrics dict.

        Parameters
        ----------
        metrics : dict
            Keys are metric names (must match ``AlertRule.metric``),
            values are current measurements.

        Returns
        -------
        list of Alert
            All fired alerts.
        """
        fired: list[Alert] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for rule in self.rules:
            if rule.metric not in metrics:
                continue
            value = metrics[rule.metric]
            if self._evaluate(rule, value):
                msg = rule.message_template.format(value=value, threshold=rule.threshold)
                alert = Alert(
                    rule_name=rule.name,
                    metric=rule.metric,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=msg,
                    fired_at=now,
                )
                fired.append(alert)
                self.fired_alerts.append(alert)
                self.on_alert(alert)

                if self.log_file:
                    import json
                    import os

                    os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(alert.__dict__) + "\n")

        return fired

    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.rules.append(rule)

    def summary(self) -> str:
        """Return a summary of all fired alerts."""
        if not self.fired_alerts:
            return "✅ No alerts fired."
        counts = {"info": 0, "warning": 0, "critical": 0}
        for a in self.fired_alerts:
            counts[a.severity] = counts.get(a.severity, 0) + 1
        return f"🚨 {counts['critical']} critical | " f"⚠️ {counts['warning']} warning | " f"ℹ️ {counts['info']} info"
