"""
recommendations.py — Master recommendation aggregator for KiteML Phase 2.

Collects findings from all intelligence modules and produces a single,
prioritized, non-redundant recommendation report.
"""

from dataclasses import dataclass
from typing import Optional

from kiteml.intelligence.cardinality_analyzer import CardinalityReport
from kiteml.intelligence.correlation_analyzer import CorrelationReport
from kiteml.intelligence.feature_recommender import FeatureRecommendationReport
from kiteml.intelligence.imbalance_detector import ImbalanceReport
from kiteml.intelligence.leakage_detector import LeakageReport
from kiteml.intelligence.memory_optimizer import MemoryReport
from kiteml.intelligence.outlier_detector import OutlierReport
from kiteml.intelligence.quality_analyzer import QualityReport, Severity


@dataclass
class MasterRecommendation:
    """One consolidated recommendation."""

    priority: str  # "critical" | "high" | "medium" | "low"
    category: str  # "leakage" | "quality" | "imbalance" | "features" | "memory" | ...
    message: str
    action: str | None = None


@dataclass
class MasterRecommendationReport:
    """All recommendations aggregated from Phase 2 intelligence modules."""

    recommendations: list[MasterRecommendation]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    overall_health: str  # "excellent" | "good" | "fair" | "poor"

    def by_priority(self, priority: str) -> list[MasterRecommendation]:
        return [r for r in self.recommendations if r.priority == priority]

    def print_report(self) -> None:
        W = 56
        print("\n" + "═" * W)
        print("  🪁  KiteML — Intelligence Recommendations")
        print("═" * W)
        print(f"  Overall Dataset Health: {self.overall_health.upper()}")
        print(
            f"  Critical: {self.critical_count}  High: {self.high_count}  "
            f"Medium: {self.medium_count}  Low: {self.low_count}"
        )
        print("─" * W)

        priority_icons = {"critical": "🚨", "high": "⚠️ ", "medium": "💡", "low": "ℹ️ "}
        for rec in self.recommendations:
            icon = priority_icons.get(rec.priority, "  ")
            print(f"  {icon} [{rec.category.upper()}] {rec.message}")
            if rec.action:
                print(f"     → {rec.action}")

        print("═" * W)


def build_recommendation_report(
    quality: QualityReport | None = None,
    leakage: LeakageReport | None = None,
    imbalance: ImbalanceReport | None = None,
    outliers: OutlierReport | None = None,
    features: FeatureRecommendationReport | None = None,
    correlations: CorrelationReport | None = None,
    cardinality: CardinalityReport | None = None,
    memory: MemoryReport | None = None,
) -> MasterRecommendationReport:
    """
    Build a unified, prioritized recommendation report from all intelligence modules.

    All parameters are optional — pass only what you've computed.
    """
    recs: list[MasterRecommendation] = []

    # ── Leakage (critical) ────────────────────────────────────────────────
    if leakage:
        for risk in leakage.risks:
            priority = "critical" if risk.risk_level == "critical" else "high"
            recs.append(
                MasterRecommendation(
                    priority=priority,
                    category="leakage",
                    message=f"'{risk.column}': {risk.reason}",
                    action="Remove this column before training.",
                )
            )

    # ── Quality errors ────────────────────────────────────────────────────
    if quality:
        for issue in quality.issues:
            if issue.severity == Severity.ERROR:
                recs.append(
                    MasterRecommendation(
                        priority="high",
                        category="quality",
                        message=issue.description,
                        action=issue.recommendation,
                    )
                )
            elif issue.severity == Severity.WARNING:
                recs.append(
                    MasterRecommendation(
                        priority="medium",
                        category="quality",
                        message=issue.description,
                        action=issue.recommendation,
                    )
                )

    # ── Imbalance ─────────────────────────────────────────────────────────
    if imbalance and imbalance.is_imbalanced:
        priority = "high" if imbalance.severity in ("severe", "extreme") else "medium"
        for msg in imbalance.recommendations:
            recs.append(
                MasterRecommendation(
                    priority=priority,
                    category="imbalance",
                    message=msg,
                )
            )

    # ── Feature recommendations ───────────────────────────────────────────
    if features:
        for feat_rec in features.recommendations:
            p = "high" if feat_rec.priority == "high" else "medium"
            recs.append(
                MasterRecommendation(
                    priority=p,
                    category="features",
                    message=feat_rec.reason,
                    action=feat_rec.impact,
                )
            )

    # ── Correlation / redundancy ──────────────────────────────────────────
    if correlations:
        for msg in correlations.recommendations:
            recs.append(
                MasterRecommendation(
                    priority="medium",
                    category="correlation",
                    message=msg,
                )
            )

    # ── Cardinality ───────────────────────────────────────────────────────
    if cardinality:
        for msg in cardinality.recommendations:
            recs.append(
                MasterRecommendation(
                    priority="medium",
                    category="cardinality",
                    message=msg,
                )
            )

    # ── Outliers ──────────────────────────────────────────────────────────
    if outliers and outliers.has_outliers:
        for msg in outliers.recommendations:
            recs.append(
                MasterRecommendation(
                    priority="low",
                    category="outliers",
                    message=msg,
                )
            )

    # ── Memory ───────────────────────────────────────────────────────────
    if memory and memory.potential_savings_mb > 1.0:
        recs.append(
            MasterRecommendation(
                priority="low",
                category="memory",
                message=f"Memory optimization available: {memory.potential_savings_mb:.1f} MB savings possible.",
                action="Apply dtype downcasting before training large datasets.",
            )
        )

    # ── Sort: critical → high → medium → low ─────────────────────────────
    _order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda r: _order.get(r.priority, 9))

    counts = {p: sum(1 for r in recs if r.priority == p) for p in ("critical", "high", "medium", "low")}

    if counts["critical"] > 0:
        health = "poor"
    elif counts["high"] > 2:
        health = "fair"
    elif counts["high"] > 0 or counts["medium"] > 3:
        health = "good"
    else:
        health = "excellent"

    return MasterRecommendationReport(
        recommendations=recs,
        critical_count=counts["critical"],
        high_count=counts["high"],
        medium_count=counts["medium"],
        low_count=counts["low"],
        overall_health=health,
    )
