"""
report_generator.py — Rich terminal profiling reports for KiteML.
"""

from typing import Any

from kiteml.intelligence.data_profiler import DataProfile


def generate_profile_report(profile: DataProfile) -> str:
    """
    Generate a rich formatted terminal profiling report from a DataProfile.

    Returns
    -------
    str
        Full report as a printable string.
    """
    W = 60
    lines = []

    def sep(char="─"): lines.append(char * W)
    def header(text): lines.append(f"  {text}")
    def row(label, value): lines.append(f"  {label:<30} {value}")

    lines.append("═" * W)
    lines.append("  🪁  KiteML — Dataset Intelligence Report")
    lines.append("═" * W)

    # ── Overview ──────────────────────────────────────────────────────────
    s = profile.schema
    header(f"📊 Dataset Overview")
    sep()
    row("Rows", f"{s.n_rows:,}")
    row("Columns", f"{s.n_cols:,}")
    row("Memory", f"{s.memory_bytes / 1e6:.2f} MB")
    row("Target column", profile.target)
    row("Problem type", f"{profile.problem_type} ({profile.problem_inference.subtype})")
    row("Inference confidence", f"{profile.problem_inference.confidence:.0%}")

    # ── Column types ──────────────────────────────────────────────────────
    sep()
    header("🗂️  Column Types")
    sep()
    for col_type, count in profile.column_analysis.type_summary.items():
        row(f"  {col_type}", str(count))

    # ── Data quality ──────────────────────────────────────────────────────
    sep()
    header("🔍 Data Quality")
    sep()
    q = profile.quality
    row("Quality score", f"{q.score}/100")
    row("Errors", str(len(q.by_severity(q.by_severity.__self__.__class__.ERROR
                                        if False else type(q.issues[0]).severity.__class__.ERROR
                                        if q.issues else 'ERROR')))
        if False else str(sum(1 for i in q.issues if i.severity.value == "error")))
    row("Warnings", str(sum(1 for i in q.issues if i.severity.value == "warning")))
    if q.issues:
        header("  Issues:")
        for issue in q.issues[:5]:
            icon = "🔴" if issue.severity.value == "error" else "🟡"
            lines.append(f"    {icon} {issue.description}")

    # ── Class imbalance ───────────────────────────────────────────────────
    if profile.imbalance:
        sep()
        header("⚖️  Class Imbalance")
        sep()
        im = profile.imbalance
        row("Severity", im.severity.upper())
        row("Imbalance ratio", f"{im.imbalance_ratio:.1f}:1")
        for cls, frac in list(im.class_distribution.items())[:6]:
            row(f"  Class '{cls}'", f"{frac:.1%}")

    # ── Leakage ───────────────────────────────────────────────────────────
    sep()
    header("🚨 Leakage Detection")
    sep()
    lk = profile.leakage
    if lk.has_leakage_risk:
        row("Status", "⚠️  RISKS DETECTED")
        for risk in lk.risks[:5]:
            lines.append(f"    [{risk.risk_level.upper()}] '{risk.column}': {risk.reason}")
    else:
        row("Status", "✅ No leakage risks")

    # ── Correlations ─────────────────────────────────────────────────────
    if profile.correlations.top_predictors:
        sep()
        header("📈 Top Feature Correlations with Target")
        sep()
        for feat in profile.correlations.top_predictors[:5]:
            corr = profile.correlations.target_correlations.get(feat, 0)
            bar = "█" * int(corr * 20)
            row(f"  {feat}", f"|r|={corr:.3f}  {bar}")

    # ── Outliers ─────────────────────────────────────────────────────────
    sep()
    header("📉 Outlier Summary")
    sep()
    ol = profile.outliers
    row("Columns with outliers", str(len(ol.columns_with_outliers)))
    row("Outlier rows", f"{ol.total_outlier_rows:,} ({ol.outlier_row_ratio:.1%})")

    # ── Text / datetime columns ───────────────────────────────────────────
    if profile.text.has_text:
        sep()
        header("📝 Text Columns (NLP candidates)")
        sep()
        for col in profile.text.text_columns:
            info = profile.text.details[col]
            row(f"  {col}", f"avg {info.avg_word_count:.1f} words")

    if profile.datetime.has_datetime:
        sep()
        header("📅 Datetime Columns")
        sep()
        for col in profile.datetime.datetime_columns:
            info = profile.datetime.details[col]
            row(f"  {col}", f"range: {info.date_range_days} days")

    # ── Memory ────────────────────────────────────────────────────────────
    sep()
    header("💾 Memory")
    sep()
    mem = profile.memory
    row("Total memory", f"{mem.total_memory_mb:.2f} MB")
    if mem.potential_savings_mb > 0:
        row("Potential savings", f"{mem.potential_savings_mb:.2f} MB")

    # ── Recommendations summary ───────────────────────────────────────────
    sep()
    header("💡 Top Recommendations")
    sep()
    mr = profile.master_recommendations
    row("Overall health", mr.overall_health.upper())
    row(f"Critical / High / Medium / Low",
        f"{mr.critical_count} / {mr.high_count} / {mr.medium_count} / {mr.low_count}")
    for rec in mr.recommendations[:6]:
        icon = {"critical": "🚨", "high": "⚠️ ", "medium": "💡", "low": "ℹ️ "}.get(rec.priority, "  ")
        lines.append(f"    {icon} {rec.message[:70]}")

    lines.append("═" * W)
    return "\n".join(lines)
