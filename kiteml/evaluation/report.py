"""
report.py — Human-readable evaluation report generator.

Converts the raw metrics dict from evaluate_model() into a terminal-friendly,
branded KiteML report string.

The report includes:
    * Header with KiteML branding
    * Problem type and best model name
    * Primary metrics (formatted to 4 d.p.)
    * For classification: confusion matrix + full sklearn report
    * For regression: all four regression metrics
    * Model leaderboard (ranked, with winner marked)

All formatting is done with plain ASCII/Unicode — no external dependencies.
"""

from typing import Any, Optional


def generate_report(
    metrics: dict[str, Any],
    problem_type: str = "classification",
    model_name: str | None = None,
    all_results: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    Generate a formatted KiteML evaluation report string.

    Parameters
    ----------
    metrics : dict
        Output from :func:`~kiteml.evaluation.metrics.evaluate_model`.
    problem_type : str
        ``'classification'`` or ``'regression'``.
    model_name : str, optional
        Display name of the winning model.  Shown in the header when provided.
    all_results : dict, optional
        Structured results dict from
        :func:`~kiteml.models.selector.select_best_model`.
        When provided, a ranked leaderboard is appended to the report.

    Returns
    -------
    str
        Multi-line formatted report ready to print() or log.
    """
    W = 52  # report width
    SEP = "─" * W
    THICK = "═" * W

    lines = []

    # ── Header ───────────────────────────────────────────────────────────
    lines.append(THICK)
    lines.append("  🪁  KiteML — Model Evaluation Report")
    lines.append(THICK)
    if model_name:
        lines.append(f"  Model       : {model_name}")
    lines.append(f"  Problem     : {problem_type.capitalize()}")
    lines.append(SEP)

    # ── Core Metrics ─────────────────────────────────────────────────────
    if problem_type == "classification":
        lines.append(f"  Accuracy    : {metrics['accuracy']:.4f}")
        lines.append(f"  Precision   : {metrics['precision']:.4f}")
        lines.append(f"  Recall      : {metrics['recall']:.4f}")
        lines.append(f"  F1 Score    : {metrics['f1_score']:.4f}")
        lines.append(SEP)

        # Confusion matrix
        lines.append("  Confusion Matrix:")
        for row in metrics.get("confusion_matrix", []):
            lines.append(f"    {row}")
        lines.append(SEP)

        # Full sklearn classification report
        lines.append("  Classification Report:")
        for row in metrics.get("classification_report", "").splitlines():
            lines.append(f"  {row}")
    else:
        lines.append(f"  R² Score    : {metrics['r2_score']:.4f}")
        lines.append(f"  MSE         : {metrics['mse']:.4f}")
        lines.append(f"  RMSE        : {metrics['rmse']:.4f}")
        lines.append(f"  MAE         : {metrics['mae']:.4f}")

    # ── Leaderboard (optional) ───────────────────────────────────────────
    if all_results:
        lines.append(SEP)
        lines.append("  Model Leaderboard:")

        # Sort by rank (errors go last)
        def _rank_key(item):
            rank = item[1].get("rank")
            return rank if rank is not None else 9999

        sorted_items = sorted(all_results.items(), key=_rank_key)
        for name, info in sorted_items:
            score = info.get("score")
            rank = info.get("rank")
            error = info.get("error")

            if score is not None:
                marker = " ✓" if rank == 1 else ""
                rank_str = f"#{rank}"
                lines.append(f"  {rank_str:<4} {name:<26} {score:.4f}{marker}")
            else:
                lines.append(f"  ERR  {name:<26} ⚠ {error[:28] if error else 'unknown error'}")

    lines.append(THICK)
    return "\n".join(lines)
