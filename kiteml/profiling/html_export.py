"""
html_export.py — HTML report export for KiteML DataProfile.

Generates a self-contained, styled HTML file with the complete
dataset intelligence report.  No external CSS or JS dependencies.
"""

import html
from typing import Any

from kiteml.intelligence.data_profiler import DataProfile


_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e2e8f0;margin:0;padding:24px;}
.card{background:#1a1f2e;border-radius:12px;padding:24px;margin-bottom:20px;border:1px solid #2d3748;}
h1{color:#7c3aed;font-size:1.8rem;margin-bottom:4px;}
h2{color:#a78bfa;font-size:1.1rem;border-bottom:1px solid #2d3748;padding-bottom:8px;margin-top:0;}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:bold;}
.badge-ok{background:#065f46;color:#6ee7b7;}
.badge-warn{background:#78350f;color:#fcd34d;}
.badge-err{background:#7f1d1d;color:#fca5a5;}
table{width:100%;border-collapse:collapse;font-size:.9rem;}
th{text-align:left;padding:6px 10px;background:#2d3748;color:#a78bfa;}
td{padding:6px 10px;border-bottom:1px solid #2d3748;}
.bar-container{background:#2d3748;border-radius:4px;height:10px;width:120px;display:inline-block;vertical-align:middle;}
.bar-fill{background:#7c3aed;height:10px;border-radius:4px;}
.subtitle{color:#94a3b8;font-size:.9rem;}
.rec-critical{border-left:4px solid #f87171;padding:8px 12px;margin:4px 0;background:#1f1f2e;}
.rec-high{border-left:4px solid #fb923c;padding:8px 12px;margin:4px 0;background:#1f1f2e;}
.rec-medium{border-left:4px solid #facc15;padding:8px 12px;margin:4px 0;background:#1f1f2e;}
.rec-low{border-left:4px solid #4ade80;padding:8px 12px;margin:4px 0;background:#1f1f2e;}
"""


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{html.escape(str(text))}</span>'


def _bar(ratio: float) -> str:
    pct = int(min(ratio, 1.0) * 100)
    return (
        f'<div class="bar-container">'
        f'<div class="bar-fill" style="width:{pct}%"></div>'
        f'</div> {pct}%'
    )


def export_html(profile: DataProfile, path: str = "kiteml_report.html") -> str:
    """
    Export the DataProfile as a self-contained HTML report.

    Parameters
    ----------
    profile : DataProfile
    path : str
        Output file path. Default ``'kiteml_report.html'``.

    Returns
    -------
    str
        The file path where the report was saved.
    """
    s = profile.schema
    q = profile.quality
    lk = profile.leakage
    mr = profile.master_recommendations
    ol = profile.outliers
    mem = profile.memory
    im = profile.imbalance

    sections = []

    # ── Overview ──────────────────────────────────────────────────────────
    health_cls = {"excellent": "ok", "good": "ok", "fair": "warn", "poor": "err"}.get(
        mr.overall_health, "warn")
    sections.append(f"""
    <div class="card">
      <h1>🪁 KiteML — Dataset Intelligence Report</h1>
      <p class="subtitle">Target: <b>{html.escape(profile.target)}</b> &nbsp;|&nbsp;
         Problem: <b>{html.escape(profile.problem_type)}</b>
         ({html.escape(profile.problem_inference.subtype)}) &nbsp;|&nbsp;
         Health: {_badge(mr.overall_health.upper(), health_cls)}
      </p>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Rows</td><td>{s.n_rows:,}</td></tr>
        <tr><td>Columns</td><td>{s.n_cols:,}</td></tr>
        <tr><td>Memory</td><td>{s.memory_bytes/1e6:.2f} MB</td></tr>
        <tr><td>Quality Score</td><td>{q.score}/100</td></tr>
        <tr><td>Leakage Risk</td><td>{_badge("YES", "err") if lk.has_leakage_risk else _badge("NONE", "ok")}</td></tr>
      </table>
    </div>""")

    # ── Column types ──────────────────────────────────────────────────────
    type_rows = "".join(
        f"<tr><td>{t}</td><td>{c}</td></tr>"
        for t, c in profile.column_analysis.type_summary.items()
    )
    sections.append(f"""
    <div class="card">
      <h2>🗂️ Column Type Summary</h2>
      <table><tr><th>Type</th><th>Count</th></tr>{type_rows}</table>
    </div>""")

    # ── Quality issues ────────────────────────────────────────────────────
    if q.issues:
        issue_rows = "".join(
            f'<tr><td>{_badge(i.severity.value, "err" if i.severity.value=="error" else "warn")}</td>'
            f'<td>{html.escape(i.description)}</td>'
            f'<td>{html.escape(i.recommendation)}</td></tr>'
            for i in q.issues
        )
        sections.append(f"""
    <div class="card">
      <h2>🔍 Data Quality Issues</h2>
      <table><tr><th>Severity</th><th>Issue</th><th>Recommendation</th></tr>
      {issue_rows}</table>
    </div>""")

    # ── Imbalance ─────────────────────────────────────────────────────────
    if im:
        dist_rows = "".join(
            f"<tr><td>{html.escape(str(cls))}</td><td>{_bar(frac)}</td></tr>"
            for cls, frac in list(im.class_distribution.items())[:10]
        )
        sections.append(f"""
    <div class="card">
      <h2>⚖️ Class Imbalance — {_badge(im.severity.upper(), "err" if im.severity in ("severe","extreme") else "warn" if im.is_imbalanced else "ok")}</h2>
      <p>Imbalance ratio: <b>{im.imbalance_ratio:.1f}:1</b></p>
      <table><tr><th>Class</th><th>Distribution</th></tr>{dist_rows}</table>
    </div>""")

    # ── Leakage ───────────────────────────────────────────────────────────
    if lk.has_leakage_risk:
        leak_rows = "".join(
            f'<tr><td>{_badge(r.risk_level.upper(), "err" if r.risk_level=="critical" else "warn")}</td>'
            f'<td>{html.escape(r.column)}</td>'
            f'<td>{html.escape(r.reason)}</td></tr>'
            for r in lk.risks
        )
        sections.append(f"""
    <div class="card">
      <h2>🚨 Leakage Risks</h2>
      <table><tr><th>Level</th><th>Column</th><th>Reason</th></tr>{leak_rows}</table>
    </div>""")

    # ── Top correlations ──────────────────────────────────────────────────
    if profile.correlations.top_predictors:
        corr_rows = "".join(
            f"<tr><td>{html.escape(feat)}</td>"
            f"<td>{_bar(profile.correlations.target_correlations.get(feat, 0))}</td></tr>"
            for feat in profile.correlations.top_predictors[:10]
        )
        sections.append(f"""
    <div class="card">
      <h2>📈 Top Feature Correlations with Target</h2>
      <table><tr><th>Feature</th><th>|Correlation|</th></tr>{corr_rows}</table>
    </div>""")

    # ── Master recommendations ────────────────────────────────────────────
    rec_html = "".join(
        f'<div class="rec-{r.priority}"><b>[{r.category.upper()}]</b> {html.escape(r.message)}'
        + (f'<br><small>→ {html.escape(r.action)}</small>' if r.action else "")
        + "</div>"
        for r in mr.recommendations
    )
    sections.append(f"""
    <div class="card">
      <h2>💡 Recommendations ({mr.critical_count} critical, {mr.high_count} high,
          {mr.medium_count} medium, {mr.low_count} low)</h2>
      {rec_html}
    </div>""")

    # ── Memory ────────────────────────────────────────────────────────────
    sections.append(f"""
    <div class="card">
      <h2>💾 Memory Usage</h2>
      <p>Total: <b>{mem.total_memory_mb:.2f} MB</b> &nbsp;|&nbsp;
         Potential savings: <b>{mem.potential_savings_mb:.2f} MB</b></p>
    </div>""")

    body = "\n".join(sections)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>KiteML Intelligence Report</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"📄 HTML report saved → {path}")
    return path
