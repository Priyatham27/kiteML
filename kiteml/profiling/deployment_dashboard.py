"""
deployment_dashboard.py — HTML production dashboard for KiteML.

Generates a rich, self-contained HTML dashboard showing model metadata,
metrics, feature importance, drift status, and deployment readiness.
"""

import html
import time
from typing import Any, Optional

_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#080c14;color:#e2e8f0;margin:0;padding:24px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:20px;}
.card{background:#111827;border-radius:12px;padding:20px;border:1px solid #1e293b;}
.card h2{color:#818cf8;font-size:1rem;margin:0 0 12px 0;text-transform:uppercase;letter-spacing:.05em;}
.kpi{font-size:2rem;font-weight:700;color:#c7d2fe;}
.kpi-label{font-size:.8rem;color:#64748b;margin-top:4px;}
h1{color:#818cf8;font-size:1.6rem;margin-bottom:4px;}
.subtitle{color:#64748b;font-size:.9rem;margin-bottom:24px;}
table{width:100%;border-collapse:collapse;font-size:.85rem;}
th{text-align:left;padding:6px 8px;background:#1e293b;color:#818cf8;}
td{padding:5px 8px;border-bottom:1px solid #1e293b;}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600;}
.badge-ok{background:#064e3b;color:#6ee7b7;}
.badge-warn{background:#713f12;color:#fde68a;}
.badge-err{background:#7f1d1d;color:#fca5a5;}
.bar{height:8px;border-radius:4px;background:#818cf8;}
.bar-bg{background:#1e293b;border-radius:4px;height:8px;margin-top:4px;}
"""


def _badge(text, cls):
    return f'<span class="badge badge-{cls}">{html.escape(str(text))}</span>'


def _bar(ratio):
    return f'<div class="bar-bg"><div class="bar" style="width:{int(min(ratio,1)*100)}%"></div></div>'


def generate_dashboard(
    result: Any,
    path: str = "kiteml_dashboard.html",
    drift_report: Optional[Any] = None,
) -> str:
    """
    Generate a production deployment dashboard HTML file.

    Parameters
    ----------
    result : Result
    path : str
        Output file path. Default ``'kiteml_dashboard.html'``.
    drift_report : DriftReport, optional
        If provided, include drift status in dashboard.

    Returns
    -------
    str
        Path to saved file.
    """
    score_str = f"{result.score:.4f}" if result.score is not None else "N/A"
    try:
        score_val = float(result.score or 0)
    except Exception:
        score_val = 0.0

    has_profile = result.data_profile is not None

    sections = []

    # ── Header ────────────────────────────────────────────────────────────
    sections.append(f"""
    <h1>🪁 KiteML — Production Dashboard</h1>
    <p class="subtitle">
      Model: <b>{html.escape(result.model_name)}</b> &nbsp;|&nbsp;
      Type: <b>{html.escape(result.problem_type)}</b> &nbsp;|&nbsp;
      Generated: {time.strftime("%Y-%m-%d %H:%M UTC")}
    </p>""")

    # ── KPI Cards ─────────────────────────────────────────────────────────
    drift_badge = _badge("N/A", "warn")
    if drift_report:
        if drift_report.drift_detected:
            drift_badge = _badge(f"DRIFT ({drift_report.severity.upper()})", "err")
        else:
            drift_badge = _badge("STABLE", "ok")

    sections.append(f"""
    <div class="grid">
      <div class="card">
        <h2>Best Score</h2>
        <div class="kpi">{score_str}</div>
        <div class="kpi-label">{result.problem_type}</div>
        {_bar(score_val)}
      </div>
      <div class="card">
        <h2>Training Time</h2>
        <div class="kpi">{result.times.total:.2f}s</div>
        <div class="kpi-label">model.fit: {result.times.training:.2f}s</div>
      </div>
      <div class="card">
        <h2>Features</h2>
        <div class="kpi">{len(result.feature_names or [])}</div>
        <div class="kpi-label">input dimensions</div>
      </div>
      <div class="card">
        <h2>Drift Status</h2>
        <div style="margin-top:12px">{drift_badge}</div>
        <div class="kpi-label" style="margin-top:8px">
          {"PSI=" + str(drift_report.overall_psi) if drift_report else "No drift data"}
        </div>
      </div>
    </div>""")

    # ── Model Leaderboard ─────────────────────────────────────────────────
    if result.all_results:

        def _get_score(val):
            if isinstance(val, dict):
                return float(val.get("score") or 0)
            return float(val or 0)

        sorted_results = sorted(result.all_results.items(), key=lambda x: _get_score(x[1]), reverse=True)
        rows = "".join(
            f"<tr><td>{'★ ' if i==0 else '  '}{html.escape(model)}</td>"
            f"<td>{round(_get_score(score),4)}</td>"
            f"<td>{_bar(_get_score(score) if _get_score(score) > 0 else 0)}</td></tr>"
            for i, (model, score) in enumerate(sorted_results[:8])
        )
        sections.append(f"""
    <div class="card">
      <h2>📊 Model Leaderboard</h2>
      <table><tr><th>Model</th><th>CV Score</th><th>Relative</th></tr>{rows}</table>
    </div>""")

    # ── Feature Importance ────────────────────────────────────────────────
    if result.feature_importances:
        fi_sorted = sorted(result.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        max_fi = abs(fi_sorted[0][1]) if fi_sorted else 1.0
        rows = "".join(
            f"<tr><td>{html.escape(feat)}</td><td>{round(abs(imp),4)}</td>"
            f"<td>{_bar(abs(imp)/(max_fi+1e-8))}</td></tr>"
            for feat, imp in fi_sorted
        )
        sections.append(f"""
    <div class="card">
      <h2>🔍 Feature Importances</h2>
      <table><tr><th>Feature</th><th>Importance</th><th>Relative</th></tr>{rows}</table>
    </div>""")

    # ── Data Profile Summary ──────────────────────────────────────────────
    if has_profile:
        q = result.data_profile.quality
        mr = result.data_profile.master_recommendations
        health_cls = {"excellent": "ok", "good": "ok", "fair": "warn", "poor": "err"}.get(mr.overall_health, "warn")
        sections.append(f"""
    <div class="card">
      <h2>🧠 Data Intelligence</h2>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Quality Score</td><td>{q.score}/100</td></tr>
        <tr><td>Overall Health</td><td>{_badge(mr.overall_health.upper(), health_cls)}</td></tr>
        <tr><td>Critical Issues</td><td>{mr.critical_count}</td></tr>
        <tr><td>High Issues</td><td>{mr.high_count}</td></tr>
        <tr><td>Leakage Risk</td><td>{_badge("YES","err") if result.data_profile.leakage.has_leakage_risk else _badge("NONE","ok")}</td></tr>
      </table>
    </div>""")

    # ── Drift Details ─────────────────────────────────────────────────────
    if drift_report and drift_report.drifted_features:
        drift_rows = "".join(
            f"<tr><td>{html.escape(f)}</td>"
            f"<td>{result.feature_results[f].psi if hasattr(result,'feature_results') else drift_report.feature_results.get(f).psi if drift_report.feature_results.get(f) else 'N/A'}</td>"
            f"<td>{_badge(drift_report.feature_results[f].severity,'err' if drift_report.feature_results[f].severity=='high' else 'warn')}</td></tr>"
            for f in drift_report.drifted_features[:8]
        )
        sections.append(f"""
    <div class="card">
      <h2>🌊 Drifted Features</h2>
      <table><tr><th>Feature</th><th>PSI</th><th>Severity</th></tr>{drift_rows}</table>
    </div>""")

    # ── Deployment Readiness ──────────────────────────────────────────────
    checks = [
        ("Model trained", True, "ok"),
        ("Preprocessor attached", result.preprocessor is not None, "ok" if result.preprocessor else "warn"),
        ("Feature names defined", bool(result.feature_names), "ok" if result.feature_names else "err"),
        ("DataProfile available", has_profile, "ok" if has_profile else "warn"),
        (
            "No leakage detected",
            not (has_profile and result.data_profile.leakage.has_leakage_risk),
            "ok" if not (has_profile and result.data_profile.leakage.has_leakage_risk) else "err",
        ),
    ]
    check_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{_badge('PASS' if ok else 'FAIL', cls)}</td></tr>"
        for label, ok, cls in checks
    )
    sections.append(f"""
    <div class="card">
      <h2>✅ Deployment Readiness</h2>
      <table><tr><th>Check</th><th>Status</th></tr>{check_rows}</table>
    </div>""")

    body = "\n".join(sections)
    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>KiteML Production Dashboard</title>
<style>{_CSS}</style></head>
<body>{body}</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"📊 Dashboard saved → {path}")
    return path
