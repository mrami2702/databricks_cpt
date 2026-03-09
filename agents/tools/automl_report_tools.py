"""
AutoML report generation tools for automl_agent.

Takes the JSON results returned by the AutoML notebook and produces:
  1. A markdown summary for inline display in the agent response
  2. A self-contained HTML report saved to agents/output/ with a file:// URL

Mirrors the structure and CSS of eda_report_tools.py exactly.
No external dependencies — pure Python stdlib + HTML string generation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# CSS — identical to eda_report_tools.py so both reports share the same look
# ---------------------------------------------------------------------------
_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  line-height: 1.6;
  padding: 2rem;
}
.container { max-width: 1100px; margin: 0 auto; }
header { margin-bottom: 2rem; }
header h1 { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
header .meta { color: #64748b; font-size: 0.875rem; margin-top: 0.25rem; }
.cards { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 1rem 1.5rem;
  flex: 1;
  min-width: 140px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.card .label {
  font-size: 0.72rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 600;
}
.card .value { font-size: 2rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; }
.card .value-sm { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; word-break: break-all; }
h2 {
  font-size: 1rem;
  font-weight: 600;
  margin: 2rem 0 0.75rem;
  color: #0f172a;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid #e2e8f0;
}
.table-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
  border: 1px solid #e2e8f0;
}
th {
  background: #f1f5f9;
  color: #475569;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: .06em;
  padding: 0.75rem 1rem;
  text-align: left;
  font-weight: 600;
}
td {
  padding: 0.6rem 1rem;
  border-top: 1px solid #f1f5f9;
  font-size: 0.875rem;
  vertical-align: middle;
}
tr:hover td { background: #f8fafc; }
tr.best-row td { background: #f0fdf4; font-weight: 600; }
tr.best-row:hover td { background: #dcfce7; }
code {
  background: #f1f5f9;
  padding: 0.1em 0.4em;
  border-radius: 4px;
  font-family: "SF Mono", "Fira Code", Consolas, monospace;
  font-size: 0.82em;
}
.badge {
  display: inline-block;
  padding: 0.2em 0.6em;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.badge-regression    { background: #dbeafe; color: #1d4ed8; }
.badge-classification { background: #fce7f3; color: #be185d; }
.badge-forecast      { background: #fef3c7; color: #92400e; }
.experiment-link {
  display: inline-block;
  margin-top: 0.5rem;
  padding: 0.5rem 1rem;
  background: #3b82f6;
  color: #fff;
  border-radius: 6px;
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 600;
}
.experiment-link:hover { background: #2563eb; }
.metric-primary td:first-child { color: #0f172a; font-weight: 700; }
.metric-primary td:last-child  { color: #16a34a; font-weight: 700; }
footer {
  margin-top: 3rem;
  color: #94a3b8;
  font-size: 0.75rem;
  text-align: center;
}
"""


def generate_automl_report(automl_json: str) -> dict:
    """Generate a markdown summary and a local HTML report from AutoML notebook output.

    Call this immediately after get_job_run_output returns a successful AutoML run.
    Pass the value of 'notebook_output' from that result directly to this function.

    Args:
        automl_json: JSON string from the AutoML notebook (the notebook_output field
                     from get_job_run_output). Can also accept a dict if already parsed.

    Returns:
        dict with:
          - 'markdown_summary': formatted table for inline display in the chat
          - 'report_url': file:// URL — copy into browser address bar to view
          - 'report_path': absolute local path to the HTML file
          - 'message': human-readable summary of what was generated
    """
    try:
        data = json.loads(automl_json) if isinstance(automl_json, str) else automl_json
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Could not parse AutoML JSON: {e}", "raw": str(automl_json)[:200]}

    if "error" in data:
        return {"error": f"AutoML run returned an error: {data['error']}"}

    markdown = _build_markdown_summary(data)
    html = _build_html_report(data)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_slug = (
        data.get("table_name", "table")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"automl_{table_slug}_{timestamp}.html"
    file_path = _OUTPUT_DIR / filename
    file_path.write_text(html, encoding="utf-8")

    report_url = file_path.resolve().as_uri()
    return {
        "markdown_summary": markdown,
        "report_url": report_url,
        "report_path": str(file_path.resolve()),
        "message": (
            f"AutoML report saved. To view the full HTML report, copy this URL "
            f"into your browser address bar: {report_url}"
        ),
    }


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _get_primary_metric(metrics: dict, problem_type: str) -> tuple[str, float, bool]:
    """Return (metric_name, value, higher_is_better) for the headline metric."""
    if not metrics:
        return ("unknown", 0.0, True)

    # Preference order by problem type; tuple is (key_substring, higher_is_better)
    preferences = {
        "regression": [
            ("r2_score", True),
            ("r2",       True),
            ("rmse",     False),
            ("mae",      False),
        ],
        "classification": [
            ("f1_score",       True),
            ("f1",             True),
            ("accuracy_score", True),
            ("accuracy",       True),
            ("roc_auc",        True),
            ("log_loss",       False),
        ],
        "forecast": [
            ("smape", False),
            ("mdape", False),
            ("mse",   False),
            ("mae",   False),
        ],
    }
    for substring, higher in preferences.get(problem_type, []):
        for key, val in metrics.items():
            if substring in key.lower():
                return (key, float(val), higher)

    # Fallback: return first metric, guess direction from name
    first_key = next(iter(metrics))
    higher = any(s in first_key.lower() for s in ("r2", "accuracy", "f1", "auc"))
    return (first_key, float(metrics[first_key]), higher)


def _build_markdown_summary(data: dict) -> str:
    table_name    = data.get("table_name", "unknown")
    target_col    = data.get("target_col", "unknown")
    problem_type  = data.get("problem_type", "unknown")
    best_model    = data.get("best_model", "unknown")
    best_metrics  = data.get("best_metrics", {})
    num_trials    = data.get("num_trials", 0)
    experiment_url = data.get("experiment_url", "")
    trials        = data.get("trials_summary", [])

    metric_name, metric_val, higher = _get_primary_metric(best_metrics, problem_type)
    direction = "↑ higher is better" if higher else "↓ lower is better"

    lines = [
        f"### AutoML — `{table_name}`",
        f"**Problem:** {problem_type}  ·  **Target:** `{target_col}`  ·  **Trials:** {num_trials}",
        "",
        f"**Best model:** `{best_model}`",
        f"**{metric_name}:** `{metric_val:.4f}` ({direction})",
        "",
    ]

    # All best-model metrics
    if best_metrics:
        lines += [
            "**Best model — all metrics:**",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in best_metrics.items():
            star = " ★" if k == metric_name else ""
            lines.append(f"| `{k}` | `{v:.4f}`{star} |")
        lines.append("")

    # Top trials table
    if trials:
        # Collect all metric keys across trials for consistent columns
        all_metric_keys = []
        seen = set()
        for t in trials:
            for k in (t.get("metrics") or {}).keys():
                if k not in seen:
                    all_metric_keys.append(k)
                    seen.add(k)

        header_row = "| # | Model | " + " | ".join(all_metric_keys) + " |"
        sep_row    = "|---|-------|" + "|".join("-------" for _ in all_metric_keys) + "|"
        lines += ["**Top trials:**", header_row, sep_row]
        for i, t in enumerate(trials, 1):
            m = t.get("metrics", {})
            metric_cells = " | ".join(f"`{m.get(k, 'n/a')}`" for k in all_metric_keys)
            lines.append(f"| {i} | `{t.get('model', 'unknown')}` | {metric_cells} |")
        lines.append("")

    if experiment_url:
        lines.append(f"**MLflow experiment:** {experiment_url}")

    return "\n".join(lines)


def _build_html_report(data: dict) -> str:
    table_name     = data.get("table_name", "unknown")
    target_col     = data.get("target_col", "unknown")
    problem_type   = data.get("problem_type", "unknown")
    best_model     = data.get("best_model", "unknown")
    best_metrics   = data.get("best_metrics", {})
    num_trials     = data.get("num_trials", 0)
    experiment_url = data.get("experiment_url", "")
    best_run_id    = data.get("best_run_id", "")
    trials         = data.get("trials_summary", [])
    generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metric_name, metric_val, higher = _get_primary_metric(best_metrics, problem_type)
    direction_label = "higher is better" if higher else "lower is better"
    badge_class = f"badge-{problem_type}" if problem_type in ("regression", "classification", "forecast") else "badge-regression"

    # Stat cards
    metric_display = f"{metric_val:.4f}"

    # Best model metrics table
    metrics_rows = ""
    for k, v in best_metrics.items():
        primary_class = ' class="metric-primary"' if k == metric_name else ""
        metrics_rows += (
            f'<tr{primary_class}>'
            f"<td><code>{_esc(k)}</code></td>"
            f"<td>{v:.6f}</td>"
            f"</tr>"
        )

    # Top trials table
    trials_section = ""
    if trials:
        all_metric_keys = []
        seen: set = set()
        for t in trials:
            for k in (t.get("metrics") or {}).keys():
                if k not in seen:
                    all_metric_keys.append(k)
                    seen.add(k)

        header_cells = "".join(f"<th>{_esc(k)}</th>" for k in all_metric_keys)
        trial_rows = ""
        for i, t in enumerate(trials, 1):
            m = t.get("metrics", {})
            row_class = ' class="best-row"' if i == 1 else ""
            rank_cell = f"<td>{'★ 1' if i == 1 else str(i)}</td>"
            model_cell = f"<td><code>{_esc(t.get('model', 'unknown'))}</code></td>"
            metric_cells = "".join(
                f"<td>{m.get(k, '—')}</td>" for k in all_metric_keys
            )
            trial_rows += f"<tr{row_class}>{rank_cell}{model_cell}{metric_cells}</tr>"

        trials_section = (
            "<h2>Top Trials</h2>"
            '<div class="table-wrap">'
            "<table>"
            f"<thead><tr><th>#</th><th>Model</th>{header_cells}</tr></thead>"
            f"<tbody>{trial_rows}</tbody>"
            "</table>"
            "</div>"
        )

    # MLflow link
    mlflow_section = ""
    if experiment_url:
        mlflow_section = (
            "<h2>MLflow Experiment</h2>"
            f'<p>View all trials, metrics, and model artifacts in the Databricks MLflow UI.</p>'
            f'<a class="experiment-link" href="{_esc(experiment_url)}" target="_blank">'
            "Open MLflow Experiment →"
            "</a>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoML — {_esc(table_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>
      AutoML Report — <code>{_esc(table_name)}</code>
      &nbsp;<span class="badge {badge_class}">{_esc(problem_type)}</span>
    </h1>
    <div class="meta">
      Target: <strong>{_esc(target_col)}</strong>
      &nbsp;·&nbsp; Generated {generated_at}
    </div>
  </header>

  <div class="cards">
    <div class="card">
      <div class="label">Problem Type</div>
      <div class="value-sm">{_esc(problem_type)}</div>
    </div>
    <div class="card">
      <div class="label">Target Column</div>
      <div class="value-sm">{_esc(target_col)}</div>
    </div>
    <div class="card">
      <div class="label">Trials Run</div>
      <div class="value">{num_trials}</div>
    </div>
    <div class="card">
      <div class="label">{_esc(metric_name)}</div>
      <div class="value">{metric_display}</div>
      <div class="meta">{direction_label}</div>
    </div>
  </div>

  <h2>Best Model</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th>Algorithm</th><th colspan="2"><code>{_esc(best_model)}</code> &nbsp; run: <code>{_esc(best_run_id)}</code></th></tr>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>{metrics_rows}</tbody>
    </table>
  </div>

  {trials_section}

  {mlflow_section}

  <footer>Generated by AutoMLAgent · AutoML Report Tool</footer>
</div>
</body>
</html>"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
