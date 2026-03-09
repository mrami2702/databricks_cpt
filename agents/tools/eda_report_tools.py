"""
EDA report generation tools for databricks_job_agent.

Takes the JSON profile returned by the EDA notebook and produces:
  1. A markdown summary table for inline display in the agent response
  2. A self-contained HTML report saved to agents/output/ with a file:// URL

No external dependencies — pure Python stdlib + HTML string generation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# HTML reports are written here; directory is created on first use
_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# CSS is kept as a plain string so curly braces don't conflict with f-strings
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
code {
  background: #f1f5f9;
  padding: 0.1em 0.4em;
  border-radius: 4px;
  font-family: "SF Mono", "Fira Code", Consolas, monospace;
  font-size: 0.82em;
}
.null-bar-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.null-bar-track {
  flex: 1;
  background: #e5e7eb;
  border-radius: 4px;
  height: 10px;
  overflow: hidden;
}
.null-bar-fill { height: 100%; border-radius: 4px; }
.null-label {
  min-width: 46px;
  text-align: right;
  font-size: 0.82em;
  font-weight: 600;
}
footer {
  margin-top: 3rem;
  color: #94a3b8;
  font-size: 0.75rem;
  text-align: center;
}
"""


def generate_eda_report(eda_json: str) -> dict:
    """Generate a markdown summary and a local HTML report from EDA notebook output.

    Call this immediately after get_job_run_output returns a successful EDA run.
    Pass the value of 'notebook_output' from that result directly to this function.

    Args:
        eda_json: JSON string from the EDA notebook (the notebook_output field from
                  get_job_run_output). Can also accept a dict if already parsed.

    Returns:
        dict with:
          - 'markdown_summary': formatted table for inline display in the chat
          - 'report_url': file:// URL — copy into browser address bar to view
          - 'report_path': absolute local path to the HTML file
          - 'message': human-readable summary of what was generated
    """
    try:
        data = json.loads(eda_json) if isinstance(eda_json, str) else eda_json
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Could not parse EDA JSON: {e}", "raw": str(eda_json)[:200]}

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
    filename = f"eda_{table_slug}_{timestamp}.html"
    file_path = _OUTPUT_DIR / filename
    file_path.write_text(html, encoding="utf-8")

    report_url = file_path.resolve().as_uri()
    return {
        "markdown_summary": markdown,
        "report_url": report_url,
        "report_path": str(file_path.resolve()),
        "message": (
            f"Report saved. To view the full HTML report, copy this URL into your "
            f"browser address bar: {report_url}"
        ),
    }


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_markdown_summary(data: dict) -> str:
    table_name = data.get("table_name", "unknown")
    row_count = data.get("row_count", 0)
    col_count = data.get("column_count", 0)
    schema = data.get("schema", [])
    null_rates = data.get("null_rates", {})

    lines = [
        f"### EDA — `{table_name}`",
        f"**{row_count:,} rows** · **{col_count} columns**",
        "",
        "| Column | Type | Null % |",
        "|--------|------|--------|",
    ]

    for col in schema:
        name = col["name"]
        col_type = col.get("type", "")
        nr = null_rates.get(name, 0.0)
        pct = f"{nr * 100:.1f}%"
        flag = " ⚠️" if nr > 0.20 else ""
        lines.append(f"| `{name}` | `{col_type}` | {pct}{flag} |")

    # Call out the worst offenders
    high_null = sorted(
        [(k, v) for k, v in null_rates.items() if v > 0.05],
        key=lambda x: x[1],
        reverse=True,
    )
    if high_null:
        lines.append("")
        lines.append(
            "**Columns with >5% nulls:** "
            + ", ".join(f"`{k}` ({v * 100:.1f}%)" for k, v in high_null[:6])
        )

    return "\n".join(lines)


def _build_html_report(data: dict) -> str:
    table_name = data.get("table_name", "unknown")
    row_count = data.get("row_count", 0)
    col_count = data.get("column_count", 0)
    schema = data.get("schema", [])
    null_rates = data.get("null_rates", {})
    sample_rows = data.get("sample_rows", [])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Schema + null rate rows
    schema_rows_html = ""
    for col in schema:
        name = col["name"]
        col_type = col.get("type", "")
        nullable = "yes" if col.get("nullable") else "no"
        nr = null_rates.get(name, 0.0)
        pct_val = nr * 100
        pct_str = f"{pct_val:.1f}%"
        if nr > 0.20:
            bar_color = "#ef4444"
        elif nr > 0.05:
            bar_color = "#f97316"
        else:
            bar_color = "#22c55e"

        schema_rows_html += (
            f"<tr>"
            f"<td><code>{_esc(name)}</code></td>"
            f"<td><code>{_esc(col_type)}</code></td>"
            f"<td>{nullable}</td>"
            f"<td>"
            f'<div class="null-bar-wrap">'
            f'<div class="null-bar-track">'
            f'<div class="null-bar-fill" style="width:{pct_val:.2f}%;background:{bar_color};"></div>'
            f"</div>"
            f'<span class="null-label" style="color:{bar_color};">{pct_str}</span>'
            f"</div>"
            f"</td>"
            f"</tr>"
        )

    # Sample rows table
    sample_section = ""
    if sample_rows:
        headers = "".join(f"<th>{_esc(k)}</th>" for k in sample_rows[0].keys())
        rows_html = ""
        for row in sample_rows:
            cells = "".join(f"<td>{_esc(str(v))}</td>" for v in row.values())
            rows_html += f"<tr>{cells}</tr>"
        sample_section = (
            "<h2>Sample Rows</h2>"
            '<div class="table-wrap">'
            "<table>"
            f"<thead><tr>{headers}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA — {_esc(table_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>EDA Report — <code>{_esc(table_name)}</code></h1>
    <div class="meta">Generated {generated_at}</div>
  </header>

  <div class="cards">
    <div class="card">
      <div class="label">Rows</div>
      <div class="value">{row_count:,}</div>
    </div>
    <div class="card">
      <div class="label">Columns</div>
      <div class="value">{col_count}</div>
    </div>
  </div>

  <h2>Schema &amp; Null Rates</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Nullable</th>
          <th>Null Rate</th>
        </tr>
      </thead>
      <tbody>{schema_rows_html}</tbody>
    </table>
  </div>

  {sample_section}

  <footer>Generated by DatabricksJobAgent · EDA Report Tool</footer>
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
