# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Report
# MAGIC
# MAGIC A visual dashboard of your training experiments — metrics, parameters,
# MAGIC comparisons, and model artifacts — all pulled directly from MLflow.
# MAGIC
# MAGIC **No model loading needed.** This notebook only reads experiment data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# TODO: Fill in your MLflow experiment name
EXPERIMENT_NAME = ""  # e.g. "/Shared/cpt-sft-mistral"

# Optional: filter to specific run types
SHOW_CPT_RUNS = True
SHOW_SFT_RUNS = True

# Max runs to display
MAX_RUNS = 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Experiment Data

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime, timedelta

client = MlflowClient()

# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Check your experiment name.")

experiment_id = experiment.experiment_id

# Pull all runs
runs_df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    max_results=MAX_RUNS,
    order_by=["start_time DESC"],
)

# Filter by training type if requested
if not SHOW_CPT_RUNS:
    runs_df = runs_df[runs_df.get("params.training_type", "") != "CPT"]
if not SHOW_SFT_RUNS:
    runs_df = runs_df[runs_df.get("params.training_type", "") != "SFT"]

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Experiment ID: {experiment_id}")
print(f"Total runs found: {len(runs_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Experiment Overview

# COMMAND ----------

# Build overview stats
total_runs = len(runs_df)
completed_runs = len(runs_df[runs_df["status"] == "FINISHED"]) if "status" in runs_df.columns else total_runs
failed_runs = len(runs_df[runs_df["status"] == "FAILED"]) if "status" in runs_df.columns else 0

# Time range
if "start_time" in runs_df.columns and len(runs_df) > 0:
    earliest = pd.to_datetime(runs_df["start_time"].min(), unit="ms") if runs_df["start_time"].dtype != "datetime64[ns]" else runs_df["start_time"].min()
    latest = pd.to_datetime(runs_df["start_time"].max(), unit="ms") if runs_df["start_time"].dtype != "datetime64[ns]" else runs_df["start_time"].max()
    time_range = f"{earliest.strftime('%Y-%m-%d %H:%M')} — {latest.strftime('%Y-%m-%d %H:%M')}"
else:
    time_range = "N/A"

# Training types
training_types = []
if "params.training_type" in runs_df.columns:
    training_types = runs_df["params.training_type"].dropna().unique().tolist()

# Detect metric columns
metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
param_cols = [c for c in runs_df.columns if c.startswith("params.")]

overview_html = f"""
<style>
    .report-container {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0; }}
    .card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 24px; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    .card.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
    .card.orange {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
    .card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
    .card.dark {{ background: linear-gradient(135deg, #434343 0%, #000000 100%); }}
    .card-value {{ font-size: 36px; font-weight: 700; margin: 8px 0; }}
    .card-label {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }}
    .card-sub {{ font-size: 12px; opacity: 0.75; margin-top: 4px; }}
    .section-title {{ font-size: 22px; font-weight: 600; color: #1a1a2e; margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 3px solid #667eea; }}
</style>

<div class="report-container">
    <h1 style="font-size: 28px; color: #1a1a2e; margin-bottom: 4px;">MLflow Experiment Report</h1>
    <p style="color: #666; font-size: 14px; margin-top: 0;">Experiment: <code>{EXPERIMENT_NAME}</code></p>

    <div class="card-grid">
        <div class="card">
            <div class="card-label">Total Runs</div>
            <div class="card-value">{total_runs}</div>
            <div class="card-sub">{time_range}</div>
        </div>
        <div class="card green">
            <div class="card-label">Completed</div>
            <div class="card-value">{completed_runs}</div>
            <div class="card-sub">{failed_runs} failed</div>
        </div>
        <div class="card blue">
            <div class="card-label">Metrics Tracked</div>
            <div class="card-value">{len(metric_cols)}</div>
            <div class="card-sub">{len(param_cols)} parameters logged</div>
        </div>
        <div class="card orange">
            <div class="card-label">Training Types</div>
            <div class="card-value">{len(training_types)}</div>
            <div class="card-sub">{', '.join(training_types) if training_types else 'Not tagged'}</div>
        </div>
    </div>
</div>
"""

displayHTML(overview_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Run Comparison Table

# COMMAND ----------

# Build comparison table with key metrics and params
display_cols = []
col_labels = []

# Run name
if "tags.mlflow.runName" in runs_df.columns:
    display_cols.append("tags.mlflow.runName")
    col_labels.append("Run Name")

# Training type
if "params.training_type" in runs_df.columns:
    display_cols.append("params.training_type")
    col_labels.append("Type")

# Status
if "status" in runs_df.columns:
    display_cols.append("status")
    col_labels.append("Status")

# Key metrics — pick the most important ones
priority_metrics = ["metrics.final_loss", "metrics.train_loss", "metrics.loss",
                    "metrics.eval_loss", "metrics.total_steps"]
for mc in priority_metrics:
    if mc in runs_df.columns:
        display_cols.append(mc)
        col_labels.append(mc.replace("metrics.", "").replace("_", " ").title())

# Key params
priority_params = ["params.learning_rate", "params.num_epochs", "params.max_steps",
                   "params.lora_r", "params.sft_pairs", "params.base_model"]
for pc in priority_params:
    if pc in runs_df.columns:
        display_cols.append(pc)
        col_labels.append(pc.replace("params.", "").replace("_", " ").title())

# Start time
if "start_time" in runs_df.columns:
    display_cols.append("start_time")
    col_labels.append("Started")

# Build HTML table
table_rows = ""
for idx, row in runs_df.iterrows():
    cells = ""
    for col in display_cols:
        val = row.get(col, "")
        if pd.isna(val):
            val = "—"
        elif col == "start_time":
            try:
                if isinstance(val, (int, float)):
                    val = datetime.fromtimestamp(val / 1000).strftime("%b %d, %H:%M")
                else:
                    val = pd.to_datetime(val).strftime("%b %d, %H:%M")
            except Exception:
                val = str(val)[:16]
        elif col == "status":
            color = "#38ef7d" if val == "FINISHED" else "#f5576c" if val == "FAILED" else "#ffd93d"
            val = f'<span style="color:{color}; font-weight:600;">●</span> {val}'
        elif isinstance(val, float):
            val = f"{val:.6g}"
        cells += f"<td style='padding: 10px 14px; border-bottom: 1px solid #eee;'>{val}</td>"
    table_rows += f"<tr style='transition: background 0.2s;' onmouseover=\"this.style.background='#f8f9ff'\" onmouseout=\"this.style.background='white'\">{cells}</tr>"

header_cells = "".join(
    f"<th style='padding: 12px 14px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #667eea; border-bottom: 2px solid #667eea; background: #f8f9ff;'>{label}</th>"
    for label in col_labels
)

table_html = f"""
<div class="report-container">
    <div class="section-title">Run Comparison</div>
    <div style="overflow-x: auto; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
        <table style="width: 100%; border-collapse: collapse; font-size: 14px; background: white;">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    <p style="color: #999; font-size: 12px; margin-top: 8px;">Showing {len(runs_df)} most recent runs</p>
</div>
"""

displayHTML(table_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Training Loss Curves

# COMMAND ----------

import json

# Find all runs with step-level metrics (loss history)
loss_curves = {}
for _, row in runs_df.iterrows():
    run_id = row["run_id"]
    run_name = row.get("tags.mlflow.runName", run_id[:8])

    try:
        # Get loss metric history
        history = client.get_metric_history(run_id, "loss")
        if not history:
            history = client.get_metric_history(run_id, "train_loss")
        if not history:
            history = client.get_metric_history(run_id, "train/loss")

        if history and len(history) > 1:
            loss_curves[run_name] = [
                {"step": m.step, "value": m.value}
                for m in sorted(history, key=lambda x: x.step)
            ]
    except Exception:
        continue

if loss_curves:
    # Generate SVG chart
    colors = ["#667eea", "#f5576c", "#38ef7d", "#ffd93d", "#4facfe", "#f093fb", "#ff6b6b", "#48dbfb"]

    # Find global min/max for scaling
    all_steps = []
    all_values = []
    for points in loss_curves.values():
        all_steps.extend([p["step"] for p in points])
        all_values.extend([p["value"] for p in points])

    min_step, max_step = min(all_steps), max(all_steps)
    min_val, max_val = min(all_values), max(all_values)

    # Add padding
    val_range = max_val - min_val if max_val != min_val else 1
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val

    step_range = max_step - min_step if max_step != min_step else 1

    chart_w, chart_h = 900, 350
    pad_l, pad_r, pad_t, pad_b = 70, 30, 20, 50

    def scale_x(step):
        return pad_l + (step - min_step) / step_range * (chart_w - pad_l - pad_r)

    def scale_y(val):
        return pad_t + (1 - (val - min_val) / val_range) * (chart_h - pad_t - pad_b)

    # Grid lines
    grid_svg = ""
    num_y_ticks = 6
    for i in range(num_y_ticks + 1):
        y_val = min_val + (val_range * i / num_y_ticks)
        y_pos = scale_y(y_val)
        grid_svg += f'<line x1="{pad_l}" y1="{y_pos}" x2="{chart_w - pad_r}" y2="{y_pos}" stroke="#eee" stroke-width="1"/>'
        grid_svg += f'<text x="{pad_l - 8}" y="{y_pos + 4}" text-anchor="end" fill="#999" font-size="11">{y_val:.4g}</text>'

    num_x_ticks = 8
    for i in range(num_x_ticks + 1):
        x_val = min_step + (step_range * i / num_x_ticks)
        x_pos = scale_x(x_val)
        grid_svg += f'<line x1="{x_pos}" y1="{pad_t}" x2="{x_pos}" y2="{chart_h - pad_b}" stroke="#eee" stroke-width="1"/>'
        grid_svg += f'<text x="{x_pos}" y="{chart_h - pad_b + 18}" text-anchor="middle" fill="#999" font-size="11">{int(x_val)}</text>'

    # Plot lines
    lines_svg = ""
    legend_items = ""
    for i, (name, points) in enumerate(loss_curves.items()):
        color = colors[i % len(colors)]
        path_points = " ".join(f"{scale_x(p['step'])},{scale_y(p['value'])}" for p in points)
        lines_svg += f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>'

        # Start and end dots
        first = points[0]
        last = points[-1]
        lines_svg += f'<circle cx="{scale_x(first["step"])}" cy="{scale_y(first["value"])}" r="4" fill="{color}"/>'
        lines_svg += f'<circle cx="{scale_x(last["step"])}" cy="{scale_y(last["value"])}" r="4" fill="{color}"/>'
        lines_svg += f'<text x="{scale_x(last["step"]) + 8}" y="{scale_y(last["value"]) + 4}" fill="{color}" font-size="11" font-weight="600">{last["value"]:.4g}</text>'

        # Legend
        legend_items += f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-right: 20px;">
            <div style="width: 16px; height: 3px; background: {color}; border-radius: 2px;"></div>
            <span style="font-size: 12px; color: #444;">{name}</span>
        </div>"""

    chart_html = f"""
    <div class="report-container">
        <div class="section-title">Training Loss Curves</div>
        <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
            <svg width="{chart_w}" height="{chart_h}" style="display: block; margin: 0 auto;">
                {grid_svg}
                {lines_svg}
                <text x="{chart_w // 2}" y="{chart_h - 5}" text-anchor="middle" fill="#999" font-size="12">Training Step</text>
                <text x="15" y="{chart_h // 2}" text-anchor="middle" fill="#999" font-size="12" transform="rotate(-90, 15, {chart_h // 2})">Loss</text>
            </svg>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;">
                {legend_items}
            </div>
        </div>
    </div>
    """
    displayHTML(chart_html)
else:
    print("No step-level loss history found. Metrics may only be logged at run level.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Best Run Spotlight

# COMMAND ----------

# Find the best run by final loss
loss_col = None
for candidate in ["metrics.final_loss", "metrics.train_loss", "metrics.loss", "metrics.eval_loss"]:
    if candidate in runs_df.columns and runs_df[candidate].notna().any():
        loss_col = candidate
        break

if loss_col:
    best_idx = runs_df[loss_col].idxmin()
    best_run = runs_df.loc[best_idx]
    best_loss = best_run[loss_col]
    best_name = best_run.get("tags.mlflow.runName", best_run["run_id"][:8])

    # Gather all params for best run
    best_params_html = ""
    for pc in sorted(param_cols):
        val = best_run.get(pc, None)
        if pd.notna(val):
            label = pc.replace("params.", "").replace("_", " ").title()
            best_params_html += f"""
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                <span style="color: #666; font-size: 13px;">{label}</span>
                <span style="font-weight: 600; color: #1a1a2e; font-size: 13px;">{val}</span>
            </div>"""

    # Gather all metrics for best run
    best_metrics_html = ""
    for mc in sorted(metric_cols):
        val = best_run.get(mc, None)
        if pd.notna(val):
            label = mc.replace("metrics.", "").replace("_", " ").title()
            if isinstance(val, float):
                val_str = f"{val:.6g}"
            else:
                val_str = str(val)
            best_metrics_html += f"""
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                <span style="color: #666; font-size: 13px;">{label}</span>
                <span style="font-weight: 600; color: #1a1a2e; font-size: 13px;">{val_str}</span>
            </div>"""

    # Duration
    duration_str = "N/A"
    if "start_time" in best_run.index and "end_time" in best_run.index:
        try:
            start = best_run["start_time"]
            end = best_run["end_time"]
            if isinstance(start, (int, float)):
                start = datetime.fromtimestamp(start / 1000)
            if isinstance(end, (int, float)):
                end = datetime.fromtimestamp(end / 1000)
            duration = end - start
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m"
            else:
                duration_str = f"{minutes}m"
        except Exception:
            pass

    spotlight_html = f"""
    <div class="report-container">
        <div class="section-title">Best Run Spotlight</div>

        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 32px; color: white; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8;">Best Performing Run</div>
                    <div style="font-size: 28px; font-weight: 700; margin: 8px 0;">{best_name}</div>
                    <div style="font-size: 14px; opacity: 0.8;">Run ID: {best_run['run_id'][:12]}...</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8;">Final Loss</div>
                    <div style="font-size: 42px; font-weight: 700;">{best_loss:.4f}</div>
                    <div style="font-size: 14px; opacity: 0.8;">Duration: {duration_str}</div>
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;">Parameters</div>
                {best_params_html if best_params_html else '<p style="color: #999;">No parameters logged</p>'}
            </div>
            <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;">Metrics</div>
                {best_metrics_html if best_metrics_html else '<p style="color: #999;">No metrics logged</p>'}
            </div>
        </div>
    </div>
    """
    displayHTML(spotlight_html)
else:
    print("No loss metric found to determine best run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Parameter Impact Analysis

# COMMAND ----------

# Show how different parameter choices affected the final loss
if loss_col and len(runs_df) >= 2:
    # Find params that vary across runs
    varying_params = []
    for pc in param_cols:
        unique_vals = runs_df[pc].dropna().unique()
        if len(unique_vals) > 1:
            varying_params.append(pc)

    if varying_params:
        impact_rows = ""
        for pc in varying_params[:8]:
            label = pc.replace("params.", "").replace("_", " ").title()
            unique_vals = runs_df[pc].dropna().unique()

            # For each value, find the average loss
            val_summaries = []
            for val in sorted(unique_vals, key=str):
                mask = runs_df[pc] == val
                avg_loss = runs_df.loc[mask, loss_col].mean()
                count = mask.sum()
                if pd.notna(avg_loss):
                    val_summaries.append((str(val), avg_loss, count))

            if not val_summaries:
                continue

            best_val = min(val_summaries, key=lambda x: x[1])
            worst_val = max(val_summaries, key=lambda x: x[1])

            values_html = " ".join(
                f'<span style="display: inline-block; background: {"#667eea" if v == best_val[0] else "#f0f0f0"}; '
                f'color: {"white" if v == best_val[0] else "#444"}; padding: 4px 10px; border-radius: 20px; '
                f'font-size: 12px; margin: 2px;">{v} (loss: {l:.4g}, n={c})</span>'
                for v, l, c in val_summaries
            )

            impact_rows += f"""
            <div style="padding: 16px 0; border-bottom: 1px solid #f0f0f0;">
                <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 8px;">{label}</div>
                <div>{values_html}</div>
            </div>"""

        if impact_rows:
            impact_html = f"""
            <div class="report-container">
                <div class="section-title">Parameter Impact Analysis</div>
                <p style="color: #666; font-size: 14px;">How different parameter choices affected training loss. The highlighted value performed best.</p>
                <div style="background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
                    {impact_rows}
                </div>
            </div>
            """
            displayHTML(impact_html)
        else:
            print("Not enough data to show parameter impact.")
    else:
        print("All runs used the same parameters — no variation to analyze.")
else:
    print("Need at least 2 runs with loss metrics for parameter impact analysis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run Timeline

# COMMAND ----------

if "start_time" in runs_df.columns and "end_time" in runs_df.columns and len(runs_df) > 0:
    timeline_items = ""
    colors = ["#667eea", "#f5576c", "#38ef7d", "#ffd93d", "#4facfe", "#f093fb"]

    for i, (_, row) in enumerate(runs_df.iterrows()):
        run_name = row.get("tags.mlflow.runName", row["run_id"][:8])
        training_type = row.get("params.training_type", "Unknown")
        status = row.get("status", "UNKNOWN")

        try:
            start = row["start_time"]
            end = row["end_time"]
            if isinstance(start, (int, float)):
                start = datetime.fromtimestamp(start / 1000)
            if isinstance(end, (int, float)):
                end = datetime.fromtimestamp(end / 1000)
            duration = end - start
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            date_str = start.strftime("%b %d, %Y at %H:%M")
        except Exception:
            duration_str = "N/A"
            date_str = "N/A"

        loss_val = "—"
        for lc in ["metrics.final_loss", "metrics.train_loss", "metrics.loss"]:
            if lc in row.index and pd.notna(row.get(lc)):
                loss_val = f"{row[lc]:.4g}"
                break

        color = colors[i % len(colors)]
        status_icon = "✓" if status == "FINISHED" else "✗" if status == "FAILED" else "⟳"
        status_color = "#38ef7d" if status == "FINISHED" else "#f5576c" if status == "FAILED" else "#ffd93d"

        timeline_items += f"""
        <div style="display: flex; gap: 20px; margin-bottom: 4px;">
            <div style="display: flex; flex-direction: column; align-items: center; min-width: 20px;">
                <div style="width: 14px; height: 14px; border-radius: 50%; background: {color}; border: 3px solid white; box-shadow: 0 0 0 2px {color}; z-index: 1;"></div>
                <div style="width: 2px; flex: 1; background: #e0e0e0; margin-top: 4px;"></div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 16px 20px; flex: 1; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: #1a1a2e; font-size: 15px;">{run_name}</span>
                        <span style="background: #f0f0f0; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; color: #666;">{training_type}</span>
                        <span style="color: {status_color}; margin-left: 8px; font-weight: 600;">{status_icon}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 13px; color: #999;">{date_str}</span>
                    </div>
                </div>
                <div style="display: flex; gap: 24px; margin-top: 8px; font-size: 13px; color: #666;">
                    <span>Duration: <strong style="color: #1a1a2e;">{duration_str}</strong></span>
                    <span>Loss: <strong style="color: #1a1a2e;">{loss_val}</strong></span>
                </div>
            </div>
        </div>"""

    timeline_html = f"""
    <div class="report-container">
        <div class="section-title">Run Timeline</div>
        <div style="padding: 8px 0;">
            {timeline_items}
        </div>
    </div>
    """
    displayHTML(timeline_html)
else:
    print("No timing data available for timeline.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Artifacts & Model Details

# COMMAND ----------

# List artifacts for recent runs
artifacts_html = ""
for _, row in runs_df.head(5).iterrows():
    run_id = row["run_id"]
    run_name = row.get("tags.mlflow.runName", run_id[:8])

    try:
        artifacts = client.list_artifacts(run_id)
        if not artifacts:
            continue

        artifact_list = ""
        total_size = 0
        for art in artifacts:
            size_str = ""
            if art.file_size:
                total_size += art.file_size
                if art.file_size > 1_000_000:
                    size_str = f"{art.file_size / 1_000_000:.1f} MB"
                elif art.file_size > 1_000:
                    size_str = f"{art.file_size / 1_000:.1f} KB"
                else:
                    size_str = f"{art.file_size} B"

            icon = "📁" if art.is_dir else "📄"
            artifact_list += f"""
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f5f5f5; font-size: 13px;">
                <span>{icon} {art.path}</span>
                <span style="color: #999;">{size_str}</span>
            </div>"""

        size_total_str = f"{total_size / 1_000_000:.1f} MB" if total_size > 1_000_000 else f"{total_size / 1_000:.1f} KB" if total_size > 1_000 else ""

        artifacts_html += f"""
        <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; color: #1a1a2e;">{run_name}</span>
                <span style="font-size: 12px; color: #999;">{len(artifacts)} artifacts {('(' + size_total_str + ')') if size_total_str else ''}</span>
            </div>
            {artifact_list}
        </div>"""

    except Exception:
        continue

if artifacts_html:
    full_artifacts_html = f"""
    <div class="report-container">
        <div class="section-title">Artifacts & Saved Models</div>
        {artifacts_html}
    </div>
    """
    displayHTML(full_artifacts_html)
else:
    print("No artifacts found for recent runs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Quick Metrics Summary

# COMMAND ----------

# Final summary card showing key takeaways
if loss_col and len(runs_df) > 0:
    best_loss = runs_df[loss_col].min()
    worst_loss = runs_df[loss_col].max()
    avg_loss = runs_df[loss_col].mean()
    best_run_name = runs_df.loc[runs_df[loss_col].idxmin()].get("tags.mlflow.runName", "Unknown")

    improvement = ((worst_loss - best_loss) / worst_loss * 100) if worst_loss != 0 else 0

    summary_html = f"""
    <div class="report-container">
        <div class="section-title">Key Takeaways</div>
        <div style="background: white; border-radius: 12px; padding: 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; text-align: center;">
                <div>
                    <div style="font-size: 32px; font-weight: 700; color: #38ef7d;">{best_loss:.4f}</div>
                    <div style="font-size: 13px; color: #999; margin-top: 4px;">Best Loss</div>
                    <div style="font-size: 12px; color: #666;">({best_run_name})</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 700; color: #f5576c;">{worst_loss:.4f}</div>
                    <div style="font-size: 13px; color: #999; margin-top: 4px;">Worst Loss</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 700; color: #667eea;">{avg_loss:.4f}</div>
                    <div style="font-size: 13px; color: #999; margin-top: 4px;">Average Loss</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 700; color: #4facfe;">{improvement:.1f}%</div>
                    <div style="font-size: 13px; color: #999; margin-top: 4px;">Improvement Range</div>
                    <div style="font-size: 12px; color: #666;">(worst to best)</div>
                </div>
            </div>
        </div>
    </div>
    """
    displayHTML(summary_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Compare runs** — Click on any run in the MLflow UI to see full details
# MAGIC 2. **Register best model** — Promote the best run's model to the Model Registry
# MAGIC 3. **Iterate** — Adjust hyperparameters and rerun training
# MAGIC 4. **Share** — Export this notebook as HTML for stakeholders
