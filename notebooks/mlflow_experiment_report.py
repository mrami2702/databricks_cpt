# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Report
# MAGIC
# MAGIC A visual summary of a single training run — loss curve, parameters,
# MAGIC metrics, and artifacts — all pulled from MLflow and displayed with
# MAGIC rich HTML formatting.
# MAGIC
# MAGIC **No model loading needed.** This notebook only reads experiment data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# TODO: Fill in your MLflow experiment name
EXPERIMENT_NAME = ""  # e.g. "/Shared/cpt-sft-mistral"

# Which run to spotlight — set to None to auto-pick the latest completed run
# You can also paste a specific run ID here
RUN_ID = None  # e.g. "abc123def456"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Run Data

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

client = MlflowClient()

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Check your experiment name.")

if RUN_ID:
    run = client.get_run(RUN_ID)
else:
    # Get the latest completed run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    if len(runs) == 0:
        raise ValueError("No completed runs found in this experiment.")
    RUN_ID = runs.iloc[0]["run_id"]
    run = client.get_run(RUN_ID)

run_name = run.data.tags.get("mlflow.runName", RUN_ID[:12])
metrics = run.data.metrics
params = run.data.params
tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}

# Duration
start_time = datetime.fromtimestamp(run.info.start_time / 1000)
end_time = datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
if end_time:
    duration = end_time - start_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
else:
    duration_str = "In progress..."

print(f"Run: {run_name}")
print(f"ID: {RUN_ID}")
print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M')}")
print(f"Duration: {duration_str}")
print(f"Metrics: {len(metrics)}")
print(f"Parameters: {len(params)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Run Overview

# COMMAND ----------

# Training type
training_type = params.get("training_type", "Unknown")
base_model = params.get("base_model", "Unknown")
num_samples = params.get("sft_pairs", params.get("train_samples", "—"))

# Find the "headline" loss metric
headline_loss = None
for key in ["final_loss", "train_loss", "loss", "eval_loss"]:
    if key in metrics:
        headline_loss = metrics[key]
        break

loss_display = f"{headline_loss:.4f}" if headline_loss is not None else "—"

overview_html = f"""
<style>
    .rpt {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 0 auto; }}
    .hero {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 40px; color: white;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 24px;
    }}
    .hero-title {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8; }}
    .hero-name {{ font-size: 32px; font-weight: 700; margin: 8px 0 4px 0; }}
    .hero-sub {{ font-size: 14px; opacity: 0.7; }}
    .stat-row {{ display: flex; gap: 24px; margin-top: 28px; flex-wrap: wrap; }}
    .stat {{
        background: rgba(255,255,255,0.15); border-radius: 12px;
        padding: 16px 24px; min-width: 140px; backdrop-filter: blur(10px);
    }}
    .stat-val {{ font-size: 28px; font-weight: 700; }}
    .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8; margin-top: 4px; }}
    .section {{ font-size: 20px; font-weight: 600; color: #1a1a2e; margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 3px solid #667eea; }}
    .card {{
        background: white; border-radius: 12px; padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 16px;
    }}
    .row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #f0f0f0; }}
    .row:last-child {{ border-bottom: none; }}
    .row-label {{ color: #666; font-size: 14px; }}
    .row-value {{ font-weight: 600; color: #1a1a2e; font-size: 14px; }}
</style>

<div class="rpt">
    <div class="hero">
        <div class="hero-title">{training_type} Training Run</div>
        <div class="hero-name">{run_name}</div>
        <div class="hero-sub">Run ID: {RUN_ID} &nbsp;|&nbsp; {start_time.strftime('%B %d, %Y at %I:%M %p')}</div>

        <div class="stat-row">
            <div class="stat">
                <div class="stat-val">{loss_display}</div>
                <div class="stat-label">Final Loss</div>
            </div>
            <div class="stat">
                <div class="stat-val">{duration_str}</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat">
                <div class="stat-val">{num_samples}</div>
                <div class="stat-label">Training Samples</div>
            </div>
            <div class="stat">
                <div class="stat-val">{base_model}</div>
                <div class="stat-label">Base Model</div>
            </div>
        </div>
    </div>
</div>
"""

displayHTML(overview_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Training Loss Curve

# COMMAND ----------

# Pull step-level loss history
loss_history = []
for metric_key in ["loss", "train_loss", "train/loss"]:
    try:
        history = client.get_metric_history(RUN_ID, metric_key)
        if history and len(history) > 1:
            loss_history = sorted(history, key=lambda x: x.step)
            break
    except Exception:
        continue

if loss_history:
    steps = [m.step for m in loss_history]
    values = [m.value for m in loss_history]

    min_step, max_step = min(steps), max(steps)
    min_val, max_val = min(values), max(values)
    val_range = max_val - min_val if max_val != min_val else 1
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val
    step_range = max_step - min_step if max_step != min_step else 1

    w, h = 900, 320
    pl, pr, pt, pb = 70, 40, 30, 50

    def sx(s):
        return pl + (s - min_step) / step_range * (w - pl - pr)
    def sy(v):
        return pt + (1 - (v - min_val) / val_range) * (h - pt - pb)

    # Grid
    grid = ""
    for i in range(7):
        yv = min_val + val_range * i / 6
        yp = sy(yv)
        grid += f'<line x1="{pl}" y1="{yp}" x2="{w-pr}" y2="{yp}" stroke="#f0f0f0" stroke-width="1"/>'
        grid += f'<text x="{pl-10}" y="{yp+4}" text-anchor="end" fill="#aaa" font-size="11" font-family="sans-serif">{yv:.4g}</text>'

    for i in range(9):
        xv = min_step + step_range * i / 8
        xp = sx(xv)
        grid += f'<line x1="{xp}" y1="{pt}" x2="{xp}" y2="{h-pb}" stroke="#f0f0f0" stroke-width="1"/>'
        grid += f'<text x="{xp}" y="{h-pb+18}" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">{int(xv)}</text>'

    # Loss line
    points = " ".join(f"{sx(s)},{sy(v)}" for s, v in zip(steps, values))

    # Gradient fill under curve
    fill_points = f"{sx(steps[0])},{sy(min_val)} " + points + f" {sx(steps[-1])},{sy(min_val)}"

    # Start/end markers
    start_val = values[0]
    end_val = values[-1]
    drop_pct = (start_val - end_val) / start_val * 100 if start_val != 0 else 0

    chart_html = f"""
    <div class="rpt">
        <div class="section">Training Loss</div>
        <div class="card" style="padding: 28px;">
            <svg width="{w}" height="{h}" style="display: block; margin: 0 auto;">
                <defs>
                    <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#667eea" stop-opacity="0.15"/>
                        <stop offset="100%" stop-color="#667eea" stop-opacity="0.01"/>
                    </linearGradient>
                </defs>
                {grid}
                <polygon points="{fill_points}" fill="url(#fillGrad)"/>
                <polyline points="{points}" fill="none" stroke="#667eea" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="{sx(steps[0])}" cy="{sy(start_val)}" r="5" fill="#f5576c"/>
                <text x="{sx(steps[0])+10}" y="{sy(start_val)-8}" fill="#f5576c" font-size="12" font-weight="600" font-family="sans-serif">Start: {start_val:.4f}</text>
                <circle cx="{sx(steps[-1])}" cy="{sy(end_val)}" r="5" fill="#38ef7d"/>
                <text x="{sx(steps[-1])-10}" y="{sy(end_val)-8}" fill="#38ef7d" font-size="12" font-weight="600" text-anchor="end" font-family="sans-serif">End: {end_val:.4f}</text>
                <text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif">Training Step</text>
                <text x="14" y="{h//2}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif" transform="rotate(-90, 14, {h//2})">Loss</text>
            </svg>
            <div style="text-align: center; margin-top: 16px; padding-top: 16px; border-top: 1px solid #f0f0f0;">
                <span style="font-size: 14px; color: #666;">Loss dropped <strong style="color: #38ef7d;">{drop_pct:.1f}%</strong> over <strong>{len(steps)}</strong> logged steps</span>
            </div>
        </div>
    </div>
    """
    displayHTML(chart_html)
else:
    print("No step-level loss history found for this run.")
    print("Loss may only be logged at the end of training.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Parameters & Metrics

# COMMAND ----------

# Parameters
params_html = ""
for k, v in sorted(params.items()):
    params_html += f"""
    <div class="row">
        <span class="row-label">{k.replace('_', ' ').title()}</span>
        <span class="row-value">{v}</span>
    </div>"""

# Metrics
metrics_html = ""
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        v_str = f"{v:.6g}"
    else:
        v_str = str(v)
    metrics_html += f"""
    <div class="row">
        <span class="row-label">{k.replace('_', ' ').title()}</span>
        <span class="row-value">{v_str}</span>
    </div>"""

# Tags
tags_html = ""
if tags:
    for k, v in sorted(tags.items()):
        tags_html += f"""
        <div class="row">
            <span class="row-label">{k}</span>
            <span class="row-value">{v}</span>
        </div>"""

details_html = f"""
<div class="rpt">
    <div class="section">Run Details</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <div class="card">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;">
                    Parameters
                    <span style="font-size: 12px; color: #999; font-weight: 400; margin-left: 8px;">{len(params)} logged</span>
                </div>
                {params_html if params_html else '<p style="color: #999;">No parameters logged</p>'}
            </div>
        </div>
        <div>
            <div class="card">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;">
                    Metrics
                    <span style="font-size: 12px; color: #999; font-weight: 400; margin-left: 8px;">{len(metrics)} tracked</span>
                </div>
                {metrics_html if metrics_html else '<p style="color: #999;">No metrics logged</p>'}
            </div>
            {'<div class="card"><div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px;">Tags</div>' + tags_html + '</div>' if tags_html else ''}
        </div>
    </div>
</div>
"""

displayHTML(details_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Eval Loss Curve (if available)

# COMMAND ----------

# Check for eval loss history
eval_history = []
for metric_key in ["eval_loss", "eval/loss"]:
    try:
        history = client.get_metric_history(RUN_ID, metric_key)
        if history and len(history) > 1:
            eval_history = sorted(history, key=lambda x: x.step)
            break
    except Exception:
        continue

if eval_history and loss_history:
    # Overlay train + eval loss
    all_steps = [m.step for m in loss_history] + [m.step for m in eval_history]
    all_vals = [m.value for m in loss_history] + [m.value for m in eval_history]

    min_step, max_step = min(all_steps), max(all_steps)
    min_val, max_val = min(all_vals), max(all_vals)
    val_range = max_val - min_val if max_val != min_val else 1
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val
    step_range = max_step - min_step if max_step != min_step else 1

    w, h = 900, 320
    pl, pr, pt, pb = 70, 40, 30, 50

    def sx2(s):
        return pl + (s - min_step) / step_range * (w - pl - pr)
    def sy2(v):
        return pt + (1 - (v - min_val) / val_range) * (h - pt - pb)

    grid2 = ""
    for i in range(7):
        yv = min_val + val_range * i / 6
        yp = sy2(yv)
        grid2 += f'<line x1="{pl}" y1="{yp}" x2="{w-pr}" y2="{yp}" stroke="#f0f0f0" stroke-width="1"/>'
        grid2 += f'<text x="{pl-10}" y="{yp+4}" text-anchor="end" fill="#aaa" font-size="11" font-family="sans-serif">{yv:.4g}</text>'

    for i in range(9):
        xv = min_step + step_range * i / 8
        xp = sx2(xv)
        grid2 += f'<text x="{xp}" y="{h-pb+18}" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">{int(xv)}</text>'

    train_pts = " ".join(f"{sx2(m.step)},{sy2(m.value)}" for m in loss_history)
    eval_pts = " ".join(f"{sx2(m.step)},{sy2(m.value)}" for m in eval_history)

    eval_chart_html = f"""
    <div class="rpt">
        <div class="section">Train vs Eval Loss</div>
        <div class="card" style="padding: 28px;">
            <svg width="{w}" height="{h}" style="display: block; margin: 0 auto;">
                {grid2}
                <polyline points="{train_pts}" fill="none" stroke="#667eea" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>
                <polyline points="{eval_pts}" fill="none" stroke="#f5576c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6,4"/>
                <text x="{w//2}" y="{h-5}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif">Training Step</text>
                <text x="14" y="{h//2}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif" transform="rotate(-90, 14, {h//2})">Loss</text>
            </svg>
            <div style="display: flex; justify-content: center; gap: 32px; margin-top: 16px; padding-top: 16px; border-top: 1px solid #f0f0f0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 3px; background: #667eea; border-radius: 2px;"></div>
                    <span style="font-size: 13px; color: #666;">Train Loss</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 3px; background: #f5576c; border-radius: 2px; border-top: 2px dashed #f5576c;"></div>
                    <span style="font-size: 13px; color: #666;">Eval Loss</span>
                </div>
            </div>
        </div>
    </div>
    """
    displayHTML(eval_chart_html)
elif eval_history:
    print(f"Eval loss found ({len(eval_history)} points) but no train loss to overlay.")
else:
    print("No eval loss history found — eval may not have been logged per step.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Artifacts

# COMMAND ----------

try:
    artifacts = client.list_artifacts(RUN_ID)

    if artifacts:
        artifact_rows = ""
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

            icon = "&#128193;" if art.is_dir else "&#128196;"
            bg = "#f8f9ff" if art.is_dir else "white"
            artifact_rows += f"""
            <div class="row" style="background: {bg}; padding: 12px 16px; border-radius: 8px; margin-bottom: 4px;">
                <span class="row-label" style="font-size: 14px;">{icon}&nbsp; {art.path}</span>
                <span class="row-value" style="color: #999; font-weight: 400;">{size_str}</span>
            </div>"""

        total_str = f"{total_size / 1_000_000:.1f} MB" if total_size > 1_000_000 else f"{total_size / 1_000:.1f} KB"

        artifacts_html = f"""
        <div class="rpt">
            <div class="section">Saved Artifacts</div>
            <div class="card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
                    <span style="color: #666; font-size: 13px;">{len(artifacts)} items</span>
                    <span style="color: #666; font-size: 13px;">Total: {total_str}</span>
                </div>
                {artifact_rows}
            </div>
        </div>
        """
        displayHTML(artifacts_html)
    else:
        print("No artifacts saved for this run.")
except Exception as e:
    print(f"Could not load artifacts: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Summary

# COMMAND ----------

# Final summary
lr = params.get("learning_rate", "—")
epochs = params.get("num_epochs", "—")
max_steps_val = params.get("max_steps", "—")
lora_r = params.get("lora_r", "—")
total_steps_val = metrics.get("total_steps", "—")
if isinstance(total_steps_val, float):
    total_steps_val = int(total_steps_val)

summary_html = f"""
<div class="rpt">
    <div class="section">Summary</div>
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; padding: 32px; color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Run</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{run_name}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Training Type</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{training_type}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Base Model</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{base_model}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Learning Rate</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{lr}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Epochs / Max Steps</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{epochs} epochs / {max_steps_val} max steps</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">LoRA Rank</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{lora_r}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Steps Completed</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{total_steps_val}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); font-size: 13px;">Duration</td>
                <td style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-weight: 600; text-align: right;">{duration_str}</td>
            </tr>
            <tr>
                <td style="padding: 12px 0; color: rgba(255,255,255,0.6); font-size: 13px;">Final Loss</td>
                <td style="padding: 12px 0; font-weight: 700; text-align: right; font-size: 20px; color: #38ef7d;">{loss_display}</td>
            </tr>
        </table>
    </div>
</div>
"""

displayHTML(summary_html)
