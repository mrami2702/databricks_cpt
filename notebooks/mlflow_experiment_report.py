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

# Optional: compare against another run (set to None to skip comparison)
COMPARE_RUN_ID = None  # e.g. "xyz789ghi012"

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

# --- Plain-English explanations for parameters and metrics ---
# Think of these as the "what does this knob do?" labels on a machine

param_explanations = {
    "learning_rate": "How fast the model adjusts its knowledge — like a volume dial. Too high and it forgets what it knew, too low and it barely learns anything new.",
    "num_epochs": "How many times the model reads through the entire dataset — like re-reading a textbook multiple times to absorb more.",
    "max_steps": "A hard cap on training updates. Training stops here even if there are more epochs left — like a timer on an experiment.",
    "lora_r": "LoRA rank — controls how much new capacity the model gets. Higher rank = more room to learn, but also more memory. Like choosing the size of a notebook to write in.",
    "lora_alpha": "Scaling factor for LoRA. Usually 2x the rank. Controls how strongly the new learned weights influence predictions.",
    "lora_dropout": "Randomly ignores some adapter connections during training to prevent overfitting — like intentionally leaving out flashcards to test real understanding.",
    "per_device_train_batch_size": "How many examples the model sees before updating its weights. Larger = smoother learning but more GPU memory.",
    "gradient_accumulation_steps": "Simulates a larger batch by accumulating updates across multiple mini-batches — a memory-saving trick that doesn't sacrifice quality.",
    "training_type": "CPT = learning domain vocabulary (reading). SFT = learning to answer questions (practicing Q&A).",
    "base_model": "The pre-trained model we started from — its existing knowledge before we taught it anything new.",
    "sft_pairs": "Number of question-answer pairs used for training — the size of the study guide.",
    "val_pairs": "Held-out Q&A pairs used to check if the model is actually learning vs. just memorizing.",
    "bf16": "Uses half-precision math (bfloat16) to cut memory usage in half with minimal accuracy loss — like rounding to fewer decimal places.",
    "optim": "The optimization algorithm. paged_adamw_8bit is memory-efficient — it figures out the best direction to adjust weights each step.",
    "weight_decay": "Gently shrinks weights toward zero each step to prevent overfitting — like adding friction to keep the model from over-correcting.",
    "warmup_ratio": "Starts with a very low learning rate and gradually increases. Like warming up before a workout — avoids damaging early updates.",
    "lr_scheduler_type": "How the learning rate changes over time. 'cosine' starts strong and gradually slows down, like decelerating into a parking spot.",
    "max_seq_length": "Maximum number of tokens (roughly words) the model processes at once. Longer = more context but more memory.",
    "max_grad_norm": "Clips large gradient updates to prevent training instability — like a safety valve on a pressure cooker.",
    "save_steps": "How often a checkpoint is saved. If training crashes, you can resume from the last checkpoint.",
    "save_total_limit": "Only keeps this many checkpoints on disk — older ones are deleted to save storage.",
    "train_samples": "Total number of training examples the model learned from.",
}

metric_explanations = {
    "final_loss": "The model's error rate at the end of training. Lower = better. Think of it as the score on a final exam — closer to 0 means fewer mistakes.",
    "train_loss": "Same as final loss — how wrong the model's predictions were on the training data.",
    "loss": "How wrong the model's predictions are. Lower = better. This is the primary number to watch.",
    "eval_loss": "Error rate on held-out data the model never trained on. Measures real-world performance — like a pop quiz vs. homework.",
    "total_steps": "Total number of weight updates performed. Each step = one batch of data processed and learned from.",
    "train_runtime": "Total wall-clock time spent training in seconds.",
    "train_samples_per_second": "Throughput — how many examples the model processes per second. Higher = more efficient use of GPU.",
    "train_steps_per_second": "How many weight update steps happen per second.",
    "epoch": "How many complete passes through the data were completed.",
}

# Parameters with explanations
params_html = ""
for k, v in sorted(params.items()):
    label = k.replace('_', ' ').title()
    explanation = param_explanations.get(k, "")
    desc_html = f'<div style="font-size: 11px; color: #999; margin-top: 2px; line-height: 1.4;">{explanation}</div>' if explanation else ""
    params_html += f"""
    <div style="padding: 12px 0; border-bottom: 1px solid #f0f0f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <span class="row-label">{label}</span>
            <span class="row-value" style="white-space: nowrap; margin-left: 12px;">{v}</span>
        </div>
        {desc_html}
    </div>"""

# Metrics with explanations
metrics_html = ""
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        v_str = f"{v:.6g}"
    else:
        v_str = str(v)
    label = k.replace('_', ' ').title()
    explanation = metric_explanations.get(k, "")
    desc_html = f'<div style="font-size: 11px; color: #999; margin-top: 2px; line-height: 1.4;">{explanation}</div>' if explanation else ""
    metrics_html += f"""
    <div style="padding: 12px 0; border-bottom: 1px solid #f0f0f0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <span class="row-label">{label}</span>
            <span class="row-value" style="white-space: nowrap; margin-left: 12px;">{v_str}</span>
        </div>
        {desc_html}
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
    <p style="color: #666; font-size: 13px; margin-top: -8px;">Each parameter and metric includes a plain-English explanation of what it means and why it matters.</p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <div class="card">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 4px;">
                    Parameters
                    <span style="font-size: 12px; color: #999; font-weight: 400; margin-left: 8px;">{len(params)} logged</span>
                </div>
                <div style="font-size: 12px; color: #999; margin-bottom: 12px;">The settings and choices that went INTO training — the recipe.</div>
                {params_html if params_html else '<p style="color: #999;">No parameters logged</p>'}
            </div>
        </div>
        <div>
            <div class="card">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 4px;">
                    Metrics
                    <span style="font-size: 12px; color: #999; font-weight: 400; margin-left: 8px;">{len(metrics)} tracked</span>
                </div>
                <div style="font-size: 12px; color: #999; margin-bottom: 12px;">The scores that came OUT of training — the taste test results.</div>
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Run Comparison (Optional)
# MAGIC
# MAGIC Set `COMPARE_RUN_ID` in the config above to compare the spotlight run
# MAGIC against another run — side by side, parameter by parameter, metric by metric.

# COMMAND ----------

if COMPARE_RUN_ID:
    compare_run = client.get_run(COMPARE_RUN_ID)
    compare_name = compare_run.data.tags.get("mlflow.runName", COMPARE_RUN_ID[:12])
    compare_metrics = compare_run.data.metrics
    compare_params = compare_run.data.params

    # Duration of comparison run
    c_start = datetime.fromtimestamp(compare_run.info.start_time / 1000)
    c_end = datetime.fromtimestamp(compare_run.info.end_time / 1000) if compare_run.info.end_time else None
    if c_end:
        c_dur = c_end - c_start
        c_hours = int(c_dur.total_seconds() // 3600)
        c_mins = int((c_dur.total_seconds() % 3600) // 60)
        c_duration_str = f"{c_hours}h {c_mins}m" if c_hours > 0 else f"{c_mins}m"
    else:
        c_duration_str = "—"

    # --- Header: side by side run names ---
    c_training_type = compare_params.get("training_type", "Unknown")

    # Find headline loss for comparison run
    c_loss = None
    for key in ["final_loss", "train_loss", "loss", "eval_loss"]:
        if key in compare_metrics:
            c_loss = compare_metrics[key]
            break
    c_loss_display = f"{c_loss:.4f}" if c_loss is not None else "—"

    # Determine winner for loss
    if headline_loss is not None and c_loss is not None:
        if headline_loss < c_loss:
            a_loss_color = "#38ef7d"
            b_loss_color = "#f5576c"
            winner_text = f"{run_name} wins by {((c_loss - headline_loss) / c_loss * 100):.1f}%"
        elif c_loss < headline_loss:
            a_loss_color = "#f5576c"
            b_loss_color = "#38ef7d"
            winner_text = f"{compare_name} wins by {((headline_loss - c_loss) / headline_loss * 100):.1f}%"
        else:
            a_loss_color = "#667eea"
            b_loss_color = "#667eea"
            winner_text = "Tied"
    else:
        a_loss_color = "#fff"
        b_loss_color = "#fff"
        winner_text = ""

    header_html = f"""
    <div class="rpt">
        <div class="section">Run Comparison</div>
        <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 0; margin-bottom: 24px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px 0 0 16px; padding: 28px; color: white; text-align: center;">
                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7;">Run A</div>
                <div style="font-size: 22px; font-weight: 700; margin: 6px 0;">{run_name}</div>
                <div style="font-size: 12px; opacity: 0.7;">{training_type}</div>
                <div style="font-size: 36px; font-weight: 700; margin-top: 16px; color: {a_loss_color};">{loss_display}</div>
                <div style="font-size: 11px; opacity: 0.7;">Final Loss</div>
                <div style="font-size: 13px; margin-top: 8px; opacity: 0.8;">{duration_str}</div>
            </div>
            <div style="background: #1a1a2e; display: flex; align-items: center; justify-content: center; padding: 0 20px; color: white;">
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 700;">VS</div>
                    <div style="font-size: 11px; color: #999; margin-top: 4px; max-width: 100px;">{winner_text}</div>
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 0 16px 16px 0; padding: 28px; color: white; text-align: center;">
                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7;">Run B</div>
                <div style="font-size: 22px; font-weight: 700; margin: 6px 0;">{compare_name}</div>
                <div style="font-size: 12px; opacity: 0.7;">{c_training_type}</div>
                <div style="font-size: 36px; font-weight: 700; margin-top: 16px; color: {b_loss_color};">{c_loss_display}</div>
                <div style="font-size: 11px; opacity: 0.7;">Final Loss</div>
                <div style="font-size: 13px; margin-top: 8px; opacity: 0.8;">{c_duration_str}</div>
            </div>
        </div>
    """
    displayHTML(header_html)

    # --- Parameter diff ---
    all_param_keys = sorted(set(list(params.keys()) + list(compare_params.keys())))

    diff_rows = ""
    same_rows = ""
    for k in all_param_keys:
        a_val = params.get(k, "—")
        b_val = compare_params.get(k, "—")
        label = k.replace("_", " ").title()
        is_different = str(a_val) != str(b_val)

        if is_different:
            diff_rows += f"""
            <tr style="background: #fff8f0;">
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-size: 13px; color: #666;">{label}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-weight: 600; text-align: center; color: #667eea;">{a_val}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-weight: 600; text-align: center; color: #11998e;">{b_val}</td>
            </tr>"""
        else:
            same_rows += f"""
            <tr>
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-size: 13px; color: #666;">{label}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; text-align: center; color: #999;">{a_val}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; text-align: center; color: #999;">{b_val}</td>
            </tr>"""

    param_diff_html = f"""
    <div class="rpt">
        <div class="card">
            <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 4px;">Parameter Comparison</div>
            <div style="font-size: 12px; color: #999; margin-bottom: 16px;">Highlighted rows show parameters that differ — these are the ingredients you changed between runs.</div>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="padding: 10px 14px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; border-bottom: 2px solid #eee;">Parameter</th>
                        <th style="padding: 10px 14px; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #667eea; border-bottom: 2px solid #eee;">{run_name}</th>
                        <th style="padding: 10px 14px; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #11998e; border-bottom: 2px solid #eee;">{compare_name}</th>
                    </tr>
                </thead>
                <tbody>
                    {diff_rows}
                    {same_rows}
                </tbody>
            </table>
        </div>
    </div>
    """
    displayHTML(param_diff_html)

    # --- Metric comparison ---
    all_metric_keys = sorted(set(list(metrics.keys()) + list(compare_metrics.keys())))

    # Metrics where lower is better
    lower_better = {"loss", "train_loss", "final_loss", "eval_loss", "rmse", "mse", "mae",
                    "train_runtime", "error", "log_loss"}

    metric_rows = ""
    for k in all_metric_keys:
        a_val = metrics.get(k)
        b_val = compare_metrics.get(k)
        label = k.replace("_", " ").title()

        a_str = f"{a_val:.6g}" if isinstance(a_val, float) else (str(a_val) if a_val is not None else "—")
        b_str = f"{b_val:.6g}" if isinstance(b_val, float) else (str(b_val) if b_val is not None else "—")

        # Determine winner
        a_style = ""
        b_style = ""
        delta_html = ""
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)) and a_val is not None and b_val is not None:
            is_lower_better = any(t in k.lower() for t in lower_better)
            if is_lower_better:
                if a_val < b_val:
                    a_style = "color: #38ef7d; font-weight: 700;"
                    pct = (b_val - a_val) / abs(b_val) * 100 if b_val != 0 else 0
                    delta_html = f'<span style="color: #38ef7d; font-size: 11px;"> ({pct:.1f}% better)</span>'
                elif b_val < a_val:
                    b_style = "color: #38ef7d; font-weight: 700;"
                    pct = (a_val - b_val) / abs(a_val) * 100 if a_val != 0 else 0
                    delta_html = f'<span style="color: #38ef7d; font-size: 11px;"> ({pct:.1f}% better)</span>'
            else:
                if a_val > b_val:
                    a_style = "color: #38ef7d; font-weight: 700;"
                    pct = (a_val - b_val) / abs(b_val) * 100 if b_val != 0 else 0
                    delta_html = f'<span style="color: #38ef7d; font-size: 11px;"> ({pct:.1f}% better)</span>'
                elif b_val > a_val:
                    b_style = "color: #38ef7d; font-weight: 700;"
                    pct = (b_val - a_val) / abs(a_val) * 100 if a_val != 0 else 0
                    delta_html = f'<span style="color: #38ef7d; font-size: 11px;"> ({pct:.1f}% better)</span>'

        metric_rows += f"""
        <tr>
            <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-size: 13px; color: #666;">{label}{delta_html}</td>
            <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; text-align: center; {a_style}">{a_str}</td>
            <td style="padding: 10px 14px; border-bottom: 1px solid #f0f0f0; text-align: center; {b_style}">{b_str}</td>
        </tr>"""

    metric_diff_html = f"""
    <div class="rpt">
        <div class="card">
            <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 4px;">Metric Comparison</div>
            <div style="font-size: 12px; color: #999; margin-bottom: 16px;">The winning value is highlighted in green. For loss/error metrics, lower is better. For accuracy metrics, higher is better.</div>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="padding: 10px 14px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; border-bottom: 2px solid #eee;">Metric</th>
                        <th style="padding: 10px 14px; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #667eea; border-bottom: 2px solid #eee;">{run_name}</th>
                        <th style="padding: 10px 14px; text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #11998e; border-bottom: 2px solid #eee;">{compare_name}</th>
                    </tr>
                </thead>
                <tbody>{metric_rows}</tbody>
            </table>
        </div>
    </div>
    """
    displayHTML(metric_diff_html)

    # --- Loss curves overlay ---
    compare_loss_history = []
    for metric_key in ["loss", "train_loss", "train/loss"]:
        try:
            ch = client.get_metric_history(COMPARE_RUN_ID, metric_key)
            if ch and len(ch) > 1:
                compare_loss_history = sorted(ch, key=lambda x: x.step)
                break
        except Exception:
            continue

    if loss_history and compare_loss_history:
        all_s = [m.step for m in loss_history] + [m.step for m in compare_loss_history]
        all_v = [m.value for m in loss_history] + [m.value for m in compare_loss_history]

        mn_s, mx_s = min(all_s), max(all_s)
        mn_v, mx_v = min(all_v), max(all_v)
        vr = mx_v - mn_v if mx_v != mn_v else 1
        mn_v -= vr * 0.05
        mx_v += vr * 0.05
        vr = mx_v - mn_v
        sr = mx_s - mn_s if mx_s != mn_s else 1

        cw, ch_h = 900, 320
        cpl, cpr, cpt, cpb = 70, 40, 30, 50

        def csx(s):
            return cpl + (s - mn_s) / sr * (cw - cpl - cpr)
        def csy(v):
            return cpt + (1 - (v - mn_v) / vr) * (ch_h - cpt - cpb)

        cgrid = ""
        for i in range(7):
            yv = mn_v + vr * i / 6
            yp = csy(yv)
            cgrid += f'<line x1="{cpl}" y1="{yp}" x2="{cw-cpr}" y2="{yp}" stroke="#f0f0f0" stroke-width="1"/>'
            cgrid += f'<text x="{cpl-10}" y="{yp+4}" text-anchor="end" fill="#aaa" font-size="11" font-family="sans-serif">{yv:.4g}</text>'

        for i in range(9):
            xv = mn_s + sr * i / 8
            xp = csx(xv)
            cgrid += f'<text x="{xp}" y="{ch_h-cpb+18}" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">{int(xv)}</text>'

        pts_a = " ".join(f"{csx(m.step)},{csy(m.value)}" for m in loss_history)
        pts_b = " ".join(f"{csx(m.step)},{csy(m.value)}" for m in compare_loss_history)

        overlay_html = f"""
        <div class="rpt">
            <div class="card" style="padding: 28px;">
                <div style="font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 16px;">Loss Curves Overlay</div>
                <svg width="{cw}" height="{ch_h}" style="display: block; margin: 0 auto;">
                    {cgrid}
                    <polyline points="{pts_a}" fill="none" stroke="#667eea" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
                    <polyline points="{pts_b}" fill="none" stroke="#38ef7d" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
                    <text x="{cw//2}" y="{ch_h-5}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif">Training Step</text>
                    <text x="14" y="{ch_h//2}" text-anchor="middle" fill="#aaa" font-size="12" font-family="sans-serif" transform="rotate(-90, 14, {ch_h//2})">Loss</text>
                </svg>
                <div style="display: flex; justify-content: center; gap: 32px; margin-top: 16px; padding-top: 16px; border-top: 1px solid #f0f0f0;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 3px; background: #667eea; border-radius: 2px;"></div>
                        <span style="font-size: 13px; color: #666;">{run_name}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 3px; background: #38ef7d; border-radius: 2px;"></div>
                        <span style="font-size: 13px; color: #666;">{compare_name}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        displayHTML(overlay_html)

else:
    print("No comparison run configured. Set COMPARE_RUN_ID in the config to compare two runs.")
