# Databricks notebook source
# MAGIC %md
# MAGIC # Generate SFT Training Data from MLflow Registry
# MAGIC
# MAGIC This notebook queries your MLflow Model Registry and generates
# MAGIC question-answer pairs about model selection, comparison, and deployment.
# MAGIC
# MAGIC **No trained model needed** — this is pure data processing.
# MAGIC Run this alongside or after `generate_sft_data.py`.
# MAGIC
# MAGIC The output is a table of (instruction, response, category) rows
# MAGIC that can be combined with the scientific domain Q&A for SFT training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Where to save the generated Q&A pairs
DEST_CATALOG = "dev_europa"
DEST_SCHEMA = "gold_roses"
DEST_TABLE = "sft_mlflow_data"

# Max models to pull from registry
MAX_MODELS = 50

# Max versions per model to analyze
MAX_VERSIONS_PER_MODEL = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Query MLflow Registry

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Pull all registered models
registered_models = client.search_registered_models()
print(f"Found {len(registered_models)} registered models in MLflow")

# Gather full model info
models = []

for rm in registered_models[:MAX_MODELS]:
    model_name = rm.name
    description = rm.description or ""
    tags = dict(rm.tags) if rm.tags else {}

    versions = client.get_latest_versions(model_name)

    for v in versions[:MAX_VERSIONS_PER_MODEL]:
        version_info = {
            "name": model_name,
            "version": v.version,
            "stage": v.current_stage,
            "description": v.description or description,
            "tags": tags,
            "metrics": {},
            "params": {},
            "run_id": v.run_id,
        }

        # Pull run data if available
        if v.run_id:
            try:
                run = client.get_run(v.run_id)
                version_info["metrics"] = dict(run.data.metrics)
                version_info["params"] = dict(run.data.params)
                version_info["tags"].update({
                    tk: tv for tk, tv in run.data.tags.items()
                    if not tk.startswith("mlflow.")
                })
            except Exception as e:
                print(f"  Could not fetch run for {model_name} v{v.version}: {e}")

        models.append(version_info)
        print(f"  {model_name} v{v.version} ({v.current_stage}) — "
              f"{len(version_info['metrics'])} metrics, {len(version_info['params'])} params")

print(f"\nTotal model versions collected: {len(models)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Analyze Models
# MAGIC
# MAGIC Compute cross-model comparisons so answers are accurate.

# COMMAND ----------

# Group by model name
from collections import defaultdict

models_by_name = defaultdict(list)
for m in models:
    models_by_name[m["name"]].append(m)

# Find all unique metric names across all models
all_metric_names = set()
for m in models:
    all_metric_names.update(m["metrics"].keys())

# Find all unique param names
all_param_names = set()
for m in models:
    all_param_names.update(m["params"].keys())

# Identify metrics where lower is better vs higher is better
lower_is_better = {"rmse", "mse", "mae", "loss", "error", "log_loss", "logloss",
                   "inference_latency_ms", "latency", "latency_ms", "training_time"}
higher_is_better = {"accuracy", "r2", "r2_score", "f1", "precision", "recall",
                    "auc", "roc_auc", "ap", "map", "ndcg"}

# Find best model per metric
best_per_metric = {}
for metric_name in all_metric_names:
    models_with_metric = [
        (m, m["metrics"][metric_name])
        for m in models
        if metric_name in m["metrics"] and m["metrics"][metric_name] is not None
    ]
    if not models_with_metric:
        continue

    metric_lower = metric_name.lower()
    if any(k in metric_lower for k in lower_is_better):
        best = min(models_with_metric, key=lambda x: x[1])
        direction = "lowest"
    elif any(k in metric_lower for k in higher_is_better):
        best = max(models_with_metric, key=lambda x: x[1])
        direction = "highest"
    else:
        best = max(models_with_metric, key=lambda x: x[1])
        direction = "highest"

    best_per_metric[metric_name] = {
        "model": best[0],
        "value": best[1],
        "direction": direction,
        "all_values": [(m["name"], m["version"], v) for m, v in models_with_metric],
    }

print(f"Unique metrics across all models: {len(all_metric_names)}")
print(f"Unique params across all models: {len(all_param_names)}")
print(f"Metrics analyzed: {list(all_metric_names)}")

# Models by stage
stages = defaultdict(list)
for m in models:
    stages[m["stage"]].append(m)

print(f"\nModels by stage:")
for stage, stage_models in stages.items():
    print(f"  {stage}: {len(stage_models)} model versions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Helper Functions

# COMMAND ----------

def fmt_metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) > 10000:
            return f"{value:.6g}"
        return f"{value:.4f}"
    return str(value)

def model_label(m):
    return f"{m['name']} (v{m['version']})"

def model_summary(m):
    parts = [f"{model_label(m)}, stage: {m['stage']}"]
    if m["description"]:
        parts.append(f"Description: {m['description']}")
    if m["metrics"]:
        metric_strs = [f"{k}: {fmt_metric(v)}" for k, v in sorted(m["metrics"].items()) if v is not None]
        if metric_strs:
            parts.append(f"Metrics: {'; '.join(metric_strs)}")
    if m["params"]:
        param_strs = [f"{k}: {v}" for k, v in sorted(m["params"].items())[:10]]
        parts.append(f"Parameters: {'; '.join(param_strs)}")
    return ". ".join(parts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Individual Model Questions

# COMMAND ----------

qa_pairs = []

for m in models:
    label = model_label(m)

    # --- Describe this model ---
    qa_pairs.append({
        "instruction": f"Describe the {m['name']} model.",
        "response": model_summary(m),
        "category": "model_info",
    })

    # --- What are its metrics? ---
    if m["metrics"]:
        metric_text = "; ".join(
            f"{k}: {fmt_metric(v)}" for k, v in sorted(m["metrics"].items())
            if v is not None
        )
        qa_pairs.append({
            "instruction": f"What are the performance metrics for {label}?",
            "response": (
                f"The performance metrics for {label} are: {metric_text}. "
                f"This model is currently in the {m['stage']} stage."
            ),
            "category": "model_info",
        })

    # --- What are its parameters? ---
    if m["params"]:
        param_text = "; ".join(
            f"{k}: {v}" for k, v in sorted(m["params"].items())[:15]
        )
        qa_pairs.append({
            "instruction": f"What parameters were used to train {label}?",
            "response": (
                f"The training parameters for {label} are: {param_text}."
            ),
            "category": "model_info",
        })

    # --- What stage is it in? ---
    qa_pairs.append({
        "instruction": f"What stage is {m['name']} in?",
        "response": (
            f"{label} is currently in the {m['stage']} stage."
            + (f" Description: {m['description']}." if m["description"] else "")
        ),
        "category": "model_info",
    })

    # --- Ask about a specific metric ---
    for metric_name, metric_value in m["metrics"].items():
        if metric_value is None:
            continue
        qa_pairs.append({
            "instruction": f"What is the {metric_name} for {m['name']}?",
            "response": (
                f"The {metric_name} for {label} is {fmt_metric(metric_value)}."
            ),
            "category": "model_info",
        })

print(f"Generated {len(qa_pairs)} individual model Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Comparison Questions

# COMMAND ----------

count_before = len(qa_pairs)

model_names = list(models_by_name.keys())

# --- Compare all models on a specific metric ---
for metric_name, info in best_per_metric.items():
    if len(info["all_values"]) < 2:
        continue

    ranking = sorted(
        info["all_values"],
        key=lambda x: x[2],
        reverse=(info["direction"] == "highest"),
    )
    ranking_text = "; ".join(
        f"{name} v{ver}: {fmt_metric(val)}" for name, ver, val in ranking
    )

    qa_pairs.append({
        "instruction": f"Compare all models on {metric_name}.",
        "response": (
            f"Ranking models by {metric_name} ({info['direction']} is best): "
            f"{ranking_text}. "
            f"The best performing model is {ranking[0][0]} v{ranking[0][1]} "
            f"with {metric_name} of {fmt_metric(ranking[0][2])}."
        ),
        "category": "model_comparison",
    })

    # --- Which model has the best X? ---
    qa_pairs.append({
        "instruction": f"Which model has the best {metric_name}?",
        "response": (
            f"{ranking[0][0]} (v{ranking[0][1]}) has the best {metric_name} "
            f"at {fmt_metric(ranking[0][2])}. "
            + (
                f"The next closest is {ranking[1][0]} (v{ranking[1][1]}) "
                f"at {fmt_metric(ranking[1][2])}."
                if len(ranking) > 1 else ""
            )
        ),
        "category": "model_comparison",
    })

# --- Pairwise comparison between models ---
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1 = models[i]
        m2 = models[j]

        # Find shared metrics
        shared_metrics = set(m1["metrics"].keys()) & set(m2["metrics"].keys())
        if not shared_metrics:
            continue

        comparison_parts = []
        m1_wins = 0
        m2_wins = 0

        for metric in sorted(shared_metrics):
            v1 = m1["metrics"][metric]
            v2 = m2["metrics"][metric]
            if v1 is None or v2 is None:
                continue

            metric_lower = metric.lower()
            if any(k in metric_lower for k in lower_is_better):
                winner = model_label(m1) if v1 < v2 else model_label(m2)
                if v1 < v2:
                    m1_wins += 1
                else:
                    m2_wins += 1
            else:
                winner = model_label(m1) if v1 > v2 else model_label(m2)
                if v1 > v2:
                    m1_wins += 1
                else:
                    m2_wins += 1

            comparison_parts.append(
                f"{metric}: {model_label(m1)} = {fmt_metric(v1)}, "
                f"{model_label(m2)} = {fmt_metric(v2)}"
            )

        if comparison_parts:
            overall = model_label(m1) if m1_wins > m2_wins else model_label(m2) if m2_wins > m1_wins else "neither clearly"
            qa_pairs.append({
                "instruction": f"Compare {m1['name']} and {m2['name']}.",
                "response": (
                    f"Comparing {model_label(m1)} vs {model_label(m2)}: "
                    + "; ".join(comparison_parts)
                    + f". Overall, {overall} performs better across the majority of metrics."
                ),
                "category": "model_comparison",
            })

print(f"Generated {len(qa_pairs) - count_before} model comparison Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Model Selection & Recommendation Questions

# COMMAND ----------

count_before = len(qa_pairs)

# --- Which model should I deploy? (general) ---
if models:
    # Find model that wins on most metrics
    win_counts = defaultdict(int)
    for metric_name, info in best_per_metric.items():
        best_model = info["model"]
        win_counts[model_label(best_model)] += 1

    if win_counts:
        overall_best_label = max(win_counts, key=win_counts.get)
        overall_best = next(m for m in models if model_label(m) == overall_best_label)
        win_count = win_counts[overall_best_label]
        total_metrics = len(best_per_metric)

        metric_wins = []
        for metric_name, info in best_per_metric.items():
            if model_label(info["model"]) == overall_best_label:
                metric_wins.append(f"{metric_name}: {fmt_metric(info['value'])}")

        qa_pairs.append({
            "instruction": "Which model should I deploy to production?",
            "response": (
                f"Based on the metrics in MLflow, {overall_best_label} performs best "
                f"across {win_count} of {total_metrics} tracked metrics. "
                f"It leads in: {'; '.join(metric_wins)}. "
                f"It is currently in the {overall_best['stage']} stage."
                + (f" Description: {overall_best['description']}." if overall_best["description"] else "")
            ),
            "category": "recommendation",
        })

        qa_pairs.append({
            "instruction": "What is your recommendation for the best model?",
            "response": (
                f"I recommend {overall_best_label} as it outperforms other models "
                f"on {win_count} out of {total_metrics} metrics. "
                f"Key strengths: {'; '.join(metric_wins[:5])}."
            ),
            "category": "recommendation",
        })

# --- Recommendations with constraints ---
# Find fastest model (if latency metric exists)
latency_metrics = [m for m in all_metric_names if "latency" in m.lower() or "time" in m.lower()]
for lat_metric in latency_metrics:
    models_with_latency = [
        (m, m["metrics"][lat_metric])
        for m in models
        if lat_metric in m["metrics"] and m["metrics"][lat_metric] is not None
    ]
    if len(models_with_latency) < 2:
        continue

    fastest = min(models_with_latency, key=lambda x: x[1])
    most_accurate = None

    # Find accuracy-type metrics
    for acc_metric in all_metric_names:
        acc_lower = acc_metric.lower()
        if any(k in acc_lower for k in higher_is_better):
            acc_models = [
                (m, m["metrics"][acc_metric])
                for m in models
                if acc_metric in m["metrics"] and m["metrics"][acc_metric] is not None
            ]
            if acc_models:
                most_accurate = max(acc_models, key=lambda x: x[1])
                break

    qa_pairs.append({
        "instruction": "Which model is fastest for real-time inference?",
        "response": (
            f"The fastest model is {model_label(fastest[0])} with "
            f"{lat_metric} of {fmt_metric(fastest[1])}."
            + (
                f" However, if accuracy is more important, {model_label(most_accurate[0])} "
                f"may be preferred despite higher latency."
                if most_accurate and model_label(most_accurate[0]) != model_label(fastest[0])
                else ""
            )
        ),
        "category": "recommendation",
    })

    qa_pairs.append({
        "instruction": "I need low latency — which model should I use?",
        "response": (
            f"For low latency requirements, {model_label(fastest[0])} is the best choice "
            f"with {lat_metric} of {fmt_metric(fastest[1])}. "
            f"It is currently in the {fastest[0]['stage']} stage."
        ),
        "category": "recommendation",
    })

# --- Accuracy-focused recommendation ---
for acc_metric in all_metric_names:
    acc_lower = acc_metric.lower()
    if any(k in acc_lower for k in higher_is_better):
        acc_models = [
            (m, m["metrics"][acc_metric])
            for m in models
            if acc_metric in m["metrics"] and m["metrics"][acc_metric] is not None
        ]
        if len(acc_models) < 2:
            continue

        best_acc = max(acc_models, key=lambda x: x[1])
        qa_pairs.append({
            "instruction": f"Which model has the best {acc_metric} for maximum accuracy?",
            "response": (
                f"For maximum accuracy, {model_label(best_acc[0])} has the highest "
                f"{acc_metric} at {fmt_metric(best_acc[1])}. "
                f"It is in the {best_acc[0]['stage']} stage."
            ),
            "category": "recommendation",
        })

for err_metric in all_metric_names:
    err_lower = err_metric.lower()
    if any(k in err_lower for k in lower_is_better) and "latency" not in err_lower and "time" not in err_lower:
        err_models = [
            (m, m["metrics"][err_metric])
            for m in models
            if err_metric in m["metrics"] and m["metrics"][err_metric] is not None
        ]
        if len(err_models) < 2:
            continue

        best_err = min(err_models, key=lambda x: x[1])
        qa_pairs.append({
            "instruction": f"Which model has the lowest {err_metric}?",
            "response": (
                f"The model with the lowest {err_metric} is {model_label(best_err[0])} "
                f"at {fmt_metric(best_err[1])}."
            ),
            "category": "recommendation",
        })

print(f"Generated {len(qa_pairs) - count_before} recommendation Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Deployment & Stage Questions

# COMMAND ----------

count_before = len(qa_pairs)

# --- What is in production? ---
production_models = stages.get("Production", [])
if production_models:
    prod_list = "; ".join(model_label(m) for m in production_models)
    qa_pairs.append({
        "instruction": "What models are currently in production?",
        "response": f"The following models are currently in the Production stage: {prod_list}.",
        "category": "deployment",
    })

    for m in production_models:
        if m["metrics"]:
            metric_text = "; ".join(
                f"{k}: {fmt_metric(v)}" for k, v in sorted(m["metrics"].items())
                if v is not None
            )
            qa_pairs.append({
                "instruction": f"How is {m['name']} performing in production?",
                "response": (
                    f"{model_label(m)} is in Production with metrics: {metric_text}."
                ),
                "category": "deployment",
            })

# --- What is in staging? ---
staging_models = stages.get("Staging", [])
if staging_models:
    staging_list = "; ".join(model_label(m) for m in staging_models)
    qa_pairs.append({
        "instruction": "What models are in staging?",
        "response": f"The following models are in the Staging stage: {staging_list}.",
        "category": "deployment",
    })

# --- Should we promote staging to production? ---
if production_models and staging_models:
    for staging_m in staging_models:
        for prod_m in production_models:
            shared_metrics = set(staging_m["metrics"].keys()) & set(prod_m["metrics"].keys())
            if not shared_metrics:
                continue

            improvements = []
            regressions = []
            for metric in sorted(shared_metrics):
                sv = staging_m["metrics"][metric]
                pv = prod_m["metrics"][metric]
                if sv is None or pv is None:
                    continue

                metric_lower = metric.lower()
                if any(k in metric_lower for k in lower_is_better):
                    if sv < pv:
                        pct = 100 * (pv - sv) / abs(pv) if pv != 0 else 0
                        improvements.append(f"{metric} improved by {pct:.1f}% ({fmt_metric(pv)} -> {fmt_metric(sv)})")
                    elif sv > pv:
                        pct = 100 * (sv - pv) / abs(pv) if pv != 0 else 0
                        regressions.append(f"{metric} regressed by {pct:.1f}% ({fmt_metric(pv)} -> {fmt_metric(sv)})")
                else:
                    if sv > pv:
                        pct = 100 * (sv - pv) / abs(pv) if pv != 0 else 0
                        improvements.append(f"{metric} improved by {pct:.1f}% ({fmt_metric(pv)} -> {fmt_metric(sv)})")
                    elif sv < pv:
                        pct = 100 * (pv - sv) / abs(pv) if pv != 0 else 0
                        regressions.append(f"{metric} regressed by {pct:.1f}% ({fmt_metric(pv)} -> {fmt_metric(sv)})")

            if improvements or regressions:
                qa_pairs.append({
                    "instruction": f"Should I promote {staging_m['name']} from staging to production over {prod_m['name']}?",
                    "response": (
                        f"Comparing {model_label(staging_m)} (Staging) vs {model_label(prod_m)} (Production): "
                        + (f"Improvements: {'; '.join(improvements)}. " if improvements else "No improvements. ")
                        + (f"Regressions: {'; '.join(regressions)}. " if regressions else "No regressions. ")
                        + (
                            f"With {len(improvements)} improvements and {len(regressions)} regressions, "
                            + ("promotion is recommended." if len(improvements) > len(regressions) else
                               "promotion should be evaluated carefully." if len(improvements) == len(regressions) else
                               "the current production model may still be preferable.")
                        )
                    ),
                    "category": "deployment",
                })

# --- How many models do we have? ---
qa_pairs.append({
    "instruction": "How many models are registered in MLflow?",
    "response": (
        f"There are {len(registered_models)} registered models in MLflow "
        f"with {len(models)} total versions tracked. "
        + (f"Stages: " + ", ".join(f"{s}: {len(ms)}" for s, ms in stages.items()) + ".")
    ),
    "category": "deployment",
})

# --- List all models ---
qa_pairs.append({
    "instruction": "List all registered models.",
    "response": (
        f"Registered models in MLflow: "
        + "; ".join(
            f"{m['name']} v{m['version']} ({m['stage']})"
            for m in models
        )
        + "."
    ),
    "category": "deployment",
})

print(f"Generated {len(qa_pairs) - count_before} deployment Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Trade-off & Risk Questions

# COMMAND ----------

count_before = len(qa_pairs)

# --- Accuracy vs latency trade-off ---
if latency_metrics and models:
    for lat_metric in latency_metrics:
        for acc_metric in all_metric_names:
            acc_lower = acc_metric.lower()
            if not any(k in acc_lower for k in higher_is_better):
                continue

            both_models = [
                m for m in models
                if lat_metric in m["metrics"] and acc_metric in m["metrics"]
                and m["metrics"][lat_metric] is not None
                and m["metrics"][acc_metric] is not None
            ]
            if len(both_models) < 2:
                continue

            trade_off_parts = []
            for m in sorted(both_models, key=lambda x: -x["metrics"][acc_metric]):
                trade_off_parts.append(
                    f"{model_label(m)}: {acc_metric}={fmt_metric(m['metrics'][acc_metric])}, "
                    f"{lat_metric}={fmt_metric(m['metrics'][lat_metric])}"
                )

            qa_pairs.append({
                "instruction": f"What is the trade-off between {acc_metric} and {lat_metric} across models?",
                "response": (
                    f"There is a trade-off between {acc_metric} and {lat_metric}: "
                    + "; ".join(trade_off_parts)
                    + ". Higher accuracy models tend to have higher latency."
                ),
                "category": "trade_off",
            })
            break  # One per latency metric
        break  # One overall

# --- Version progression ---
for model_name, versions in models_by_name.items():
    if len(versions) < 2:
        continue

    sorted_versions = sorted(versions, key=lambda x: int(x["version"]))
    progression_parts = []
    for ver in sorted_versions:
        metric_text = ", ".join(
            f"{k}: {fmt_metric(val)}"
            for k, val in sorted(ver["metrics"].items())[:5]
            if val is not None
        ) if ver["metrics"] else "no metrics"
        progression_parts.append(f"v{ver['version']} ({ver['stage']}): {metric_text}")

    qa_pairs.append({
        "instruction": f"How has {model_name} improved across versions?",
        "response": (
            f"Version history for {model_name}: "
            + "; ".join(progression_parts) + "."
        ),
        "category": "trade_off",
    })

# --- Risk questions for production models ---
for m in production_models:
    risks = []

    # Check if newer versions exist in staging
    staging_same = [
        s for s in staging_models
        if s["name"] == m["name"] and int(s["version"]) > int(m["version"])
    ]
    if staging_same:
        risks.append(
            f"a newer version (v{staging_same[0]['version']}) exists in Staging "
            f"that may outperform the current production version"
        )

    # Check if it has the worst metrics
    for metric_name, info in best_per_metric.items():
        if model_label(info["model"]) != model_label(m):
            all_vals = [v for _, _, v in info["all_values"]]
            m_val = m["metrics"].get(metric_name)
            if m_val is not None and len(all_vals) > 1:
                if info["direction"] == "lowest" and m_val == max(all_vals):
                    risks.append(f"it has the worst {metric_name} ({fmt_metric(m_val)}) among all models")
                    break
                elif info["direction"] == "highest" and m_val == min(all_vals):
                    risks.append(f"it has the worst {metric_name} ({fmt_metric(m_val)}) among all models")
                    break

    if risks:
        qa_pairs.append({
            "instruction": f"What are the risks of keeping {m['name']} in production?",
            "response": (
                f"Risks for {model_label(m)} in Production: "
                + "; ".join(risks)
                + ". Consider evaluating alternatives and running A/B tests before making changes."
            ),
            "category": "trade_off",
        })

print(f"Generated {len(qa_pairs) - count_before} trade-off Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Registry Overview Questions

# COMMAND ----------

count_before = len(qa_pairs)

# --- What metrics do we track? ---
qa_pairs.append({
    "instruction": "What metrics are tracked across our models?",
    "response": (
        f"The following metrics are tracked in MLflow across {len(models)} model versions: "
        + ", ".join(sorted(all_metric_names))
        + f". Not all models have every metric — coverage varies by model."
    ),
    "category": "registry_overview",
})

# --- What parameters are tracked? ---
if all_param_names:
    qa_pairs.append({
        "instruction": "What training parameters are tracked?",
        "response": (
            f"The following training parameters are logged in MLflow: "
            + ", ".join(sorted(all_param_names))
            + "."
        ),
        "category": "registry_overview",
    })

# --- Summary of the registry ---
qa_pairs.append({
    "instruction": "Give me a summary of our MLflow model registry.",
    "response": (
        f"The MLflow registry contains {len(registered_models)} registered models "
        f"with {len(models)} tracked versions. "
        + ", ".join(f"{s}: {len(ms)} models" for s, ms in stages.items())
        + f". We track {len(all_metric_names)} unique metrics and "
        f"{len(all_param_names)} unique parameters across all models."
    ),
    "category": "registry_overview",
})

# --- Which models share the same metrics (comparable)? ---
metric_to_models = defaultdict(list)
for m in models:
    for metric in m["metrics"]:
        metric_to_models[metric].append(model_label(m))

comparable_metrics = {k: v for k, v in metric_to_models.items() if len(v) > 1}
if comparable_metrics:
    comp_text = "; ".join(
        f"{metric}: {', '.join(models_list)}"
        for metric, models_list in sorted(comparable_metrics.items())[:10]
    )
    qa_pairs.append({
        "instruction": "Which models can be directly compared?",
        "response": (
            f"Models sharing the same metrics can be compared directly. "
            f"Shared metrics: {comp_text}."
        ),
        "category": "registry_overview",
    })

print(f"Generated {len(qa_pairs) - count_before} registry overview Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Save to Unity Catalog

# COMMAND ----------

from pyspark.sql import functions as F

print(f"\nFinal Q&A pair counts:")
categories = {}
for p in qa_pairs:
    cat = p["category"]
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")
print(f"  TOTAL: {len(qa_pairs)}")

sft_df = spark.createDataFrame(qa_pairs)
dest_full = f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}"

sft_df.write.mode("overwrite").saveAsTable(dest_full)

print(f"\nSaved to {dest_full}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Preview

# COMMAND ----------

preview_df = spark.table(f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}")
total = preview_df.count()

print("=" * 70)
print(f"SFT MLFLOW DATA SUMMARY — {total} total Q&A pairs")
print("=" * 70)

print("\nBy category:")
category_counts = preview_df.groupBy("category").count().orderBy("category").collect()
for row in category_counts:
    print(f"  {row['category']}: {row['count']}")

for cat_row in category_counts:
    cat = cat_row["category"]
    print()
    print("=" * 70)
    print(f"SAMPLE — {cat.upper()}")
    print("=" * 70)

    samples = (
        preview_df
        .filter(F.col("category") == cat)
        .orderBy(F.rand(seed=42))
        .limit(2)
        .collect()
    )

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Q: {sample['instruction']}")
        print(f"A: {sample['response']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Review the samples above
# MAGIC 2. To combine with scientific domain Q&A for SFT training:
# MAGIC    ```python
# MAGIC    domain_qa = spark.table("dev_europa.gold_roses.sft_training_data")
# MAGIC    mlflow_qa = spark.table("dev_europa.gold_roses.sft_mlflow_data")
# MAGIC    combined = domain_qa.unionByName(mlflow_qa)
# MAGIC    ```
# MAGIC 3. Use the combined dataset in `train_sft_mistral.py`
