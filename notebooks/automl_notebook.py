# Databricks notebook source
# AutoML Notebook
#
# Accepts a fully qualified table name, target column, and problem type.
# Runs Databricks AutoML and exits with a JSON summary containing the best
# model, primary metric, MLflow experiment ID, and top trial summaries for
# the agent to read back via get_job_run_output().
#
# Designed to be submitted programmatically via the agent's
# run_automl_experiment tool.

# COMMAND ----------
# MAGIC %md
# MAGIC # AutoML Notebook
# MAGIC Runs Databricks AutoML (regression / classification / forecast) and returns
# MAGIC structured results via `dbutils.notebook.exit()` for agent consumption.

# COMMAND ----------
dbutils.widgets.text("table_name",      "",           "Fully qualified table name (catalog.schema.table)")
dbutils.widgets.text("target_col",      "",           "Column to predict")
dbutils.widgets.text("problem_type",    "regression", "Problem type: regression / classification / forecast")
dbutils.widgets.text("timeout_minutes", "30",         "AutoML timeout in minutes")
dbutils.widgets.text("exclude_cols",    "",           "Comma-separated columns to exclude (optional)")
dbutils.widgets.text("time_col",        "",           "Time column — required for forecast only")
dbutils.widgets.text("experiment_name", "",           "Custom MLflow experiment name (optional)")

table_name      = dbutils.widgets.get("table_name")
target_col      = dbutils.widgets.get("target_col")
problem_type    = dbutils.widgets.get("problem_type").strip().lower()
timeout_minutes = int(dbutils.widgets.get("timeout_minutes") or 30)
exclude_cols    = dbutils.widgets.get("exclude_cols")
time_col        = dbutils.widgets.get("time_col").strip()
experiment_name = dbutils.widgets.get("experiment_name").strip() or None

exclude_list = [c.strip() for c in exclude_cols.split(",") if c.strip()] if exclude_cols else []

if not table_name:
    dbutils.notebook.exit('{"error": "table_name parameter is required"}')
if not target_col:
    dbutils.notebook.exit('{"error": "target_col parameter is required"}')
if problem_type not in ("regression", "classification", "forecast"):
    dbutils.notebook.exit('{"error": "problem_type must be regression, classification, or forecast"}')
if problem_type == "forecast" and not time_col:
    dbutils.notebook.exit('{"error": "time_col is required for forecast problem type"}')

# COMMAND ----------
import json
import databricks.automl as automl

ctx  = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = ctx.apiUrl().get()

df = spark.table(table_name)

# COMMAND ----------
# Run AutoML — branch by problem type
try:
    if problem_type == "regression":
        summary = automl.regress(
            dataset=df,
            target_col=target_col,
            timeout_minutes=timeout_minutes,
            exclude_cols=exclude_list or None,
            experiment_name=experiment_name,
        )
    elif problem_type == "classification":
        summary = automl.classify(
            dataset=df,
            target_col=target_col,
            timeout_minutes=timeout_minutes,
            exclude_cols=exclude_list or None,
            experiment_name=experiment_name,
        )
    elif problem_type == "forecast":
        summary = automl.forecast(
            dataset=df,
            target_col=target_col,
            time_col=time_col,
            timeout_minutes=timeout_minutes,
            exclude_cols=exclude_list or None,
            experiment_name=experiment_name,
        )
except Exception as e:
    dbutils.notebook.exit(json.dumps({
        "error":        str(e),
        "table_name":   table_name,
        "target_col":   target_col,
        "problem_type": problem_type,
    }))

# COMMAND ----------
# Extract results from the AutoML summary object
best   = summary.best_trial
exp    = summary.experiment
exp_id = exp.experiment_id

# Top-5 trials (AutoML returns them best-first)
trials_summary = []
for t in (summary.trials or [])[:5]:
    trials_summary.append({
        "run_id":  t.mlflow_run_id,
        "model":   getattr(t, "model_description", None) or "unknown",
        "metrics": {k: round(float(v), 6) for k, v in (t.metrics or {}).items()},
    })

best_metrics = {k: round(float(v), 6) for k, v in (best.metrics or {}).items()}

result = {
    "problem_type":   problem_type,
    "table_name":     table_name,
    "target_col":     target_col,
    "experiment_id":  exp_id,
    "experiment_url": f"{host}/ml/experiments/{exp_id}",
    "best_run_id":    best.mlflow_run_id,
    "best_model":     getattr(best, "model_description", None) or "unknown",
    "best_metrics":   best_metrics,
    "num_trials":     len(summary.trials or []),
    "trials_summary": trials_summary,
    "exclude_cols":   exclude_list,
    "time_col":       time_col or None,
}

print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
