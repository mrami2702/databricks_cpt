"""
AutoML experiment tool for automl_agent.

Submits the parameterized AutoML notebook and returns a run_id for polling.
Polling and output retrieval reuse get_job_run_status / get_job_run_output
from job_tools — same pattern as the EDA workflow.
"""
from __future__ import annotations

from agents.clients.databricks_sdk_client import get_sdk_client

_VALID_PROBLEM_TYPES = ("regression", "classification", "forecast")


def run_automl_experiment(
    table_name: str,
    target_col: str,
    cluster_id: str,
    problem_type: str = "regression",
    timeout_minutes: int = 30,
    exclude_cols: str = "",
    time_col: str = "",
    experiment_name: str = "",
) -> dict:
    """Submit a Databricks AutoML experiment notebook for a table.

    Runs notebooks/automl_notebook.py on the specified cluster. After submission,
    poll with get_job_run_status(run_id) until life_cycle_state=TERMINATED, then
    call get_job_run_output(run_id) to retrieve the JSON results, and finally
    call generate_automl_report(notebook_output) to produce the HTML report.

    Args:
        table_name: Fully qualified table name (catalog.schema.table).
        target_col: Column to predict (the dependent variable).
        cluster_id: Running Databricks cluster to execute the notebook on.
        problem_type: "regression", "classification", or "forecast".
        timeout_minutes: AutoML timeout in minutes (default 30; AutoML may use
                         more trials if given longer — 60 recommended for production).
        exclude_cols: Comma-separated column names to exclude from features (optional).
                      Useful for excluding IDs, timestamps, or leakage columns.
        time_col: Time column name — required when problem_type="forecast", ignored otherwise.
        experiment_name: Custom MLflow experiment name (optional; auto-generated if blank).

    Returns:
        dict with 'run_id', experiment parameters, and polling instructions.
    """
    from agents.config import AUTOML_NOTEBOOK_PATH

    problem_type = problem_type.strip().lower()
    if problem_type not in _VALID_PROBLEM_TYPES:
        return {
            "error": (
                f"Invalid problem_type '{problem_type}'. "
                f"Must be one of: {', '.join(_VALID_PROBLEM_TYPES)}."
            )
        }
    if problem_type == "forecast" and not time_col:
        return {"error": "time_col is required for forecast experiments."}

    try:
        result = get_sdk_client().run_notebook(
            AUTOML_NOTEBOOK_PATH,
            cluster_id,
            {
                "table_name":      table_name,
                "target_col":      target_col,
                "problem_type":    problem_type,
                "timeout_minutes": str(timeout_minutes),
                "exclude_cols":    exclude_cols,
                "time_col":        time_col,
                "experiment_name": experiment_name,
            },
        )
        return {
            "status":          "submitted",
            "run_id":          result.run_id,
            "table_name":      table_name,
            "target_col":      target_col,
            "problem_type":    problem_type,
            "timeout_minutes": timeout_minutes,
            "message": (
                f"AutoML {problem_type} experiment submitted (run_id={result.run_id}). "
                f"This will run for up to {timeout_minutes} minutes. "
                f"Poll with get_job_run_status({result.run_id}) until "
                "life_cycle_state=TERMINATED, then call "
                f"get_job_run_output({result.run_id}) for the results."
            ),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "table_name": table_name}
