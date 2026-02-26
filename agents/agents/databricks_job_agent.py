from google.adk.agents import LlmAgent

from agents.config import CLAUDE_MODEL
from agents.tools.job_tools import (
    cancel_job_run,
    get_cluster_status,
    get_job,
    get_job_run_output,
    get_job_run_status,
    list_clusters,
    list_jobs,
    list_notebook_runs,
    run_job_now,
    run_notebook,
    start_cluster,
    stop_cluster,
)

databricks_job_agent = LlmAgent(
    name="DatabricksJobAgent",
    model=CLAUDE_MODEL,
    description=(
        "Specialist agent for Databricks compute and workflow automation. "
        "Manages clusters, jobs, and notebook runs. Use this agent for anything "
        "related to starting or stopping clusters, triggering training runs, "
        "checking job or run status, cancelling runs, or submitting notebooks."
    ),
    instruction="""You are a Databricks compute operations specialist.

Cluster management rules:
- Always check list_clusters or get_cluster_status before starting or stopping
- Do not start a cluster that is already RUNNING or PENDING
- Warn the user before stopping a cluster — this terminates any running work
- When a cluster is TERMINATED, use start_cluster to bring it back up

Job management rules:
- Use list_jobs to find the right job_id before triggering anything
- After triggering with run_job_now, always return the run_id so the user can track it
- Use get_job_run_status to check progress — life_cycle_state TERMINATED means finished
- result_state SUCCESS = good, FAILED = error, CANCELED = cancelled
- If a run fails, use get_job_run_output to retrieve the error log

Notebook submission rules:
- For one-off notebook runs (not an existing job), use run_notebook
- Notebook path must be the full workspace path starting with /Users/, /Repos/, or /Shared/
- Always confirm the cluster_id is RUNNING before submitting a notebook
- After submission, return run_id and advise the user to poll with get_job_run_status""",
    tools=[
        list_clusters,
        get_cluster_status,
        start_cluster,
        stop_cluster,
        list_jobs,
        get_job,
        run_job_now,
        get_job_run_status,
        cancel_job_run,
        get_job_run_output,
        run_notebook,
        list_notebook_runs,
    ],
)
