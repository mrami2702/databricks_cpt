"""
12 cluster/job/notebook management tools for databricks_job_agent.

All functions return dicts. Delegates to DatabricksSDKClient.
"""
from __future__ import annotations

from agents.clients.databricks_sdk_client import get_sdk_client


def list_clusters() -> dict:
    """List all Databricks clusters and their current state.

    Returns:
        dict with 'clusters' list of {id, name, state, runtime, num_workers}.
    """
    try:
        clusters = get_sdk_client().list_clusters()
        return {"clusters": clusters, "count": len(clusters)}
    except Exception as e:
        return {"error": str(e), "clusters": []}


def get_cluster_status(cluster_id: str) -> dict:
    """Get detailed status for a specific cluster.

    Args:
        cluster_id: Databricks cluster ID (e.g., '0218-153045-abc123').

    Returns:
        dict with cluster state, node types, runtime version, and start time.
    """
    try:
        c = get_sdk_client().get_cluster(cluster_id)
        return {
            "cluster_id": c.cluster_id,
            "name": c.cluster_name,
            "state": str(c.state.value) if c.state else "UNKNOWN",
            "state_message": c.state_message or "",
            "runtime": c.spark_version or "",
            "driver_node_type": c.driver_node_type_id or "",
            "worker_node_type": c.node_type_id or "",
            "num_workers": c.num_workers or 0,
            "autoscale": (
                {"min": c.autoscale.min_workers, "max": c.autoscale.max_workers}
                if c.autoscale
                else None
            ),
            "start_time_ms": c.start_time or 0,
        }
    except Exception as e:
        return {"error": str(e), "cluster_id": cluster_id}


def start_cluster(cluster_id: str) -> dict:
    """Start a terminated Databricks cluster.

    Args:
        cluster_id: The cluster to start.

    Returns:
        dict with 'status' and 'message'.
    """
    try:
        msg = get_sdk_client().start_cluster(cluster_id)
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "error": str(e), "cluster_id": cluster_id}


def stop_cluster(cluster_id: str) -> dict:
    """Stop (terminate) a running Databricks cluster.

    Args:
        cluster_id: The cluster to stop.

    Returns:
        dict with 'status' and 'message'.
    """
    try:
        msg = get_sdk_client().stop_cluster(cluster_id)
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "error": str(e), "cluster_id": cluster_id}


def list_jobs() -> dict:
    """List all jobs in the Databricks workspace.

    Returns:
        dict with 'jobs' list of {job_id, name}.
    """
    try:
        jobs = get_sdk_client().list_jobs()
        return {"jobs": jobs, "count": len(jobs)}
    except Exception as e:
        return {"error": str(e), "jobs": []}


def get_job(job_id: int) -> dict:
    """Get configuration and settings for a specific job.

    Args:
        job_id: Databricks job ID.

    Returns:
        dict with job name, schedule, tasks, and cluster config.
    """
    try:
        j = get_sdk_client().get_job(job_id)
        settings = j.settings
        tasks = []
        if settings and settings.tasks:
            for t in settings.tasks:
                tasks.append({
                    "task_key": t.task_key,
                    "type": (
                        "notebook" if t.notebook_task
                        else "python" if t.python_wheel_task
                        else "spark_jar" if t.spark_jar_task
                        else "other"
                    ),
                })
        return {
            "job_id": j.job_id,
            "name": settings.name if settings else "",
            "tasks": tasks,
            "schedule": str(settings.schedule) if settings and settings.schedule else None,
        }
    except Exception as e:
        return {"error": str(e), "job_id": job_id}


def run_job_now(job_id: int, params: dict = None) -> dict:
    """Trigger an immediate run of a job.

    Args:
        job_id: The job to run.
        params: Optional notebook parameter overrides as key-value dict.

    Returns:
        dict with 'run_id' of the triggered run.
    """
    try:
        result = get_sdk_client().run_job_now(job_id, params)
        return {
            "status": "triggered",
            "job_id": job_id,
            "run_id": result.run_id,
            "message": f"Job {job_id} triggered. Use get_job_run_status({result.run_id}) to monitor.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "job_id": job_id}


def get_job_run_status(run_id: int) -> dict:
    """Get the current status of a job run.

    Args:
        run_id: The run ID returned from run_job_now.

    Returns:
        dict with 'state', 'life_cycle_state', 'result_state', and 'run_duration_seconds'.
    """
    try:
        run = get_sdk_client().get_run(run_id)
        state = run.state
        return {
            "run_id": run_id,
            "job_id": run.job_id,
            "life_cycle_state": str(state.life_cycle_state.value) if state and state.life_cycle_state else "UNKNOWN",
            "result_state": str(state.result_state.value) if state and state.result_state else None,
            "state_message": state.state_message if state else "",
            "run_duration_seconds": round(run.run_duration / 1000) if run.run_duration else None,
            "start_time_ms": run.start_time or 0,
        }
    except Exception as e:
        return {"error": str(e), "run_id": run_id}


def cancel_job_run(run_id: int) -> dict:
    """Cancel an active job run.

    Args:
        run_id: The run ID to cancel.

    Returns:
        dict with 'status' and 'message'.
    """
    try:
        msg = get_sdk_client().cancel_run(run_id)
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "error": str(e), "run_id": run_id}


def get_job_run_output(run_id: int) -> dict:
    """Retrieve the output and error log of a completed notebook run.

    Args:
        run_id: The completed run ID.

    Returns:
        dict with 'notebook_output', 'error', and 'logs'.
    """
    try:
        output = get_sdk_client().get_run_output(run_id)
        notebook_output = None
        error = None
        logs = None
        if output.notebook_output:
            notebook_output = output.notebook_output.result
            logs = output.notebook_output.truncated
        if output.error:
            error = output.error
        return {
            "run_id": run_id,
            "notebook_output": notebook_output,
            "error": error,
            "output_truncated": logs,
        }
    except Exception as e:
        return {"error": str(e), "run_id": run_id}


def run_notebook(notebook_path: str, cluster_id: str, params: dict = None) -> dict:
    """Submit a one-off notebook run without a pre-existing job definition.

    Args:
        notebook_path: Absolute path to the notebook (e.g., '/Users/me/my_notebook').
        cluster_id: Cluster to run on.
        params: Optional base parameters for the notebook.

    Returns:
        dict with 'run_id' of the submitted run.
    """
    try:
        result = get_sdk_client().run_notebook(notebook_path, cluster_id, params)
        return {
            "status": "submitted",
            "run_id": result.run_id,
            "notebook_path": notebook_path,
            "cluster_id": cluster_id,
            "message": f"Notebook submitted. Use get_job_run_status({result.run_id}) to monitor.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "notebook_path": notebook_path}


def list_notebook_runs(limit: int = 20) -> dict:
    """List recent notebook/job runs in the workspace.

    Args:
        limit: Number of runs to return (default 20).

    Returns:
        dict with 'runs' list of {run_id, job_id, state, start_time_ms}.
    """
    try:
        runs = get_sdk_client().list_runs(limit=limit)
        return {"runs": runs, "count": len(runs)}
    except Exception as e:
        return {"error": str(e), "runs": []}
