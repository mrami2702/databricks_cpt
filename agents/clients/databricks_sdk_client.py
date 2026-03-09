"""
Databricks SDK client for catalog metadata and cluster/job operations.

Uses databricks-sdk WorkspaceClient — covers Unity Catalog metadata,
cluster lifecycle, and job/run management without requiring Spark.
"""
from __future__ import annotations

from agents.config import DATABRICKS_HOST, DATABRICKS_TOKEN


class DatabricksSDKClient:
    """Wrapper around databricks-sdk WorkspaceClient for catalog + compute operations."""

    def __init__(self) -> None:
        self._ws = None

    def _get_ws(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient

            self._ws = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
        return self._ws

    # -------------------------------------------------------------------------
    # Catalog / Unity Catalog operations
    # -------------------------------------------------------------------------

    def list_catalogs(self) -> list[str]:
        return [c.name for c in self._get_ws().catalogs.list()]

    def list_schemas(self, catalog: str) -> list[str]:
        return [s.name for s in self._get_ws().schemas.list(catalog_name=catalog)]

    def list_tables(self, catalog: str, schema: str) -> list[dict]:
        tables = self._get_ws().tables.list(catalog_name=catalog, schema_name=schema)
        return [
            {"name": t.name, "table_type": str(t.table_type) if t.table_type else "UNKNOWN"}
            for t in tables
        ]

    def get_table(self, catalog: str, schema: str, table: str):
        """Return full TableInfo object (columns, storage, owner, etc.)."""
        return self._get_ws().tables.get(f"{catalog}.{schema}.{table}")

    # -------------------------------------------------------------------------
    # Cluster operations
    # -------------------------------------------------------------------------

    def list_clusters(self) -> list[dict]:
        clusters = list(self._get_ws().clusters.list())
        return [
            {
                "id": c.cluster_id,
                "name": c.cluster_name,
                "state": str(c.state.value) if c.state else "UNKNOWN",
                "runtime": c.spark_version or "",
                "num_workers": c.num_workers or 0,
            }
            for c in clusters
        ]

    def get_cluster(self, cluster_id: str):
        return self._get_ws().clusters.get(cluster_id=cluster_id)

    def start_cluster(self, cluster_id: str) -> str:
        self._get_ws().clusters.start(cluster_id=cluster_id)
        return f"Start command sent for cluster {cluster_id}"

    def stop_cluster(self, cluster_id: str) -> str:
        self._get_ws().clusters.delete(cluster_id=cluster_id)
        return f"Stop command sent for cluster {cluster_id}"

    # -------------------------------------------------------------------------
    # Job operations
    # -------------------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        jobs = list(self._get_ws().jobs.list())
        return [
            {
                "job_id": j.job_id,
                "name": j.settings.name if j.settings else "",
            }
            for j in jobs
        ]

    def get_job(self, job_id: int):
        return self._get_ws().jobs.get(job_id=job_id)

    def run_job_now(self, job_id: int, params: dict | None = None):
        return self._get_ws().jobs.run_now(
            job_id=job_id, notebook_params=params or {}
        )

    def get_run(self, run_id: int):
        return self._get_ws().jobs.get_run(run_id=run_id)

    def cancel_run(self, run_id: int) -> str:
        self._get_ws().jobs.cancel_run(run_id=run_id)
        return f"Cancelled run {run_id}"

    def get_run_output(self, run_id: int):
        return self._get_ws().jobs.get_run_output(run_id=run_id)

    def list_runs(self, limit: int = 20) -> list[dict]:
        runs = list(self._get_ws().jobs.list_runs(limit=limit))
        return [
            {
                "run_id": r.run_id,
                "job_id": r.job_id,
                "state": str(r.state.life_cycle_state.value) if r.state and r.state.life_cycle_state else "UNKNOWN",
                "start_time_ms": r.start_time or 0,
            }
            for r in runs
        ]

    def run_notebook(
        self, notebook_path: str, cluster_id: str, params: dict | None = None
    ):
        """Submit a one-off notebook run via runs/submit."""
        from databricks.sdk.service.jobs import (
            NotebookTask,
            SubmitTask,
        )

        task = SubmitTask(
            task_key="adhoc",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                base_parameters=params or {},
            ),
            existing_cluster_id=cluster_id,
        )
        return self._get_ws().jobs.submit(tasks=[task])

    # -------------------------------------------------------------------------
    # Genie Space operations
    # -------------------------------------------------------------------------

    def create_genie_space(self, title: str, description: str, warehouse_id: str) -> dict:
        space = self._get_ws().genie.create_space(
            title=title, description=description, warehouse_id=warehouse_id
        )
        return {"space_id": space.space_id, "title": space.title}

    def list_genie_spaces(self) -> list[dict]:
        spaces = list(self._get_ws().genie.list_spaces())
        return [
            {
                "space_id": s.space_id,
                "title": s.title,
                "description": s.description or "",
            }
            for s in spaces
        ]

    def get_genie_space(self, space_id: str) -> dict:
        s = self._get_ws().genie.get_space(space_id=space_id)
        return {
            "space_id": s.space_id,
            "title": s.title,
            "description": s.description or "",
        }

    def query_genie_space(self, space_id: str, question: str) -> dict:
        """Stateless NL query. start_conversation_and_wait() handles async polling."""
        response = self._get_ws().genie.start_conversation_and_wait(
            space_id=space_id, content=question
        )
        answer_text, generated_sql = "", ""
        if response.attachments:
            for att in response.attachments:
                if hasattr(att, "query") and att.query:
                    generated_sql = att.query.query or ""
                if hasattr(att, "text") and att.text:
                    answer_text = att.text.content or ""
        return {
            "question": question,
            "answer": answer_text,
            "generated_sql": generated_sql,
            "status": str(response.status) if response.status else "COMPLETED",
        }


_client: DatabricksSDKClient | None = None


def get_sdk_client() -> DatabricksSDKClient:
    """Return the module-level singleton SDK client (lazy init)."""
    global _client
    if _client is None:
        _client = DatabricksSDKClient()
    return _client
