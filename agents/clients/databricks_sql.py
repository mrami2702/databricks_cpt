"""
SQL connector client for executing queries against Databricks Unity Catalog
from outside a Databricks notebook environment.

Uses the Databricks SQL Connector (not Spark) — requires a SQL Warehouse
HTTP path (DATABRICKS_HTTP_PATH env var), not a cluster.
"""
from __future__ import annotations

from agents.config import DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH


class DatabricksSQLClient:
    """Thin wrapper around databricks-sql-connector for SELECT and DDL/DML queries."""

    def __init__(self) -> None:
        self._conn = None

    def _get_connection(self):
        if self._conn is None:
            import databricks.sql as dbsql

            hostname = DATABRICKS_HOST.replace("https://", "").replace("http://", "")
            self._conn = dbsql.connect(
                server_hostname=hostname,
                http_path=DATABRICKS_HTTP_PATH,
                access_token=DATABRICKS_TOKEN,
            )
        return self._conn

    def execute_query(self, sql: str) -> list[dict]:
        """Execute a SQL statement and return results as a list of row dicts.
        Works for SELECT, DDL, and DML statements.
        """
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql)
            if cursor.description is None:
                return []
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(cols, row)) for row in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


_client: DatabricksSQLClient | None = None


def get_sql_client() -> DatabricksSQLClient:
    """Return the module-level singleton SQL client (lazy init)."""
    global _client
    if _client is None:
        _client = DatabricksSQLClient()
    return _client
