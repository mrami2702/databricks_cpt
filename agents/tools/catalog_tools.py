"""
15 Unity Catalog tools for databricks_catalog_agent.

All functions return dicts so ADK can serialize results cleanly.
SQL queries use DatabricksSQLClient (sql connector, not Spark).
Catalog metadata uses DatabricksSDKClient (WorkspaceClient).

Column type classification uses substring matching — NOT exact match.
This handles Spark type variants like DecimalType(10,2), VarcharType(255), etc.
Pattern sourced from generate_sft_data.py lines 79-100.
"""
from __future__ import annotations

from agents.clients.databricks_sql import get_sql_client
from agents.clients.databricks_sdk_client import get_sdk_client
from agents.config import DATABRICKS_CATALOG, DATABRICKS_SCHEMA

_NUMERIC_TYPES = ["double", "float", "integer", "long", "decimal", "short", "byte"]
_CATEGORICAL_TYPES = ["string", "varchar", "char", "text", "boolean", "date", "timestamp"]


def _classify_col_type(col_type_str: str) -> str:
    """Return 'numeric' or 'categorical' based on substring matching."""
    t = col_type_str.lower()
    if any(nt in t for nt in _NUMERIC_TYPES):
        return "numeric"
    if any(ct in t for ct in _CATEGORICAL_TYPES):
        return "categorical"
    return "other"


def explore_catalogs() -> dict:
    """List all available Unity Catalog catalogs in the Databricks workspace.

    Returns:
        dict with key 'catalogs' containing list of catalog name strings.
    """
    try:
        catalogs = get_sdk_client().list_catalogs()
        return {"catalogs": catalogs, "count": len(catalogs)}
    except Exception as e:
        return {"error": str(e), "catalogs": []}


def explore_schemas(catalog: str) -> dict:
    """List all schemas within a given Unity Catalog catalog.

    Args:
        catalog: The catalog name (e.g., 'dev_europa').

    Returns:
        dict with key 'schemas' containing list of schema name strings.
    """
    try:
        schemas = get_sdk_client().list_schemas(catalog)
        return {"catalog": catalog, "schemas": schemas, "count": len(schemas)}
    except Exception as e:
        return {"error": str(e), "catalog": catalog, "schemas": []}


def list_tables(catalog: str, schema: str) -> dict:
    """List all tables in a catalog.schema location.

    Args:
        catalog: Catalog name.
        schema: Schema name.

    Returns:
        dict with 'tables' list, each item has 'name' and 'table_type'.
    """
    try:
        tables = get_sdk_client().list_tables(catalog, schema)
        return {"catalog": catalog, "schema": schema, "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"error": str(e), "tables": []}


def describe_table(catalog: str, schema: str, table: str) -> dict:
    """Get schema definition and row count for a table.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.

    Returns:
        dict with 'columns' (list of {name, data_type, nullable}) and 'row_count'.
    """
    try:
        table_info = get_sdk_client().get_table(catalog, schema, table)
        columns = []
        if table_info.columns:
            for col in table_info.columns:
                columns.append({
                    "name": col.name,
                    "data_type": str(col.type_text or col.type_name or "unknown"),
                    "nullable": col.nullable if col.nullable is not None else True,
                    "comment": col.comment or "",
                })
        # Row count via SQL
        try:
            rows = get_sql_client().execute_query(
                f"SELECT COUNT(*) AS row_count FROM `{catalog}`.`{schema}`.`{table}`"
            )
            row_count = rows[0]["row_count"] if rows else 0
        except Exception:
            row_count = "unknown"

        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "columns": columns,
            "column_count": len(columns),
            "row_count": row_count,
        }
    except Exception as e:
        return {"error": str(e), "table": f"{catalog}.{schema}.{table}"}


def search_tables(keyword: str) -> dict:
    """Search for tables whose name contains the keyword.

    Args:
        keyword: Search term to look for in table names.

    Returns:
        dict with 'matches' list of {catalog, schema, table}.
    """
    try:
        sql = f"""
            SELECT table_catalog, table_schema, table_name, comment
            FROM system.information_schema.tables
            WHERE lower(table_name) LIKE lower('%{keyword}%')
            ORDER BY table_catalog, table_schema, table_name
            LIMIT 50
        """
        rows = get_sql_client().execute_query(sql)
        matches = [
            {
                "catalog": r.get("table_catalog", ""),
                "schema": r.get("table_schema", ""),
                "table": r.get("table_name", ""),
                "comment": r.get("comment", "") or "",
            }
            for r in rows
        ]
        return {"keyword": keyword, "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": str(e), "keyword": keyword, "matches": []}


def search_columns(keyword: str) -> dict:
    """Search for columns whose name contains the keyword across all tables.

    Args:
        keyword: Search term for column name matching.

    Returns:
        dict with 'matches' list of {table, column, data_type}.
    """
    try:
        sql = f"""
            SELECT table_catalog, table_schema, table_name, column_name, data_type
            FROM system.information_schema.columns
            WHERE lower(column_name) LIKE lower('%{keyword}%')
            ORDER BY table_catalog, table_schema, table_name, column_name
            LIMIT 100
        """
        rows = get_sql_client().execute_query(sql)
        matches = [
            {
                "table": f"{r.get('table_catalog')}.{r.get('table_schema')}.{r.get('table_name')}",
                "column": r.get("column_name", ""),
                "data_type": r.get("data_type", ""),
            }
            for r in rows
        ]
        return {"keyword": keyword, "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"error": str(e), "keyword": keyword, "matches": []}


def preview_data(catalog: str, schema: str, table: str, limit: int = 10) -> dict:
    """Fetch sample rows from a table.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.
        limit: Number of rows to return (default 10, max 100).

    Returns:
        dict with 'columns' list and 'rows' list of row dicts.
    """
    limit = min(limit, 100)
    try:
        rows = get_sql_client().execute_query(
            f"SELECT * FROM `{catalog}`.`{schema}`.`{table}` LIMIT {limit}"
        )
        columns = list(rows[0].keys()) if rows else []
        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "table": f"{catalog}.{schema}.{table}"}


def query_data(sql: str) -> dict:
    """Execute a SELECT SQL query and return results.

    Only SELECT statements are recommended. For DDL/DML use run_sql.

    Args:
        sql: A valid SQL SELECT statement.

    Returns:
        dict with 'columns', 'rows', and 'row_count'.
    """
    try:
        rows = get_sql_client().execute_query(sql)
        columns = list(rows[0].keys()) if rows else []
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e), "rows": [], "row_count": 0}


def run_sql(sql: str) -> dict:
    """Execute any SQL statement including DDL and DML (CREATE, INSERT, UPDATE, DELETE).

    Use with caution — this can modify data.

    Args:
        sql: Any valid SQL statement.

    Returns:
        dict with 'status' and 'result'.
    """
    try:
        rows = get_sql_client().execute_query(sql)
        return {"status": "success", "result": rows, "row_count": len(rows)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_dataset_schema(catalog: str, schema: str, table: str) -> dict:
    """Get detailed column type information with numeric/categorical classification.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.

    Returns:
        dict with 'columns' list of {name, data_type, col_class, nullable}.
    """
    try:
        table_info = get_sdk_client().get_table(catalog, schema, table)
        columns = []
        if table_info.columns:
            for col in table_info.columns:
                dtype = str(col.type_text or col.type_name or "unknown")
                columns.append({
                    "name": col.name,
                    "data_type": dtype,
                    "col_class": _classify_col_type(dtype),
                    "nullable": col.nullable if col.nullable is not None else True,
                    "comment": col.comment or "",
                })
        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "columns": columns,
            "column_count": len(columns),
        }
    except Exception as e:
        return {"error": str(e), "table": f"{catalog}.{schema}.{table}"}


def get_catalog_overview(catalog: str, schema: str) -> dict:
    """Get a summary of all tables in a schema including row counts.

    Args:
        catalog: Catalog name.
        schema: Schema name.

    Returns:
        dict with 'tables' list of {name, row_count, column_count}.
    """
    try:
        tables = get_sdk_client().list_tables(catalog, schema)
        summary = []
        for t in tables:
            name = t["name"]
            try:
                count_rows = get_sql_client().execute_query(
                    f"SELECT COUNT(*) AS n FROM `{catalog}`.`{schema}`.`{name}`"
                )
                row_count = count_rows[0]["n"] if count_rows else 0
            except Exception:
                row_count = "unknown"
            try:
                table_info = get_sdk_client().get_table(catalog, schema, name)
                col_count = len(table_info.columns) if table_info.columns else 0
            except Exception:
                col_count = "unknown"
            summary.append({"name": name, "row_count": row_count, "column_count": col_count})
        return {
            "catalog": catalog,
            "schema": schema,
            "tables": summary,
            "table_count": len(summary),
        }
    except Exception as e:
        return {"error": str(e)}


def get_null_summary(catalog: str, schema: str, table: str) -> dict:
    """Calculate the percentage of null values for each column in a table.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.

    Returns:
        dict with 'null_rates' mapping column_name -> null_percentage (0-100).
    """
    try:
        table_info = get_sdk_client().get_table(catalog, schema, table)
        if not table_info.columns:
            return {"error": "No columns found", "null_rates": {}}

        col_names = [c.name for c in table_info.columns]
        null_exprs = ", ".join(
            f"ROUND(100.0 * SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS `{c}`"
            for c in col_names
        )
        sql = f"SELECT {null_exprs} FROM `{catalog}`.`{schema}`.`{table}`"
        rows = get_sql_client().execute_query(sql)
        null_rates = rows[0] if rows else {}
        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "null_rates": null_rates,
        }
    except Exception as e:
        return {"error": str(e), "null_rates": {}}


def get_duplicate_check(catalog: str, schema: str, table: str, columns: list[str]) -> dict:
    """Count duplicate rows based on a set of key columns.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.
        columns: List of column names to check for duplicates.

    Returns:
        dict with 'duplicate_count' and 'total_rows'.
    """
    try:
        col_list = ", ".join(f"`{c}`" for c in columns)
        sql_total = f"SELECT COUNT(*) AS total FROM `{catalog}`.`{schema}`.`{table}`"
        sql_dups = f"""
            SELECT COUNT(*) AS dup_count FROM (
                SELECT {col_list}, COUNT(*) AS cnt
                FROM `{catalog}`.`{schema}`.`{table}`
                GROUP BY {col_list}
                HAVING COUNT(*) > 1
            )
        """
        total_rows = get_sql_client().execute_query(sql_total)
        dup_rows = get_sql_client().execute_query(sql_dups)
        return {
            "table": f"{catalog}.{schema}.{table}",
            "key_columns": columns,
            "total_rows": total_rows[0]["total"] if total_rows else 0,
            "duplicate_key_groups": dup_rows[0]["dup_count"] if dup_rows else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def compare_table_schema(table1: str, table2: str) -> dict:
    """Compare the schemas of two fully-qualified tables (catalog.schema.table).

    Args:
        table1: First table as 'catalog.schema.table'.
        table2: Second table as 'catalog.schema.table'.

    Returns:
        dict with 'only_in_table1', 'only_in_table2', 'common_columns', 'type_differences'.
    """
    def _parse(full_name: str):
        parts = full_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Expected 'catalog.schema.table', got: {full_name}")
        return parts

    try:
        c1, s1, t1 = _parse(table1)
        c2, s2, t2 = _parse(table2)

        info1 = get_sdk_client().get_table(c1, s1, t1)
        info2 = get_sdk_client().get_table(c2, s2, t2)

        cols1 = {c.name: str(c.type_text or c.type_name or "unknown") for c in (info1.columns or [])}
        cols2 = {c.name: str(c.type_text or c.type_name or "unknown") for c in (info2.columns or [])}

        only_in_1 = sorted(set(cols1) - set(cols2))
        only_in_2 = sorted(set(cols2) - set(cols1))
        common = sorted(set(cols1) & set(cols2))
        type_diffs = [
            {"column": c, "type_in_table1": cols1[c], "type_in_table2": cols2[c]}
            for c in common
            if cols1[c] != cols2[c]
        ]
        return {
            "table1": table1,
            "table2": table2,
            "only_in_table1": only_in_1,
            "only_in_table2": only_in_2,
            "common_columns": common,
            "type_differences": type_diffs,
        }
    except Exception as e:
        return {"error": str(e)}


def profile_dataset(catalog: str, schema: str, table: str) -> dict:
    """Compute full column statistics: min, max, mean, stddev, null count, and cardinality.

    Numeric columns: min, max, mean, stddev.
    Categorical columns: distinct value count (cardinality).
    Uses substring type detection from generate_sft_data.py (not exact match).

    Args:
        catalog: Catalog name.
        schema: Schema name.
        table: Table name.

    Returns:
        dict with 'profile' list of per-column stats.
    """
    try:
        table_info = get_sdk_client().get_table(catalog, schema, table)
        if not table_info.columns:
            return {"error": "No columns found", "profile": []}

        profile = []
        for col in table_info.columns:
            col_name = col.name
            dtype = str(col.type_text or col.type_name or "unknown")
            col_class = _classify_col_type(dtype)

            try:
                null_result = get_sql_client().execute_query(
                    f"SELECT COUNT(*) - COUNT(`{col_name}`) AS null_count, COUNT(*) AS total "
                    f"FROM `{catalog}`.`{schema}`.`{table}`"
                )
                null_count = null_result[0]["null_count"] if null_result else 0
                total = null_result[0]["total"] if null_result else 0
            except Exception:
                null_count = "unknown"
                total = "unknown"

            stat: dict = {
                "column": col_name,
                "data_type": dtype,
                "col_class": col_class,
                "null_count": null_count,
                "total_rows": total,
            }

            if col_class == "numeric":
                try:
                    agg = get_sql_client().execute_query(
                        f"SELECT MIN(`{col_name}`) AS min_val, MAX(`{col_name}`) AS max_val, "
                        f"AVG(`{col_name}`) AS mean_val, STDDEV(`{col_name}`) AS stddev_val "
                        f"FROM `{catalog}`.`{schema}`.`{table}`"
                    )
                    if agg:
                        stat.update({
                            "min": agg[0]["min_val"],
                            "max": agg[0]["max_val"],
                            "mean": round(float(agg[0]["mean_val"]), 4) if agg[0]["mean_val"] is not None else None,
                            "stddev": round(float(agg[0]["stddev_val"]), 4) if agg[0]["stddev_val"] is not None else None,
                        })
                except Exception:
                    pass
            elif col_class in ("categorical", "other"):
                try:
                    card = get_sql_client().execute_query(
                        f"SELECT COUNT(DISTINCT `{col_name}`) AS cardinality "
                        f"FROM `{catalog}`.`{schema}`.`{table}`"
                    )
                    stat["cardinality"] = card[0]["cardinality"] if card else "unknown"
                except Exception:
                    pass

            profile.append(stat)

        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "profile": profile,
        }
    except Exception as e:
        return {"error": str(e), "profile": []}
