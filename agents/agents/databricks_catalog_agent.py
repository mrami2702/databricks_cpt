from google.adk.agents import LlmAgent

from agents.config import CLAUDE_MODEL, DATABRICKS_CATALOG, DATABRICKS_SCHEMA
from agents.tools.catalog_tools import (
    compare_table_schema,
    describe_table,
    explore_catalogs,
    explore_schemas,
    get_catalog_overview,
    get_dataset_schema,
    get_duplicate_check,
    get_null_summary,
    list_tables,
    preview_data,
    profile_dataset,
    query_data,
    run_sql,
    search_columns,
    search_tables,
)

databricks_catalog_agent = LlmAgent(
    name="DatabricksCatalogAgent",
    model=CLAUDE_MODEL,
    description=(
        "Specialist agent for Unity Catalog data exploration. "
        "Handles all queries about tables, schemas, columns, data previews, "
        "null rates, duplicate checks, and statistical profiling of datasets. "
        "Use this agent for any data access, SQL queries, or table inspection tasks."
    ),
    instruction=f"""You are a data catalog specialist with access to the Databricks Unity Catalog.

Default catalog: {DATABRICKS_CATALOG}, default schema: {DATABRICKS_SCHEMA}

When exploring data:
- Use explore_catalogs → explore_schemas → list_tables to navigate the hierarchy
- Always use describe_table or get_dataset_schema before running heavy queries to understand column types
- Use preview_data to show sample rows before committing to full profiling
- Use profile_dataset for comprehensive statistics (min/max/mean/stddev for numeric, cardinality for categorical)

Column type rules (substring matching, not exact):
- Numeric: double, float, integer, long, decimal, short, byte
- Categorical: string, varchar, char, text, boolean, date, timestamp

Query safety:
- Use query_data for SELECT-only queries
- Use run_sql only when the user explicitly asks to create, insert, update, or delete data
- Always report column count, row count, and null rates when summarizing a table

For search tasks:
- search_tables finds tables by name keyword
- search_columns finds columns by name keyword across all tables
- Both search system.information_schema, so they cover all catalogs the user has access to""",
    tools=[
        explore_catalogs,
        explore_schemas,
        list_tables,
        describe_table,
        search_tables,
        search_columns,
        preview_data,
        query_data,
        run_sql,
        get_dataset_schema,
        get_catalog_overview,
        get_null_summary,
        get_duplicate_check,
        compare_table_schema,
        profile_dataset,
    ],
)
