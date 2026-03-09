# Databricks notebook source
# Genie Space Creation Notebook
#
# Creates a Databricks AI/BI Genie Space for a specified table using the
# notebook's own execution context for authentication — no external credentials
# needed. Exits with a JSON result containing space_id and space_url for the
# agent to read back via get_job_run_output().
#
# Designed to run after eda_notebook.py, either as a second task in a
# two-task Databricks job (Option B) or as a standalone agent submission (Option A).
#
# Option B upgrade: set this as Task 2 (depends on Task 1: eda_task) in a
# Databricks job. Use run_eda_genie_pipeline() tool once both tasks are configured.

# COMMAND ----------
# MAGIC %md
# MAGIC # Genie Space Creation Notebook
# MAGIC Creates a Databricks AI/BI Genie Space linked to the target table.
# MAGIC Authenticates using the notebook's own execution context.
# MAGIC Exits with JSON: {space_id, space_title, table_name, space_url}

# COMMAND ----------
dbutils.widgets.text("table_name",   "", "Fully qualified table name (catalog.schema.table)")
dbutils.widgets.text("space_title",  "", "Display name for the Genie Space")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID for Genie to run queries on")
dbutils.widgets.text("description",  "", "Optional description of what this space covers")

table_name   = dbutils.widgets.get("table_name")
space_title  = dbutils.widgets.get("space_title") or f"Genie — {table_name.split('.')[-1]}"
warehouse_id = dbutils.widgets.get("warehouse_id")
description  = dbutils.widgets.get("description") or f"AI/BI Genie Space for {table_name}"

if not table_name:
    dbutils.notebook.exit('{"error": "table_name parameter is required"}')
if not warehouse_id:
    dbutils.notebook.exit('{"error": "warehouse_id parameter is required"}')

# COMMAND ----------
import json
import requests

# Use notebook execution context for auth — no external token needed
ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host  = ctx.apiUrl().get()
token = ctx.apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type":  "application/json",
}

# COMMAND ----------
# Create the Genie Space
resp = requests.post(
    f"{host}/api/2.0/genie/spaces",
    headers=headers,
    json={
        "title":        space_title,
        "description":  description,
        "warehouse_id": warehouse_id,
    },
    timeout=30,
)

if not resp.ok:
    dbutils.notebook.exit(json.dumps({
        "error":       f"Genie API returned {resp.status_code}",
        "detail":      resp.text,
        "table_name":  table_name,
    }))

space_data = resp.json()
space_id   = space_data.get("space_id", "")

# COMMAND ----------
# Note: adding tables to the Genie Space is done via the Databricks UI.
# Open the space_url below, click "Add data", and select the target table.
# Once the table is added, use the agent's query_genie tool with the space_id.

result = {
    "space_id":    space_id,
    "space_title": space_title,
    "table_name":  table_name,
    "space_url":   f"{host}/genie/spaces/{space_id}",
    "next_step":   (
        f"Open {host}/genie/spaces/{space_id}, click 'Add data', "
        f"and add table: {table_name}"
    ),
}

print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
