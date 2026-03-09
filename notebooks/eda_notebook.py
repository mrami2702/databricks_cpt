# Databricks notebook source
# EDA Notebook — Light Profile
#
# Accepts a fully qualified table name, produces a lightweight profile, and
# exits with a JSON summary that the DatabricksJobAgent reads back via
# get_job_run_output(). Designed to be submitted programmatically via the
# agent's run_eda_notebook tool.

# COMMAND ----------
# MAGIC %md
# MAGIC # EDA Notebook — Light Profile
# MAGIC Produces: row count, column count, schema with types, null rate per column,
# MAGIC and a configurable sample of rows. Output is returned as JSON via
# MAGIC `dbutils.notebook.exit()` for agent consumption.

# COMMAND ----------
dbutils.widgets.text("table_name", "", "Fully qualified table name (catalog.schema.table)")
dbutils.widgets.text("sample_rows", "10", "Number of sample rows to include in output")

table_name  = dbutils.widgets.get("table_name")
sample_rows = int(dbutils.widgets.get("sample_rows"))

if not table_name:
    dbutils.notebook.exit('{"error": "table_name parameter is required"}')

# COMMAND ----------
import json
from pyspark.sql import functions as F

df        = spark.table(table_name)
row_count = df.count()
fields    = df.schema.fields

# COMMAND ----------
# Null rate per column
null_rates = {}
for field in fields:
    null_count = df.filter(F.col(field.name).isNull()).count()
    null_rates[field.name] = round(null_count / row_count, 4) if row_count > 0 else 0.0

# COMMAND ----------
# Sample rows — convert non-JSON-serializable types (Decimal, date, etc.) to str
sample = df.limit(sample_rows).toPandas().to_dict(orient="records")
for row in sample:
    for k, v in row.items():
        if not isinstance(v, (int, float, bool, str, type(None))):
            row[k] = str(v)

# COMMAND ----------
summary = {
    "table_name":   table_name,
    "row_count":    row_count,
    "column_count": len(fields),
    "schema": [
        {"name": f.name, "type": str(f.dataType), "nullable": f.nullable}
        for f in fields
    ],
    "null_rates":   null_rates,
    "sample_rows":  sample,
}

dbutils.notebook.exit(json.dumps(summary))
