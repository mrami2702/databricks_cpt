# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Training Data for CPT
# MAGIC
# MAGIC This notebook converts tabular scientific data from Unity Catalog into
# MAGIC natural language descriptions that the LLM can learn from.
# MAGIC
# MAGIC **Why?** LLMs learn from text, not raw table rows. This script converts
# MAGIC each table's data into rich natural language that describes what the data
# MAGIC means, preserving relationships between columns and across tables.
# MAGIC
# MAGIC **Run this BEFORE train_cpt.py**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source: where your raw scientific data lives
SOURCE_CATALOG = "dev_europa"
SOURCE_SCHEMA = "gold_roses"

# Destination: where processed training text will be saved
# (Creates a new table in the same catalog)
DEST_CATALOG = "dev_europa"
DEST_SCHEMA = "gold_roses"
DEST_TABLE = "cpt_training_text"

# Maximum rows per table to process (None = all)
MAX_ROWS_PER_TABLE = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Discover All Tables and Their Schemas

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, FloatType, IntegerType, LongType, BooleanType

tables = spark.catalog.listTables(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}")
table_names = [t.name for t in tables if t.name != DEST_TABLE]  # exclude our output table

print(f"Found {len(table_names)} tables in {SOURCE_CATALOG}.{SOURCE_SCHEMA}:")
print()

table_schemas = {}
for name in table_names:
    full_name = f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{name}"
    df = spark.table(full_name)
    schema = df.schema
    row_count = df.count()
    table_schemas[name] = {
        "full_name": full_name,
        "schema": schema,
        "columns": [(f.name, str(f.dataType)) for f in schema.fields],
        "row_count": row_count,
    }
    print(f"  {name}: {row_count} rows, {len(schema.fields)} columns")
    for f in schema.fields:
        print(f"    - {f.name} ({f.dataType})")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Table Overview Descriptions
# MAGIC
# MAGIC For each table, create a natural language overview of what the table contains,
# MAGIC its columns, and its purpose.

# COMMAND ----------

def generate_table_overview(table_name, info):
    """Generate a natural language overview of a table."""
    columns = info["columns"]
    row_count = info["row_count"]

    # Build column description
    col_descriptions = []
    numeric_cols = []
    text_cols = []
    for col_name, col_type in columns:
        col_descriptions.append(f"{col_name} ({col_type})")
        if col_type in ("DoubleType", "FloatType", "IntegerType", "LongType", "DecimalType(38,18)"):
            numeric_cols.append(col_name)
        elif col_type == "StringType":
            text_cols.append(col_name)

    # Create readable table name
    readable_name = table_name.replace("_", " ").title()

    overview = f"""Table: {table_name}
Description: {readable_name} — This table contains {row_count} records with {len(columns)} fields.
Columns: {', '.join(col_descriptions)}.
Numeric measurements: {', '.join(numeric_cols) if numeric_cols else 'None'}.
Categorical/text fields: {', '.join(text_cols) if text_cols else 'None'}.
"""
    return overview


table_overviews = {}
for name, info in table_schemas.items():
    overview = generate_table_overview(name, info)
    table_overviews[name] = overview
    print(overview)
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Convert Rows to Natural Language
# MAGIC
# MAGIC Each row becomes a text description that contextualizes the values
# MAGIC within the table's purpose and column meanings.

# COMMAND ----------

def row_to_text(row, table_name, columns, table_overview):
    """Convert a single table row into a natural language description."""
    readable_name = table_name.replace("_", " ").title()

    # Build the description parts
    parts = []
    parts.append(f"Record from {readable_name} ({table_name}):")

    # Group values by type for more natural descriptions
    measurements = []
    categories = []
    identifiers = []

    for col_name, col_type in columns:
        value = row[col_name]
        if value is None:
            continue

        clean_name = col_name.replace("_", " ")

        # Identify the type of field
        if col_type in ("DoubleType", "FloatType", "DecimalType(38,18)"):
            if isinstance(value, float):
                measurements.append(f"{clean_name}: {value:.6g}")
            else:
                measurements.append(f"{clean_name}: {value}")
        elif col_type in ("IntegerType", "LongType"):
            measurements.append(f"{clean_name}: {value}")
        elif col_type == "StringType":
            if any(id_hint in col_name.lower() for id_hint in ["id", "name", "code", "key", "sample"]):
                identifiers.append(f"{clean_name}: {value}")
            else:
                categories.append(f"{clean_name}: {value}")
        elif col_type == "BooleanType":
            categories.append(f"{clean_name}: {'Yes' if value else 'No'}")

    if identifiers:
        parts.append("Identification — " + "; ".join(identifiers) + ".")

    if measurements:
        parts.append("Measurements — " + "; ".join(measurements) + ".")

    if categories:
        parts.append("Properties — " + "; ".join(categories) + ".")

    return " ".join(parts)


# Test with first row of first table
test_table = table_names[0]
test_info = table_schemas[test_table]
test_df = spark.table(test_info["full_name"]).limit(1).collect()
if test_df:
    sample_text = row_to_text(test_df[0], test_table, test_info["columns"], table_overviews[test_table])
    print("Sample row conversion:")
    print(sample_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Cross-Table Context
# MAGIC
# MAGIC Create descriptions that explain how tables relate to each other,
# MAGIC giving the LLM understanding of the full data landscape.

# COMMAND ----------

def generate_schema_overview(table_schemas, table_overviews):
    """Generate an overview of the entire schema and table relationships."""
    total_rows = sum(info["row_count"] for info in table_schemas.values())
    total_tables = len(table_schemas)

    overview = f"""Data Schema Overview: {SOURCE_CATALOG}.{SOURCE_SCHEMA}

This schema contains {total_tables} tables with a total of {total_rows} records of scientific data.
The data supports data-driven decision making for scientific analysis and model evaluation.

Tables in this schema:
"""
    for name, info in table_schemas.items():
        readable = name.replace("_", " ").title()
        overview += f"- {readable} ({name}): {info['row_count']} records, {len(info['columns'])} fields\n"

    # Find common columns across tables (potential join keys / relationships)
    all_columns = {}
    for name, info in table_schemas.items():
        for col_name, col_type in info["columns"]:
            if col_name not in all_columns:
                all_columns[col_name] = []
            all_columns[col_name].append(name)

    shared_columns = {col: tables for col, tables in all_columns.items() if len(tables) > 1}
    if shared_columns:
        overview += "\nShared columns across tables (potential relationships):\n"
        for col, tables in shared_columns.items():
            overview += f"- '{col}' appears in: {', '.join(tables)}\n"

    return overview


schema_overview = generate_schema_overview(table_schemas, table_overviews)
print(schema_overview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Process All Tables and Save Training Text

# COMMAND ----------

import pandas as pd

all_training_texts = []

# Add schema overview as training text
all_training_texts.append(schema_overview)

# Add each table's overview
for name, overview in table_overviews.items():
    all_training_texts.append(overview)

# Process each table's rows
for table_name in table_names:
    info = table_schemas[table_name]
    full_name = info["full_name"]
    overview = table_overviews[table_name]

    print(f"Processing {table_name}...")

    df = spark.table(full_name)
    if MAX_ROWS_PER_TABLE:
        df = df.limit(MAX_ROWS_PER_TABLE)

    rows = df.collect()

    # Generate row descriptions in batches
    batch_size = 5  # Group rows together for richer context
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        # Start with table context
        text_parts = [f"Data from {table_name.replace('_', ' ').title()} ({table_name}):"]
        text_parts.append("")

        for row in batch:
            row_text = row_to_text(row, table_name, info["columns"], overview)
            text_parts.append(row_text)

        # Add a summary for the batch
        text_parts.append("")
        text_parts.append(
            f"The above records are from the {table_name.replace('_', ' ')} dataset "
            f"containing {info['row_count']} total records with "
            f"{len(info['columns'])} measured properties."
        )

        full_text = "\n".join(text_parts)
        all_training_texts.append(full_text)

    print(f"  Generated {len(rows) // batch_size + 1} training passages from {len(rows)} rows")

print(f"\nTotal training passages: {len(all_training_texts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save to Unity Catalog

# COMMAND ----------

# Create a DataFrame with the training text
training_df = spark.createDataFrame(
    [(text,) for text in all_training_texts if text.strip()],
    ["text"]
)

# Save to Unity Catalog
dest_full = f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}"
print(f"Saving {training_df.count()} training passages to {dest_full}")

training_df.write.mode("overwrite").saveAsTable(dest_full)

print(f"Done! Training data saved to {dest_full}")
print(f"Use this table in train_cpt.py config.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Preview Training Data

# COMMAND ----------

# Show some samples
preview = spark.table(f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}")
print(f"Total training passages: {preview.count()}")
print(f"Average length: {preview.select(F.avg(F.length('text'))).collect()[0][0]:.0f} characters")
print()
print("=" * 60)
print("SAMPLE TRAINING PASSAGES")
print("=" * 60)

samples = preview.orderBy(F.rand(seed=42)).limit(3).collect()
for i, row in enumerate(samples, 1):
    print(f"\n--- Passage {i} ---")
    print(row["text"][:500])
    if len(row["text"]) > 500:
        print("...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Review the sample passages above — do they make sense?
# MAGIC 2. Update `train_cpt.py` config to point at the new table:
# MAGIC    ```python
# MAGIC    "schema": "gold_roses"  # cpt_training_text table is here now
# MAGIC    ```
# MAGIC 3. Or to train ONLY on the processed text, set a specific table in the config
