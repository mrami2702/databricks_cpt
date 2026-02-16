# Databricks notebook source
# MAGIC %md
# MAGIC # Generate SFT Training Data
# MAGIC
# MAGIC This notebook reads all tables from Unity Catalog and generates
# MAGIC question-answer pairs for Supervised Fine-Tuning (SFT).
# MAGIC
# MAGIC **No model needed** — this is pure data processing.
# MAGIC Run this AFTER CPT training and BEFORE SFT training.
# MAGIC
# MAGIC The output is a table of (instruction, response, category) rows
# MAGIC that you can review before using for SFT training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source data
SOURCE_CATALOG = "dev_europa"
SOURCE_SCHEMA = "gold_roses"

# Where to save the generated Q&A pairs
DEST_CATALOG = "dev_europa"
DEST_SCHEMA = "gold_roses"
DEST_TABLE = "sft_training_data"

# --- Limits ---
# Row-level: how many rows to sample per table for individual Q&A
MAX_ROWS_PER_TABLE = 20

# Aggregation: max numeric columns to generate stats questions for per table
MAX_NUMERIC_COLS_PER_TABLE = 10

# Comparison: max category combinations per table
MAX_COMPARISONS_PER_TABLE = 5

# Hard ceiling on total Q&A pairs
MAX_TOTAL_PAIRS = 2000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Discover All Tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, FloatType, IntegerType, LongType

tables = spark.catalog.listTables(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}")
table_names = [t.name for t in tables if t.name not in (DEST_TABLE, "cpt_training_text")]

print(f"Found {len(table_names)} tables in {SOURCE_CATALOG}.{SOURCE_SCHEMA}")
print()

# Gather schema info for every table
table_info = {}
for name in table_names:
    full_name = f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{name}"
    df = spark.table(full_name)
    schema = df.schema
    row_count = df.count()

    columns = []
    numeric_cols = []
    string_cols = []
    id_cols = []

    for field in schema.fields:
        col_name = field.name
        col_type = str(field.dataType)
        columns.append((col_name, col_type))

        # Detect types using substring matching to handle all variants
        # (e.g., DecimalType(10,2), VarcharType(255), CharType(50), etc.)
        col_type_lower = col_type.lower()

        if any(t in col_type_lower for t in ["double", "float", "integer", "long", "decimal", "short", "byte"]):
            numeric_cols.append(col_name)
        elif any(t in col_type_lower for t in ["string", "varchar", "char", "text"]):
            # Put into BOTH id_cols and string_cols if it looks like an identifier
            # so it can still be used for categorical analysis
            if any(hint in col_name.lower() for hint in ["id", "key"]):
                id_cols.append(col_name)
            else:
                string_cols.append(col_name)
                # Also mark as identifier if it has name/code/sample
                if any(hint in col_name.lower() for hint in ["name", "code", "sample"]):
                    id_cols.append(col_name)
        elif "boolean" in col_type_lower:
            string_cols.append(col_name)  # Treat booleans as categorical
        elif "date" in col_type_lower or "timestamp" in col_type_lower:
            string_cols.append(col_name)  # Treat dates as categorical for grouping
        else:
            # Unknown type — check if it's not numeric and treat as categorical
            print(f"    WARNING: Unknown type '{col_type}' for column '{col_name}' — treating as categorical")
            string_cols.append(col_name)

    table_info[name] = {
        "full_name": full_name,
        "columns": columns,
        "numeric_cols": numeric_cols,
        "string_cols": string_cols,
        "id_cols": id_cols,
        "row_count": row_count,
    }

    print(f"  {name}: {row_count} rows, {len(columns)} cols "
          f"({len(numeric_cols)} numeric, {len(string_cols)} categorical, {len(id_cols)} identifiers)")
    if numeric_cols:
        print(f"    Numeric: {', '.join(numeric_cols[:5])}" + (f" + {len(numeric_cols)-5} more" if len(numeric_cols) > 5 else ""))
    if string_cols:
        print(f"    Categorical: {', '.join(string_cols[:5])}" + (f" + {len(string_cols)-5} more" if len(string_cols) > 5 else ""))
    # Show raw types for debugging
    type_set = set(ct for _, ct in columns)
    print(f"    Raw types: {type_set}")

print(f"\nTotal tables: {len(table_info)}")
total_numeric = sum(len(info["numeric_cols"]) for info in table_info.values())
total_categorical = sum(len(info["string_cols"]) for info in table_info.values())
print(f"Total numeric columns across all tables: {total_numeric}")
print(f"Total categorical columns across all tables: {total_categorical}")
if total_numeric == 0:
    print("WARNING: No numeric columns detected! Check raw types above.")
if total_categorical == 0:
    print("WARNING: No categorical columns detected! Comparisons and some aggregations will be skipped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Compute Statistics Per Table
# MAGIC
# MAGIC We need aggregated stats to generate accurate answers for
# MAGIC aggregation and comparison questions.

# COMMAND ----------

table_stats = {}

for name, info in table_info.items():
    full_name = info["full_name"]
    df = spark.table(full_name)
    stats = {"numeric": {}, "categorical": {}}

    # Numeric column stats
    numeric_cols = info["numeric_cols"][:MAX_NUMERIC_COLS_PER_TABLE]
    if numeric_cols:
        # Compute min, max, avg, stddev for each numeric column
        agg_exprs = []
        for col in numeric_cols:
            agg_exprs.extend([
                F.min(col).alias(f"{col}__min"),
                F.max(col).alias(f"{col}__max"),
                F.avg(col).alias(f"{col}__avg"),
                F.stddev(col).alias(f"{col}__stddev"),
                F.count(F.when(F.col(col).isNotNull(), 1)).alias(f"{col}__count"),
            ])

        agg_row = df.agg(*agg_exprs).collect()[0]

        for col in numeric_cols:
            stats["numeric"][col] = {
                "min": agg_row[f"{col}__min"],
                "max": agg_row[f"{col}__max"],
                "avg": agg_row[f"{col}__avg"],
                "stddev": agg_row[f"{col}__stddev"],
                "count": agg_row[f"{col}__count"],
            }

    # Categorical column stats (distinct values and counts)
    for col in info["string_cols"]:
        try:
            value_counts = (
                df.groupBy(col)
                .count()
                .orderBy(F.desc("count"))
                .limit(20)
                .collect()
            )
            stats["categorical"][col] = [
                {"value": row[col], "count": row["count"]}
                for row in value_counts
                if row[col] is not None
            ]
        except Exception:
            continue

    table_stats[name] = stats
    print(f"  {name}: {len(stats['numeric'])} numeric stats, {len(stats['categorical'])} categorical stats")

print("\nStatistics computed for all tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Sample Rows Per Table

# COMMAND ----------

table_samples = {}

for name, info in table_info.items():
    full_name = info["full_name"]
    df = spark.table(full_name)

    # Sample rows randomly
    sample_rows = df.orderBy(F.rand(seed=42)).limit(MAX_ROWS_PER_TABLE).collect()
    table_samples[name] = sample_rows
    print(f"  {name}: sampled {len(sample_rows)} rows")

print(f"\nTotal sampled rows: {sum(len(rows) for rows in table_samples.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Level 1 — Row-Level Questions
# MAGIC
# MAGIC Questions about specific records with direct lookup answers.

# COMMAND ----------

qa_pairs = []

def clean_name(col_name):
    return col_name.replace("_", " ")

def readable_table(table_name):
    return table_name.replace("_", " ").title()

def is_numeric_type(col_type):
    ct = col_type.lower()
    return any(t in ct for t in ["double", "float", "integer", "long", "decimal", "short", "byte"])

def format_value(value, col_type):
    if value is None:
        return "not recorded"
    if is_numeric_type(col_type):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    return str(value)


for table_name, rows in table_samples.items():
    info = table_info[table_name]
    readable = readable_table(table_name)

    for row in rows:
        # Find an identifier for this row
        row_id = None
        row_id_col = None
        for id_col in info["id_cols"]:
            if row[id_col] is not None:
                row_id = str(row[id_col])
                row_id_col = id_col
                break

        if row_id is None:
            # Use first non-null column as identifier
            for col_name, col_type in info["columns"]:
                if row[col_name] is not None:
                    row_id = str(row[col_name])
                    row_id_col = col_name
                    break

        if row_id is None:
            continue

        # Build answer with all measurements
        measurements = []
        properties = []
        for col_name, col_type in info["columns"]:
            if col_name == row_id_col:
                continue
            value = row[col_name]
            if value is None:
                continue
            formatted = format_value(value, col_type)
            cn = clean_name(col_name)
            if is_numeric_type(col_type):
                measurements.append(f"{cn} of {formatted}")
            else:
                properties.append(f"{cn} is {formatted}")

        # Question type 1: Ask about all measurements
        if measurements:
            question = f"What are the measurements for {clean_name(row_id_col)} {row_id} in the {readable} data?"
            answer_parts = [f"For {clean_name(row_id_col)} {row_id} in the {table_name} table:"]
            answer_parts.append(f"Measurements: {'; '.join(measurements)}.")
            if properties:
                answer_parts.append(f"Properties: {'; '.join(properties)}.")
            qa_pairs.append({
                "instruction": question,
                "response": " ".join(answer_parts),
                "category": "row_level",
            })

        # Question type 2: Ask about a specific column
        for col_name, col_type in info["columns"]:
            if col_name == row_id_col or row[col_name] is None:
                continue
            # Only generate for ~30% of columns to avoid too many pairs
            if hash(f"{row_id}{col_name}") % 3 != 0:
                continue

            formatted = format_value(row[col_name], col_type)
            question = f"What is the {clean_name(col_name)} for {clean_name(row_id_col)} {row_id} in {readable}?"
            answer = (
                f"The {clean_name(col_name)} for {clean_name(row_id_col)} {row_id} "
                f"in the {table_name} table is {formatted}."
            )
            qa_pairs.append({
                "instruction": question,
                "response": answer,
                "category": "row_level",
            })

print(f"Generated {len(qa_pairs)} row-level Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Generate Level 2 — Aggregation Questions
# MAGIC
# MAGIC Questions about averages, ranges, and distributions.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    row_count = info["row_count"]

    for col_name, col_stats in stats["numeric"].items():
        cn = clean_name(col_name)
        avg_val = col_stats["avg"]
        min_val = col_stats["min"]
        max_val = col_stats["max"]
        count = col_stats["count"]
        stddev = col_stats["stddev"]

        if avg_val is None:
            continue

        # Question: What is the average?
        qa_pairs.append({
            "instruction": f"What is the average {cn} in the {readable} data?",
            "response": (
                f"The average {cn} in the {table_name} table is {avg_val:.6g}, "
                f"computed across {count} measurements. "
                f"Values range from {format_value(min_val, 'DoubleType')} "
                f"to {format_value(max_val, 'DoubleType')}."
            ),
            "category": "aggregation",
        })

        # Question: What is the range?
        qa_pairs.append({
            "instruction": f"What is the range of {cn} values in {readable}?",
            "response": (
                f"The {cn} values in {table_name} range from "
                f"{format_value(min_val, 'DoubleType')} to {format_value(max_val, 'DoubleType')}, "
                f"with a mean of {avg_val:.6g}"
                + (f" and standard deviation of {stddev:.6g}" if stddev else "")
                + f". There are {count} non-null measurements."
            ),
            "category": "aggregation",
        })

    # Categorical counts
    for col_name, value_counts in stats["categorical"].items():
        if not value_counts:
            continue
        cn = clean_name(col_name)
        top_values = value_counts[:5]
        values_text = ", ".join(
            f"{str(v['value'])} ({v['count']} records)" for v in top_values
        )

        qa_pairs.append({
            "instruction": f"What are the most common {cn} values in {readable}?",
            "response": (
                f"The most common {cn} values in the {table_name} table are: {values_text}. "
                f"The table contains {row_count} total records."
            ),
            "category": "aggregation",
        })

        # How many distinct values?
        qa_pairs.append({
            "instruction": f"How many distinct {cn} values exist in {readable}?",
            "response": (
                f"There are at least {len(value_counts)} distinct {cn} values "
                f"in the {table_name} table. "
                f"The most frequent is '{str(value_counts[0]['value'])}' with {value_counts[0]['count']} records."
            ),
            "category": "aggregation",
        })

print(f"Generated {len(qa_pairs) - count_before} aggregation Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Level 3 — Comparison Questions
# MAGIC
# MAGIC Compare numeric metrics across categories within each table.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    numeric_cols = info["numeric_cols"][:MAX_NUMERIC_COLS_PER_TABLE]
    comparisons_made = 0

    for cat_col, value_counts in stats["categorical"].items():
        if comparisons_made >= MAX_COMPARISONS_PER_TABLE:
            break
        if len(value_counts) < 2:
            continue

        # Pick top 2-3 categories to compare
        top_categories = [v["value"] for v in value_counts[:3] if v["value"] is not None]
        if len(top_categories) < 2:
            continue

        cat_cn = clean_name(cat_col)
        df = spark.table(info["full_name"])

        for num_col in numeric_cols[:3]:
            num_cn = clean_name(num_col)

            # Compute average per category
            try:
                cat_avgs = (
                    df.filter(F.col(cat_col).isin(top_categories))
                    .groupBy(cat_col)
                    .agg(
                        F.avg(num_col).alias("avg_val"),
                        F.count(num_col).alias("cnt"),
                    )
                    .collect()
                )
            except Exception:
                continue

            if len(cat_avgs) < 2:
                continue

            # Build comparison answer
            comparison_parts = []
            for row in cat_avgs:
                cat_val = row[cat_col]
                avg = row["avg_val"]
                cnt = row["cnt"]
                if avg is not None:
                    comparison_parts.append(
                        f"{cat_val} has an average {num_cn} of {avg:.6g} ({cnt} samples)"
                    )

            if len(comparison_parts) < 2:
                continue

            question = f"How does {num_cn} compare across different {cat_cn} values in {readable}?"
            answer = (
                f"Comparing {num_cn} by {cat_cn} in the {table_name} table: "
                + "; ".join(comparison_parts) + "."
            )

            qa_pairs.append({
                "instruction": question,
                "response": answer,
                "category": "comparison",
            })

            comparisons_made += 1

print(f"Generated {len(qa_pairs) - count_before} comparison Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Generate Level 4 — Schema-Level Questions
# MAGIC
# MAGIC Questions about the overall data structure and table relationships.

# COMMAND ----------

count_before = len(qa_pairs)

# Question: What tables exist?
table_list = ", ".join(
    f"{readable_table(name)} ({info['row_count']} rows)"
    for name, info in table_info.items()
)
qa_pairs.append({
    "instruction": f"What tables are available in the {SOURCE_SCHEMA} schema?",
    "response": (
        f"The {SOURCE_CATALOG}.{SOURCE_SCHEMA} schema contains {len(table_info)} tables: {table_list}."
    ),
    "category": "schema",
})

# Question: Total data size
total_rows = sum(info["row_count"] for info in table_info.values())
total_cols = sum(len(info["columns"]) for info in table_info.values())
qa_pairs.append({
    "instruction": f"How much data is in the {SOURCE_SCHEMA} schema?",
    "response": (
        f"The {SOURCE_SCHEMA} schema contains {len(table_info)} tables "
        f"with a total of {total_rows} records and {total_cols} columns across all tables."
    ),
    "category": "schema",
})

# Question per table: describe this table
for name, info in table_info.items():
    readable = readable_table(name)
    col_names = [c[0] for c in info["columns"]]

    qa_pairs.append({
        "instruction": f"Describe the {readable} table.",
        "response": (
            f"The {name} table contains {info['row_count']} records with {len(info['columns'])} columns: "
            f"{', '.join(clean_name(c) for c in col_names)}. "
            f"It has {len(info['numeric_cols'])} numeric measurement columns "
            f"and {len(info['string_cols'])} categorical columns."
        ),
        "category": "schema",
    })

    # What columns does this table have?
    qa_pairs.append({
        "instruction": f"What columns does the {name} table have?",
        "response": (
            f"The {name} table has {len(info['columns'])} columns: "
            + ", ".join(f"{clean_name(c)} ({t})" for c, t in info["columns"])
            + "."
        ),
        "category": "schema",
    })

# Shared columns across tables
all_columns = {}
for name, info in table_info.items():
    for col_name, col_type in info["columns"]:
        if col_name not in all_columns:
            all_columns[col_name] = []
        all_columns[col_name].append(name)

shared = {col: tbls for col, tbls in all_columns.items() if len(tbls) > 1}
if shared:
    shared_text = "; ".join(
        f"'{clean_name(col)}' appears in {', '.join(tbls)}"
        for col, tbls in shared.items()
    )
    qa_pairs.append({
        "instruction": f"Which columns are shared across multiple tables in {SOURCE_SCHEMA}?",
        "response": (
            f"Several columns appear in multiple tables, indicating relationships: {shared_text}. "
            f"These shared columns can be used to join data across tables."
        ),
        "category": "schema",
    })

    qa_pairs.append({
        "instruction": f"How are the tables in {SOURCE_SCHEMA} related to each other?",
        "response": (
            f"The tables in {SOURCE_SCHEMA} are related through shared columns: {shared_text}. "
            f"These common fields allow you to link records across tables for cross-table analysis."
        ),
        "category": "schema",
    })

print(f"Generated {len(qa_pairs) - count_before} schema-level Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Generate Level 5 — Reasoning Questions
# MAGIC
# MAGIC Higher-level questions about data interpretation and insights.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)

    # Find high-variance columns (interesting for analysis)
    high_variance_cols = []
    for col_name, col_stats in stats["numeric"].items():
        if col_stats["stddev"] and col_stats["avg"] and col_stats["avg"] != 0:
            cv = abs(col_stats["stddev"] / col_stats["avg"])
            if cv > 0.3:
                high_variance_cols.append((col_name, cv, col_stats))

    if high_variance_cols:
        col_name, cv, col_stats = high_variance_cols[0]
        cn = clean_name(col_name)
        qa_pairs.append({
            "instruction": f"Which measurements show the most variability in {readable}?",
            "response": (
                f"In the {table_name} table, {cn} shows high variability with a "
                f"coefficient of variation of {cv:.2f}. "
                f"Values range from {format_value(col_stats['min'], 'DoubleType')} "
                f"to {format_value(col_stats['max'], 'DoubleType')} "
                f"with a mean of {col_stats['avg']:.6g} and standard deviation of {col_stats['stddev']:.6g}. "
                f"This suggests significant variation in experimental conditions or sample properties."
            ),
            "category": "reasoning",
        })

    # Data quality question
    total_cols = len(info["columns"])
    numeric_count = len(info["numeric_cols"])
    qa_pairs.append({
        "instruction": f"What can you tell me about the data quality of {readable}?",
        "response": (
            f"The {table_name} table has {info['row_count']} records across {total_cols} columns. "
            f"It contains {numeric_count} numeric measurement columns and "
            f"{len(info['string_cols'])} categorical columns. "
            + (
                f"The most variable measurement is {clean_name(high_variance_cols[0][0])} "
                f"(CV={high_variance_cols[0][1]:.2f}), which may warrant further investigation. "
                if high_variance_cols else ""
            )
            + "Review null counts and outlier distributions for a complete quality assessment."
        ),
        "category": "reasoning",
    })

    # Which category has the most data?
    for cat_col, value_counts in stats["categorical"].items():
        if len(value_counts) >= 2:
            cat_cn = clean_name(cat_col)
            top = value_counts[0]
            bottom = value_counts[-1]
            qa_pairs.append({
                "instruction": f"Is the {readable} data balanced across {cat_cn} categories?",
                "response": (
                    f"The {table_name} data shows imbalance across {cat_cn}: "
                    f"the most common value '{str(top['value'])}' has {top['count']} records, "
                    f"while '{str(bottom['value'])}' has only {bottom['count']}. "
                    f"This imbalance should be considered when building predictive models, "
                    f"as underrepresented categories may have less reliable statistics."
                ),
                "category": "reasoning",
            })
            break  # One per table is enough

print(f"Generated {len(qa_pairs) - count_before} reasoning Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Data Quality & QC Questions
# MAGIC
# MAGIC Outlier detection, null rates, distribution checks —
# MAGIC all computed directly from the data.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    df = spark.table(info["full_name"])
    row_count = info["row_count"]

    # --- Null rate analysis ---
    null_counts = {}
    for col_name, col_type in info["columns"]:
        n = df.filter(F.col(col_name).isNull()).count()
        if n > 0:
            null_counts[col_name] = n

    if null_counts:
        null_text = "; ".join(
            f"{clean_name(c)} has {n} nulls ({100*n/row_count:.1f}%)"
            for c, n in sorted(null_counts.items(), key=lambda x: -x[1])
        )
        qa_pairs.append({
            "instruction": f"Are there any missing values in the {readable} data?",
            "response": (
                f"Yes, the {table_name} table has missing values in {len(null_counts)} columns: "
                f"{null_text}. "
                f"The table has {row_count} total records."
            ),
            "category": "data_quality",
        })
    else:
        qa_pairs.append({
            "instruction": f"Are there any missing values in the {readable} data?",
            "response": (
                f"No, the {table_name} table has no missing values across all "
                f"{len(info['columns'])} columns and {row_count} records."
            ),
            "category": "data_quality",
        })

    # --- Outlier detection using IQR for numeric columns ---
    for col_name in info["numeric_cols"][:5]:
        cn = clean_name(col_name)
        try:
            quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
            if len(quantiles) < 2 or quantiles[0] is None or quantiles[1] is None:
                continue
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_count = df.filter(
                (F.col(col_name) < lower) | (F.col(col_name) > upper)
            ).count()

            if outlier_count > 0:
                qa_pairs.append({
                    "instruction": f"Are there outliers in {cn} in {readable}?",
                    "response": (
                        f"Using the IQR method (1.5x interquartile range), "
                        f"there are {outlier_count} potential outliers in {cn} "
                        f"in the {table_name} table. "
                        f"Q1={q1:.6g}, Q3={q3:.6g}, IQR={iqr:.6g}. "
                        f"Values below {lower:.6g} or above {upper:.6g} are flagged. "
                        f"This represents {100*outlier_count/row_count:.1f}% of {row_count} records."
                    ),
                    "category": "data_quality",
                })
            else:
                qa_pairs.append({
                    "instruction": f"Are there outliers in {cn} in {readable}?",
                    "response": (
                        f"Using the IQR method, no outliers were detected in {cn} "
                        f"in the {table_name} table. "
                        f"All {row_count} values fall within the expected range "
                        f"({lower:.6g} to {upper:.6g})."
                    ),
                    "category": "data_quality",
                })
        except Exception:
            continue

    # --- Sanity checks: what to verify ---
    checks = []
    if null_counts:
        worst_col = max(null_counts, key=null_counts.get)
        checks.append(
            f"check {clean_name(worst_col)} for missing values "
            f"({null_counts[worst_col]} nulls, {100*null_counts[worst_col]/row_count:.1f}%)"
        )
    for col_name, col_stats in stats["numeric"].items():
        if col_stats["min"] is not None and col_stats["max"] is not None:
            if col_stats["min"] == col_stats["max"]:
                checks.append(f"investigate {clean_name(col_name)} — all values are identical ({col_stats['min']})")
                break
    if info["string_cols"]:
        first_cat = info["string_cols"][0]
        cat_vals = stats["categorical"].get(first_cat, [])
        if cat_vals and cat_vals[0]["count"] > 0.8 * row_count:
            checks.append(
                f"review {clean_name(first_cat)} — value '{str(cat_vals[0]['value'])}' "
                f"dominates at {100*cat_vals[0]['count']/row_count:.0f}% of records"
            )

    if checks:
        qa_pairs.append({
            "instruction": f"What sanity checks should I run on the {readable} data before analysis?",
            "response": (
                f"For the {table_name} table ({row_count} records), key checks include: "
                + "; ".join(checks)
                + ". Also verify that numeric ranges are physically plausible "
                f"and that no duplicate records exist."
            ),
            "category": "data_quality",
        })

print(f"Generated {len(qa_pairs) - count_before} data quality Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Data Transformation & Schema Documentation
# MAGIC
# MAGIC Column definitions, observed ranges, skewness detection,
# MAGIC and unit consistency checks across tables.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)

    # --- Full schema definition ---
    schema_parts = []
    for col_name, col_type in info["columns"]:
        cn = clean_name(col_name)
        col_stat = stats["numeric"].get(col_name)
        if col_stat and col_stat["avg"] is not None:
            schema_parts.append(
                f"{cn} ({col_type}): range [{format_value(col_stat['min'], col_type)} "
                f"to {format_value(col_stat['max'], col_type)}], "
                f"mean {col_stat['avg']:.6g}"
            )
        elif col_name in stats["categorical"] and stats["categorical"][col_name]:
            vals = [str(v["value"]) for v in stats["categorical"][col_name][:5]]
            schema_parts.append(
                f"{cn} ({col_type}): values include {', '.join(vals)}"
            )
        else:
            schema_parts.append(f"{cn} ({col_type})")

    qa_pairs.append({
        "instruction": f"Create a schema definition for the {readable} table with column meanings and ranges.",
        "response": (
            f"Schema definition for {table_name} ({info['row_count']} records):\n"
            + "\n".join(f"- {p}" for p in schema_parts)
        ),
        "category": "schema_documentation",
    })

    # --- Skewness / distribution shape for numeric columns ---
    for col_name, col_stats in stats["numeric"].items():
        cn = clean_name(col_name)
        if col_stats["avg"] is None or col_stats["min"] is None or col_stats["max"] is None:
            continue
        avg = col_stats["avg"]
        min_val = col_stats["min"]
        max_val = col_stats["max"]
        data_range = max_val - min_val
        if data_range == 0:
            continue

        # Check if mean is far from midpoint (indicates skew)
        midpoint = (min_val + max_val) / 2
        if data_range > 0:
            skew_ratio = (avg - midpoint) / data_range
            if abs(skew_ratio) > 0.15:
                direction = "right (higher values)" if skew_ratio > 0 else "left (lower values)"
                qa_pairs.append({
                    "instruction": f"Is the {cn} distribution skewed in {readable}?",
                    "response": (
                        f"The {cn} distribution in {table_name} appears skewed toward the {direction}. "
                        f"The mean ({avg:.6g}) is offset from the midpoint of the range "
                        f"({midpoint:.6g}). Range: {format_value(min_val, 'DoubleType')} to "
                        f"{format_value(max_val, 'DoubleType')}. "
                        f"Consider log or power transformation if using this in a linear model."
                    ),
                    "category": "schema_documentation",
                })
                break  # One per table

# --- Cross-table unit consistency ---
# Check if same column name appears in multiple tables with different ranges
col_ranges = {}
for table_name, stats in table_stats.items():
    for col_name, col_stats in stats["numeric"].items():
        if col_stats["avg"] is None:
            continue
        if col_name not in col_ranges:
            col_ranges[col_name] = []
        col_ranges[col_name].append({
            "table": table_name,
            "min": col_stats["min"],
            "max": col_stats["max"],
            "avg": col_stats["avg"],
        })

for col_name, ranges in col_ranges.items():
    if len(ranges) < 2:
        continue
    cn = clean_name(col_name)

    # Check if ranges are wildly different (possible unit mismatch)
    avgs = [r["avg"] for r in ranges]
    if max(avgs) > 0 and min(avgs) > 0:
        ratio = max(avgs) / min(avgs)
        if ratio > 100:
            table_details = "; ".join(
                f"{r['table']} has range [{format_value(r['min'], 'DoubleType')} to "
                f"{format_value(r['max'], 'DoubleType')}], mean {r['avg']:.6g}"
                for r in ranges
            )
            qa_pairs.append({
                "instruction": f"Is {cn} measured consistently across tables?",
                "response": (
                    f"Warning: {cn} shows very different ranges across tables, "
                    f"which may indicate different units or scales. "
                    f"Details: {table_details}. "
                    f"Verify that the same units are used before joining or comparing across tables."
                ),
                "category": "schema_documentation",
            })
        else:
            table_details = "; ".join(
                f"{r['table']} mean {r['avg']:.6g}"
                for r in ranges
            )
            qa_pairs.append({
                "instruction": f"Is {cn} consistent across tables?",
                "response": (
                    f"The {cn} values are in a similar range across tables: {table_details}. "
                    f"This suggests consistent units and measurement methods."
                ),
                "category": "schema_documentation",
            })

print(f"Generated {len(qa_pairs) - count_before} schema documentation Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Uncertainty & Precision Questions
# MAGIC
# MAGIC Confidence intervals, coefficient of variation, and
# MAGIC measurement precision — all from computed statistics.

# COMMAND ----------

import math

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)

    for col_name, col_stats in stats["numeric"].items():
        cn = clean_name(col_name)
        avg = col_stats["avg"]
        stddev = col_stats["stddev"]
        count = col_stats["count"]

        if avg is None or stddev is None or count is None or count < 2:
            continue

        # --- Coefficient of variation ---
        if avg != 0:
            cv = abs(stddev / avg)
            precision = "high" if cv < 0.1 else "moderate" if cv < 0.3 else "low"
            qa_pairs.append({
                "instruction": f"How precise are the {cn} measurements in {readable}?",
                "response": (
                    f"The {cn} measurements in {table_name} have {precision} precision "
                    f"with a coefficient of variation (CV) of {cv:.4f} ({cv*100:.1f}%). "
                    f"Mean: {avg:.6g}, standard deviation: {stddev:.6g}, "
                    f"based on {count} measurements."
                ),
                "category": "uncertainty",
            })

        # --- Confidence interval for the mean ---
        se = stddev / math.sqrt(count)
        ci_lower = avg - 1.96 * se
        ci_upper = avg + 1.96 * se
        qa_pairs.append({
            "instruction": f"What is the 95% confidence interval for the mean {cn} in {readable}?",
            "response": (
                f"The 95% confidence interval for the mean {cn} in {table_name} is "
                f"[{ci_lower:.6g}, {ci_upper:.6g}]. "
                f"This is based on {count} measurements with mean {avg:.6g} "
                f"and standard error {se:.6g}."
            ),
            "category": "uncertainty",
        })

print(f"Generated {len(qa_pairs) - count_before} uncertainty Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Metadata Completeness & Reproducibility
# MAGIC
# MAGIC Check what metadata exists, what is missing,
# MAGIC and generate documentation checklists.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name in table_names:
    info = table_info[table_name]
    readable = readable_table(table_name)
    df = spark.table(info["full_name"])
    row_count = info["row_count"]

    # --- Metadata completeness ---
    complete_cols = []
    incomplete_cols = []
    for col_name, col_type in info["columns"]:
        null_count = df.filter(F.col(col_name).isNull()).count()
        pct = 100 * (1 - null_count / row_count) if row_count > 0 else 0
        if null_count == 0:
            complete_cols.append(col_name)
        else:
            incomplete_cols.append((col_name, pct))

    completeness_pct = 100 * len(complete_cols) / len(info["columns"]) if info["columns"] else 0

    qa_pairs.append({
        "instruction": f"How complete is the metadata in the {readable} table?",
        "response": (
            f"The {table_name} table has {completeness_pct:.0f}% column completeness. "
            f"{len(complete_cols)} of {len(info['columns'])} columns have no missing values"
            + (
                f". Columns with gaps: "
                + ", ".join(f"{clean_name(c)} ({p:.0f}% complete)" for c, p in incomplete_cols)
                if incomplete_cols else ". All columns are fully populated"
            )
            + "."
        ),
        "category": "reproducibility",
    })

    # --- What metadata is missing for reproducibility? ---
    has_date = any("date" in c.lower() or "time" in c.lower() for c, _ in info["columns"])
    has_operator = any("operator" in c.lower() or "user" in c.lower() or "analyst" in c.lower() for c, _ in info["columns"])
    has_batch = any("batch" in c.lower() or "run" in c.lower() or "lot" in c.lower() for c, _ in info["columns"])
    has_instrument = any("instrument" in c.lower() or "device" in c.lower() or "equipment" in c.lower() for c, _ in info["columns"])
    has_units = any("unit" in c.lower() for c, _ in info["columns"])

    missing = []
    if not has_date:
        missing.append("timestamp or date of measurement")
    if not has_operator:
        missing.append("operator or analyst identifier")
    if not has_batch:
        missing.append("batch or run identifier")
    if not has_instrument:
        missing.append("instrument or device identifier")
    if not has_units:
        missing.append("units column or unit metadata")

    present = []
    if has_date:
        present.append("timestamps")
    if has_operator:
        present.append("operator info")
    if has_batch:
        present.append("batch identifiers")
    if has_instrument:
        present.append("instrument info")

    qa_pairs.append({
        "instruction": f"What metadata is missing from {readable} for reproducibility?",
        "response": (
            f"For full reproducibility of the {table_name} data"
            + (f", the following metadata is present: {', '.join(present)}" if present else "")
            + (f". Missing metadata that would improve reproducibility: {', '.join(missing)}" if missing else ". All key reproducibility metadata appears to be present")
            + ". Consider adding calibration references and standard operating procedure versions if not tracked elsewhere."
        ),
        "category": "reproducibility",
    })

print(f"Generated {len(qa_pairs) - count_before} reproducibility Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Data Summary Questions
# MAGIC
# MAGIC Concise factual summaries of what the data contains,
# MAGIC suitable for lab meetings or quick overviews.

# COMMAND ----------

count_before = len(qa_pairs)

# --- Overall schema summary ---
total_rows = sum(info["row_count"] for info in table_info.values())
total_numeric = sum(len(info["numeric_cols"]) for info in table_info.values())
total_categorical = sum(len(info["string_cols"]) for info in table_info.values())
largest_table = max(table_info.items(), key=lambda x: x[1]["row_count"])
smallest_table = min(table_info.items(), key=lambda x: x[1]["row_count"])

qa_pairs.append({
    "instruction": f"Give me a quick summary of the {SOURCE_SCHEMA} data for a lab meeting.",
    "response": (
        f"The {SOURCE_SCHEMA} schema contains {len(table_info)} tables with {total_rows} total records. "
        f"There are {total_numeric} numeric measurement columns and {total_categorical} categorical columns across all tables. "
        f"The largest table is {largest_table[0]} ({largest_table[1]['row_count']} records) "
        f"and the smallest is {smallest_table[0]} ({smallest_table[1]['row_count']} records)."
    ),
    "category": "summary",
})

qa_pairs.append({
    "instruction": f"Summarize the {SOURCE_SCHEMA} dataset in one paragraph.",
    "response": (
        f"The {SOURCE_CATALOG}.{SOURCE_SCHEMA} dataset comprises {len(table_info)} tables "
        f"containing {total_rows} records of scientific data. "
        f"The tables capture {total_numeric} numeric measurements and {total_categorical} categorical properties. "
        + (
            f"Tables are linked through {len(shared)} shared columns, "
            f"enabling cross-table analysis. "
            if shared else ""
        )
        + f"The largest table ({largest_table[0]}) has {largest_table[1]['row_count']} records "
        f"while the smallest ({smallest_table[0]}) has {smallest_table[1]['row_count']}."
    ),
    "category": "summary",
})

# --- Per-table summaries ---
for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)

    # Build a quick stats line for key numeric columns
    key_stats = []
    for col_name in info["numeric_cols"][:5]:
        col_stats = stats["numeric"].get(col_name)
        if col_stats and col_stats["avg"] is not None:
            cn = clean_name(col_name)
            key_stats.append(f"{cn} (mean: {col_stats['avg']:.6g})")

    # Build category breakdown
    cat_info = []
    for col_name in info["string_cols"][:2]:
        cat_vals = stats["categorical"].get(col_name, [])
        if cat_vals:
            cn = clean_name(col_name)
            top_vals = ", ".join(str(v["value"]) for v in cat_vals[:3])
            cat_info.append(f"{cn}: {top_vals}")

    qa_pairs.append({
        "instruction": f"Summarize the {readable} data.",
        "response": (
            f"The {table_name} table contains {info['row_count']} records with "
            f"{len(info['columns'])} columns. "
            + (f"Key measurements: {'; '.join(key_stats)}. " if key_stats else "")
            + (f"Categories: {'; '.join(cat_info)}. " if cat_info else "")
        ),
        "category": "summary",
    })

print(f"Generated {len(qa_pairs) - count_before} summary Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14: Apply Limits and Save

# COMMAND ----------

import random

# Apply hard ceiling
if len(qa_pairs) > MAX_TOTAL_PAIRS:
    print(f"Total pairs ({len(qa_pairs)}) exceeds limit ({MAX_TOTAL_PAIRS}), sampling...")

    # Keep all schema and reasoning (they are few), sample from others
    schema_reasoning = [p for p in qa_pairs if p["category"] in ("schema", "reasoning")]
    other = [p for p in qa_pairs if p["category"] not in ("schema", "reasoning")]

    remaining = MAX_TOTAL_PAIRS - len(schema_reasoning)
    random.seed(42)
    random.shuffle(other)
    qa_pairs = schema_reasoning + other[:remaining]

print(f"\nFinal Q&A pair counts:")
categories = {}
for p in qa_pairs:
    cat = p["category"]
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")
print(f"  TOTAL: {len(qa_pairs)}")

# Save to Unity Catalog
sft_df = spark.createDataFrame(qa_pairs)
dest_full = f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}"

sft_df.write.mode("overwrite").saveAsTable(dest_full)

print(f"\nSaved to {dest_full}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15: Preview Generated Q&A Pairs
# MAGIC
# MAGIC Review samples from each category to verify quality.

# COMMAND ----------

preview_df = spark.table(f"{DEST_CATALOG}.{DEST_SCHEMA}.{DEST_TABLE}")
total = preview_df.count()

print("=" * 70)
print(f"SFT TRAINING DATA SUMMARY — {total} total Q&A pairs")
print("=" * 70)

# Count by category
print("\nBy category:")
category_counts = preview_df.groupBy("category").count().orderBy("category").collect()
for row in category_counts:
    print(f"  {row['category']}: {row['count']}")

# Show samples from each category
for cat_row in category_counts:
    cat = cat_row["category"]
    print()
    print("=" * 70)
    print(f"SAMPLE — {cat.upper()}")
    print("=" * 70)

    samples = (
        preview_df
        .filter(F.col("category") == cat)
        .orderBy(F.rand(seed=42))
        .limit(3)
        .collect()
    )

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Q: {sample['instruction']}")
        print(f"A: {sample['response']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 16: Spot Check — Search for Specific Questions

# COMMAND ----------

# Change the search term below and re-run to find specific Q&A pairs
search_term = "average"

results = (
    preview_df
    .filter(
        F.lower(F.col("instruction")).contains(search_term.lower())
        | F.lower(F.col("response")).contains(search_term.lower())
    )
    .limit(5)
    .collect()
)

print(f"Found {len(results)} pairs matching '{search_term}':\n")
for r in results:
    print(f"Q: {r['instruction']}")
    print(f"A: {r['response']}")
    print(f"Category: {r['category']}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Review the samples above — do the questions and answers make sense?
# MAGIC 2. If something looks off, adjust the templates in Steps 4-8 and re-run
# MAGIC 3. Once satisfied, run `train_sft_mistral.py` to train on these Q&A pairs
# MAGIC 4. The SFT training notebook will read from `dev_europa.gold_roses.sft_training_data`
