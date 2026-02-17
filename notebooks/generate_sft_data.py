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
MAX_TOTAL_PAIRS = 2500

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

    # Pairwise correlations between numeric columns
    stats["correlations"] = {}
    numeric_cols_for_corr = info["numeric_cols"][:MAX_NUMERIC_COLS_PER_TABLE]
    if len(numeric_cols_for_corr) >= 2:
        for i in range(len(numeric_cols_for_corr)):
            for j in range(i + 1, len(numeric_cols_for_corr)):
                col_a = numeric_cols_for_corr[i]
                col_b = numeric_cols_for_corr[j]
                try:
                    corr = df.stat.corr(col_a, col_b)
                    if corr is not None:
                        stats["correlations"][(col_a, col_b)] = corr
                except Exception:
                    continue

    table_stats[name] = stats
    print(f"  {name}: {len(stats['numeric'])} numeric stats, {len(stats['categorical'])} categorical stats, "
          f"{len(stats['correlations'])} correlations")

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

        # Question: What is the range? (includes average in answer)
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
# MAGIC ## Step 10: Uncertainty & Precision Questions
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
# MAGIC ## Step 11: Metadata Completeness & Reproducibility
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
# MAGIC ## Step 12: Correlations & Variable Relationships
# MAGIC
# MAGIC Which variables are correlated? Which are independent?
# MAGIC All computed from pairwise Pearson correlations.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    correlations = stats.get("correlations", {})

    if not correlations:
        continue

    # --- Find strong correlations (|r| > 0.7) ---
    strong_pos = []
    strong_neg = []
    weak = []
    for (col_a, col_b), corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        ca = clean_name(col_a)
        cb = clean_name(col_b)
        if abs(corr) > 0.7:
            if corr > 0:
                strong_pos.append((ca, cb, corr))
            else:
                strong_neg.append((ca, cb, corr))
        elif abs(corr) < 0.2:
            weak.append((ca, cb, corr))

    # Q: Which variables are correlated?
    if strong_pos or strong_neg:
        parts = []
        for ca, cb, corr in strong_pos[:5]:
            parts.append(f"{ca} and {cb} are strongly positively correlated (r={corr:.3f})")
        for ca, cb, corr in strong_neg[:5]:
            parts.append(f"{ca} and {cb} are strongly negatively correlated (r={corr:.3f})")

        qa_pairs.append({
            "instruction": f"Which variables are correlated in the {readable} data?",
            "response": (
                f"In the {table_name} table, the following strong correlations were found: "
                + "; ".join(parts)
                + ". Strong correlations (|r| > 0.7) suggest these variables may be measuring "
                f"related phenomena or one may be derivable from the other."
            ),
            "category": "correlation",
        })

    # Q: Which variables are independent?
    if weak:
        weak_text = "; ".join(
            f"{ca} and {cb} (r={corr:.3f})" for ca, cb, corr in weak[:5]
        )
        qa_pairs.append({
            "instruction": f"Which variables are independent of each other in {readable}?",
            "response": (
                f"In the {table_name} table, these variable pairs show weak or no correlation: "
                f"{weak_text}. "
                f"These variables appear to measure independent aspects of the data."
            ),
            "category": "correlation",
        })

    # Q: Does X affect Y? (for top correlated pairs)
    for (col_a, col_b), corr in sorted(correlations.items(), key=lambda x: -abs(x[1]))[:3]:
        ca = clean_name(col_a)
        cb = clean_name(col_b)
        if abs(corr) < 0.3:
            continue

        strength = "strong" if abs(corr) > 0.7 else "moderate"
        direction = "positive" if corr > 0 else "negative"
        meaning = (
            f"as {ca} increases, {cb} tends to increase"
            if corr > 0 else
            f"as {ca} increases, {cb} tends to decrease"
        )

        qa_pairs.append({
            "instruction": f"Does {ca} affect {cb} in {readable}?",
            "response": (
                f"There is a {strength} {direction} correlation between {ca} and {cb} "
                f"in the {table_name} table (r={corr:.3f}). This means {meaning}. "
                f"Note: correlation does not imply causation — this relationship may be "
                f"driven by a confounding variable."
            ),
            "category": "correlation",
        })

    # Q: Are any measurements redundant?
    redundant = [(ca, cb, corr) for (ca, cb), corr in correlations.items() if abs(corr) > 0.95]
    if redundant:
        redundant_text = "; ".join(
            f"{clean_name(a)} and {clean_name(b)} (r={c:.3f})" for a, b, c in redundant[:3]
        )
        qa_pairs.append({
            "instruction": f"Are any measurements redundant in {readable}?",
            "response": (
                f"The following pairs in {table_name} are very highly correlated (|r| > 0.95), "
                f"suggesting they may be measuring the same thing: {redundant_text}. "
                f"Consider whether both are needed or if one can be derived from the other."
            ),
            "category": "correlation",
        })

    # Q: What is the most important variable? (highest average correlation)
    avg_corr = {}
    for (col_a, col_b), corr in correlations.items():
        avg_corr.setdefault(col_a, []).append(abs(corr))
        avg_corr.setdefault(col_b, []).append(abs(corr))

    if avg_corr:
        most_connected = max(avg_corr.items(), key=lambda x: sum(x[1]) / len(x[1]))
        col_name = most_connected[0]
        avg_r = sum(most_connected[1]) / len(most_connected[1])

        # Find what it correlates with most
        top_partners = []
        for (col_a, col_b), corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
            if col_a == col_name:
                top_partners.append((clean_name(col_b), corr))
            elif col_b == col_name:
                top_partners.append((clean_name(col_a), corr))

        partner_text = "; ".join(
            f"{p} (r={c:.3f})" for p, c in top_partners[:3]
        )

        qa_pairs.append({
            "instruction": f"What is the most connected variable in {readable}?",
            "response": (
                f"In the {table_name} table, {clean_name(col_name)} has the highest average "
                f"correlation with other variables (mean |r|={avg_r:.3f}). "
                f"It is most strongly correlated with: {partner_text}. "
                f"This suggests {clean_name(col_name)} may be a key driver or central measurement."
            ),
            "category": "correlation",
        })

print(f"Generated {len(qa_pairs) - count_before} correlation Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Cross-Table Relationships
# MAGIC
# MAGIC How do tables relate? What can we learn by joining them?
# MAGIC Uses shared columns to link tables.

# COMMAND ----------

count_before = len(qa_pairs)

# Find shared columns across tables
all_columns = {}
for name, info in table_info.items():
    for col_name, col_type in info["columns"]:
        if col_name not in all_columns:
            all_columns[col_name] = []
        all_columns[col_name].append(name)

shared_columns = {col: tbls for col, tbls in all_columns.items() if len(tbls) > 1}

# For each pair of tables that share a column, compute cross-table stats
cross_table_pairs_done = 0
MAX_CROSS_TABLE = 15

for shared_col, tables_with_col in shared_columns.items():
    if cross_table_pairs_done >= MAX_CROSS_TABLE:
        break

    for i in range(len(tables_with_col)):
        for j in range(i + 1, len(tables_with_col)):
            if cross_table_pairs_done >= MAX_CROSS_TABLE:
                break

            t1_name = tables_with_col[i]
            t2_name = tables_with_col[j]
            t1_info = table_info[t1_name]
            t2_info = table_info[t2_name]
            t1_readable = readable_table(t1_name)
            t2_readable = readable_table(t2_name)

            # How many shared values?
            try:
                df1 = spark.table(t1_info["full_name"]).select(shared_col).distinct()
                df2 = spark.table(t2_info["full_name"]).select(shared_col).distinct()
                overlap = df1.intersect(df2).count()
                total_t1 = df1.count()
                total_t2 = df2.count()
            except Exception:
                continue

            if overlap == 0:
                continue

            shared_cn = clean_name(shared_col)

            # Q: How do these tables relate?
            qa_pairs.append({
                "instruction": f"How are {t1_readable} and {t2_readable} related?",
                "response": (
                    f"{t1_name} and {t2_name} share the column '{shared_cn}'. "
                    f"Of {total_t1} distinct values in {t1_name} and {total_t2} in {t2_name}, "
                    f"{overlap} values appear in both tables "
                    f"({100*overlap/min(total_t1, total_t2):.0f}% overlap with the smaller set). "
                    f"These tables can be joined on '{shared_cn}' to combine their measurements."
                ),
                "category": "cross_table",
            })

            # Q: What would I learn by joining them?
            t1_unique_cols = [c for c in t1_info["numeric_cols"] if c not in [cc for cc, _ in t2_info["columns"]]][:3]
            t2_unique_cols = [c for c in t2_info["numeric_cols"] if c not in [cc for cc, _ in t1_info["columns"]]][:3]

            if t1_unique_cols or t2_unique_cols:
                qa_pairs.append({
                    "instruction": f"What would I learn by joining {t1_readable} and {t2_readable}?",
                    "response": (
                        f"Joining {t1_name} and {t2_name} on '{shared_cn}' would combine "
                        + (f"measurements from {t1_name} ({', '.join(clean_name(c) for c in t1_unique_cols)}) " if t1_unique_cols else "")
                        + ("with " if t1_unique_cols and t2_unique_cols else "")
                        + (f"measurements from {t2_name} ({', '.join(clean_name(c) for c in t2_unique_cols)})" if t2_unique_cols else "")
                        + f". This would give a more complete picture of each {shared_cn}, "
                        f"allowing analysis of how measurements in one table relate to the other."
                    ),
                    "category": "cross_table",
                })

            cross_table_pairs_done += 1

print(f"Generated {len(qa_pairs) - count_before} cross-table Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14: Subgroup Analysis
# MAGIC
# MAGIC How do different categories behave? Are there subgroups
# MAGIC that stand out? All computed from grouped statistics.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    df = spark.table(info["full_name"])

    for cat_col, value_counts in stats["categorical"].items():
        if len(value_counts) < 2:
            continue

        cat_cn = clean_name(cat_col)
        top_cats = [str(v["value"]) for v in value_counts[:5] if v["value"] is not None]
        if len(top_cats) < 2:
            continue

        # For each numeric column, compute stats per category
        for num_col in info["numeric_cols"][:3]:
            num_cn = clean_name(num_col)

            try:
                grouped = (
                    df.filter(F.col(cat_col).isin(top_cats))
                    .groupBy(cat_col)
                    .agg(
                        F.avg(num_col).alias("avg"),
                        F.stddev(num_col).alias("std"),
                        F.min(num_col).alias("min_val"),
                        F.max(num_col).alias("max_val"),
                        F.count(num_col).alias("cnt"),
                    )
                    .collect()
                )
            except Exception:
                continue

            if len(grouped) < 2:
                continue

            # Find the category with highest and lowest average
            sorted_groups = sorted(grouped, key=lambda r: r["avg"] if r["avg"] is not None else 0)
            lowest = sorted_groups[0]
            highest = sorted_groups[-1]

            if lowest["avg"] is None or highest["avg"] is None:
                continue

            diff = highest["avg"] - lowest["avg"]
            if highest["avg"] != 0:
                pct_diff = 100 * abs(diff) / abs(highest["avg"])
            else:
                pct_diff = 0

            # Q: Which subgroup has the highest/lowest X?
            qa_pairs.append({
                "instruction": f"Which {cat_cn} category has the highest {num_cn} in {readable}?",
                "response": (
                    f"In the {table_name} table, '{str(highest[cat_col])}' has the highest average "
                    f"{num_cn} at {highest['avg']:.6g} ({highest['cnt']} samples), while "
                    f"'{str(lowest[cat_col])}' has the lowest at {lowest['avg']:.6g} ({lowest['cnt']} samples). "
                    f"The difference is {diff:.6g} ({pct_diff:.1f}%)."
                ),
                "category": "subgroup",
            })

            # Q: Do subgroups behave differently?
            if pct_diff > 20:
                group_detail_parts = []
                for r in sorted_groups:
                    r_avg = r['avg'] if r['avg'] is not None else 0
                    r_std = r['std'] if r['std'] is not None else 0
                    r_cnt = r['cnt'] if r['cnt'] is not None else 0
                    group_detail_parts.append(
                        f"'{str(r[cat_col])}': mean {r_avg:.6g}, std {r_std:.6g}, n={r_cnt}"
                    )
                group_details = "; ".join(group_detail_parts)
                qa_pairs.append({
                    "instruction": f"Do different {cat_cn} groups behave differently for {num_cn} in {readable}?",
                    "response": (
                        f"Yes, there is a notable difference in {num_cn} across {cat_cn} groups "
                        f"in the {table_name} table ({pct_diff:.1f}% spread). "
                        f"Breakdown: {group_details}. "
                        f"This difference may indicate that {cat_cn} is an important factor "
                        f"influencing {num_cn}."
                    ),
                    "category": "subgroup",
                })
            break  # One numeric col per categorical col to avoid explosion

print(f"Generated {len(qa_pairs) - count_before} subgroup Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15: Anomaly & Pattern Detection
# MAGIC
# MAGIC Find unusual patterns, extreme values, and unexpected
# MAGIC distributions — all computed from the data.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    df = spark.table(info["full_name"])
    row_count = info["row_count"]

    # --- Which samples are outliers across MULTIPLE measurements? ---
    numeric_cols = info["numeric_cols"][:MAX_NUMERIC_COLS_PER_TABLE]
    if len(numeric_cols) >= 2 and info["id_cols"]:
        id_col = info["id_cols"][0]

        # Count how many columns each row is an outlier in
        outlier_conditions = []
        for col_name in numeric_cols[:5]:
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
                outlier_conditions.append(
                    F.when((F.col(col_name) < lower) | (F.col(col_name) > upper), 1).otherwise(0)
                )
            except Exception:
                continue

        if len(outlier_conditions) >= 2:
            outlier_score = sum(outlier_conditions)
            multi_outliers = (
                df.withColumn("outlier_count", outlier_score)
                .filter(F.col("outlier_count") >= 2)
                .select(id_col, "outlier_count")
                .orderBy(F.desc("outlier_count"))
                .limit(5)
                .collect()
            )

            if multi_outliers:
                outlier_text = "; ".join(
                    f"{str(r[id_col])} (outlier in {r['outlier_count']} measurements)"
                    for r in multi_outliers
                )
                qa_pairs.append({
                    "instruction": f"Which samples are unusual across multiple measurements in {readable}?",
                    "response": (
                        f"The following samples in {table_name} are outliers in more than one measurement "
                        f"simultaneously: {outlier_text}. "
                        f"Multi-measurement outliers may indicate genuinely unusual samples, "
                        f"measurement errors, or samples from a different population."
                    ),
                    "category": "anomaly",
                })

    # --- Unexpected distributions: bimodality check ---
    for col_name in numeric_cols[:3]:
        col_stats = stats["numeric"].get(col_name)
        if not col_stats or col_stats["avg"] is None or col_stats["stddev"] is None:
            continue

        cn = clean_name(col_name)
        avg = col_stats["avg"]
        std = col_stats["stddev"]
        min_val = col_stats["min"]
        max_val = col_stats["max"]

        if std == 0 or min_val is None or max_val is None:
            continue

        # Check if data clusters at extremes (possible bimodality)
        try:
            midpoint = (min_val + max_val) / 2
            lower_half = df.filter(F.col(col_name) < midpoint).count()
            upper_half = df.filter(F.col(col_name) >= midpoint).count()
            middle_third = df.filter(
                (F.col(col_name) >= min_val + (max_val - min_val) * 0.33) &
                (F.col(col_name) <= min_val + (max_val - min_val) * 0.66)
            ).count()
        except Exception:
            continue

        total_valid = lower_half + upper_half
        if total_valid == 0:
            continue

        middle_pct = 100 * middle_third / total_valid

        # If very few values in the middle, might be bimodal
        if middle_pct < 15 and total_valid > 20:
            qa_pairs.append({
                "instruction": f"Is the {cn} distribution unusual in {readable}?",
                "response": (
                    f"The {cn} distribution in {table_name} may be bimodal — "
                    f"only {middle_pct:.0f}% of values fall in the middle third of the range. "
                    f"There are {lower_half} values in the lower half and {upper_half} in the upper half. "
                    f"This pattern could indicate two distinct populations, different experimental "
                    f"conditions, or a phase transition in the measured phenomenon."
                ),
                "category": "anomaly",
            })
            break  # One per table

    # --- Surprising patterns: which category has unexpected values? ---
    for cat_col, value_counts in stats["categorical"].items():
        if len(value_counts) < 2:
            continue

        cat_cn = clean_name(cat_col)
        top_cats = [str(v["value"]) for v in value_counts[:3] if v["value"] is not None]
        if len(top_cats) < 2:
            continue

        for num_col in info["numeric_cols"][:2]:
            num_cn = clean_name(num_col)
            num_stats = stats["numeric"].get(num_col)
            if not num_stats or num_stats["stddev"] is None or num_stats["stddev"] == 0:
                continue

            try:
                grouped = (
                    df.filter(F.col(cat_col).isin(top_cats))
                    .groupBy(cat_col)
                    .agg(F.avg(num_col).alias("avg"), F.stddev(num_col).alias("std"))
                    .collect()
                )
            except Exception:
                continue

            # Find if any subgroup mean is >2 stddev from overall mean
            overall_avg = num_stats["avg"]
            overall_std = num_stats["stddev"]

            for g in grouped:
                if g["avg"] is None:
                    continue
                z = abs(g["avg"] - overall_avg) / overall_std
                if z > 2:
                    direction = "unusually high" if g["avg"] > overall_avg else "unusually low"
                    qa_pairs.append({
                        "instruction": f"Are there any surprising {num_cn} values by {cat_cn} in {readable}?",
                        "response": (
                            f"Yes — the '{str(g[cat_col])}' group in {table_name} has {direction} "
                            f"{num_cn} (mean {g['avg']:.6g}) compared to the overall mean ({overall_avg:.6g}). "
                            f"This is {z:.1f} standard deviations from the overall average. "
                            f"This may warrant further investigation."
                        ),
                        "category": "anomaly",
                    })
                    break
            break  # One per table

print(f"Generated {len(qa_pairs) - count_before} anomaly Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 16: Scientific Interpretation
# MAGIC
# MAGIC What do the patterns mean? What should a scientist
# MAGIC focus on? Grounded in computed statistics.

# COMMAND ----------

count_before = len(qa_pairs)

for table_name, stats in table_stats.items():
    info = table_info[table_name]
    readable = readable_table(table_name)
    correlations = stats.get("correlations", {})

    # --- What should I focus on in this dataset? ---
    highlights = []

    # Highest variance measurement
    high_cv_cols = []
    for col_name, col_stats in stats["numeric"].items():
        if col_stats["avg"] and col_stats["stddev"] and col_stats["avg"] != 0:
            cv = abs(col_stats["stddev"] / col_stats["avg"])
            high_cv_cols.append((col_name, cv))
    high_cv_cols.sort(key=lambda x: -x[1])

    if high_cv_cols:
        top_cv = high_cv_cols[0]
        highlights.append(
            f"{clean_name(top_cv[0])} has the highest variability (CV={top_cv[1]:.2f}), "
            f"making it the most informative measurement for distinguishing samples"
        )

    # Strongest correlation
    if correlations:
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        highlights.append(
            f"{clean_name(strongest[0][0])} and {clean_name(strongest[0][1])} are strongly "
            f"correlated (r={strongest[1]:.3f}), suggesting a relationship worth investigating"
        )

    # Data size assessment
    row_count = info["row_count"]
    num_cats = len(stats["categorical"])
    if row_count < 30:
        highlights.append(f"with only {row_count} records, statistical conclusions should be treated with caution")
    elif row_count > 1000:
        highlights.append(f"with {row_count} records, there is sufficient data for robust statistical analysis")

    if highlights:
        qa_pairs.append({
            "instruction": f"What should I focus on when analyzing the {readable} data?",
            "response": (
                f"Key observations for the {table_name} table: "
                + ". ".join(highlights) + "."
            ),
            "category": "interpretation",
        })

    # --- What story does this data tell? ---
    story_parts = []
    story_parts.append(
        f"The {table_name} table contains {info['row_count']} records with "
        f"{len(info['numeric_cols'])} measurements"
    )

    if high_cv_cols:
        stable = [c for c, cv in high_cv_cols if cv < 0.1]
        variable = [c for c, cv in high_cv_cols if cv > 0.3]
        if stable:
            story_parts.append(
                f"Measurements like {', '.join(clean_name(c) for c in stable[:3])} are very consistent "
                f"across samples (low variability)"
            )
        if variable:
            story_parts.append(
                f"Measurements like {', '.join(clean_name(c) for c in variable[:3])} show wide variation, "
                f"suggesting they are sensitive to experimental conditions or sample differences"
            )

    strong_corrs = [(a, b, c) for (a, b), c in correlations.items() if abs(c) > 0.7]
    if strong_corrs:
        story_parts.append(
            f"There are {len(strong_corrs)} strongly correlated variable pairs, "
            f"indicating underlying physical or chemical relationships"
        )

    if len(story_parts) > 1:
        qa_pairs.append({
            "instruction": f"What story does the {readable} data tell?",
            "response": ". ".join(story_parts) + ".",
            "category": "interpretation",
        })

    # --- What are the key relationships? ---
    if correlations:
        # Top 5 strongest relationships
        top_corrs = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:5]
        rel_parts = []
        for (col_a, col_b), corr in top_corrs:
            direction = "positively" if corr > 0 else "negatively"
            rel_parts.append(
                f"{clean_name(col_a)} and {clean_name(col_b)} are {direction} correlated (r={corr:.3f})"
            )

        qa_pairs.append({
            "instruction": f"What are the key relationships between variables in {readable}?",
            "response": (
                f"The strongest relationships in the {table_name} table are: "
                + "; ".join(rel_parts)
                + ". These correlations suggest which variables are linked and may "
                f"help identify driving factors in the experimental results."
            ),
            "category": "interpretation",
        })

    # --- Is the data sufficient for analysis? ---
    num_cols = len(info["numeric_cols"])
    row_count = info["row_count"]
    ratio = row_count / max(num_cols, 1)

    sufficiency_note = ""
    if ratio < 5:
        sufficiency_note = (
            f"Warning: with only {ratio:.0f} samples per variable, the dataset may be underpowered "
            f"for multivariate analysis. Consider collecting more data or reducing the number of variables."
        )
    elif ratio < 20:
        sufficiency_note = (
            f"The sample-to-variable ratio is {ratio:.0f}:1, which is adequate for basic analysis "
            f"but may be insufficient for complex modeling. Simple correlations and t-tests are reliable, "
            f"but multivariate models should be validated carefully."
        )
    else:
        sufficiency_note = (
            f"With {ratio:.0f} samples per variable, the dataset is well-powered for statistical analysis "
            f"including multivariate methods."
        )

    qa_pairs.append({
        "instruction": f"Is there enough data in {readable} to draw reliable conclusions?",
        "response": (
            f"The {table_name} table has {row_count} records and {num_cols} numeric measurements. "
            + sufficiency_note
        ),
        "category": "interpretation",
    })

print(f"Generated {len(qa_pairs) - count_before} interpretation Q&A pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 17: Apply Limits and Save

# COMMAND ----------

import random

# Apply hard ceiling
if len(qa_pairs) > MAX_TOTAL_PAIRS:
    print(f"Total pairs ({len(qa_pairs)}) exceeds limit ({MAX_TOTAL_PAIRS}), sampling...")

    # Keep high-value categories, sample from repetitive ones
    priority_categories = {"schema", "reasoning", "correlation", "cross_table",
                          "anomaly", "interpretation", "subgroup"}
    schema_reasoning = [p for p in qa_pairs if p["category"] in priority_categories]
    other = [p for p in qa_pairs if p["category"] not in priority_categories]

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
# MAGIC ## Step 18: Preview Generated Q&A Pairs
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
# MAGIC ## Step 19: Spot Check — Search for Specific Questions

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
