# SFT Training Data Overview

## What This Is

The SFT (Supervised Fine-Tuning) data generation notebook (`notebooks/generate_sft_data.py`) reads all 32 tables from `dev_europa.gold_roses` in Unity Catalog and produces question-answer pairs that teach the model how to respond to natural language questions about the data.

Every answer is computed directly from the source data. Nothing is made up or inferred.

## Source Data

- **Catalog**: `dev_europa`
- **Schema**: `gold_roses`
- **Tables**: 32 tables of experimental scientific data (chemical properties, analytical responses, sample measurements)
- **Output**: Saved to `dev_europa.gold_roses.sft_training_data` as a table with columns: `instruction`, `response`, `category`

## Q&A Categories

The notebook generates 10 categories of question-answer pairs. Here is what each covers and why it matters.

---

### 1. Row-Level (`row_level`)

**What**: Questions about individual records — direct value lookups.

**How it works**: Samples up to 20 rows per table. For each row, generates questions asking about its measurements and properties. Answers pull exact values from the row.

**Example**:
- Q: "What are the measurements for sample id S-001 in the Gold Roses data?"
- A: "For sample id S-001 in the gold_roses table: Measurements: temperature of 312.5; pressure of 1.2; analytical response of 0.847. Properties: material type is oxide."

**Why it matters**: Teaches the model to retrieve and present specific data points when asked about individual records.

---

### 2. Aggregation (`aggregation`)

**What**: Questions about computed statistics — averages, ranges, distributions, category counts.

**How it works**: Computes min, max, mean, stddev for every numeric column. Counts distinct values and frequencies for every categorical column. Generates questions from those computed values.

**Example**:
- Q: "What is the average temperature in the Gold Roses data?"
- A: "The average temperature in the gold_roses table is 305.2, computed across 1500 measurements. Values range from 198.4 to 412.1."

**Why it matters**: Teaches the model to answer statistical questions with accurate, computed answers instead of guessing.

---

### 3. Comparison (`comparison`)

**What**: Questions comparing a numeric measurement across categories within a table.

**How it works**: Groups data by categorical columns, computes average of numeric columns per group, generates comparison Q&A. Limited to top 3 categories and 5 comparisons per table.

**Example**:
- Q: "How does temperature compare across different material type values in Gold Roses?"
- A: "Comparing temperature by material type in the gold_roses table: oxide has an average temperature of 312.4 (47 samples); sulfide has an average temperature of 287.1 (23 samples)."

**Why it matters**: Teaches the model to compare groups using real computed differences.

---

### 4. Schema (`schema`)

**What**: Questions about table structure, column names, data sizes, and table relationships.

**How it works**: Lists all tables with row counts and column counts. Identifies shared columns across tables (potential join keys). Generates descriptions per table.

**Example**:
- Q: "What tables are available in the gold_roses schema?"
- A: "The dev_europa.gold_roses schema contains 32 tables: Gold Roses (1500 rows), Analytical Results (12000 rows), ..."
- Q: "Which columns are shared across multiple tables?"
- A: "Several columns appear in multiple tables: 'sample id' appears in gold_roses, analytical_results, sample_properties. These shared columns can be used to join data across tables."

**Why it matters**: Teaches the model the structure of the data so it can answer questions about what data exists and how tables connect.

---

### 5. Reasoning (`reasoning`)

**What**: Higher-level observations about data patterns — variability, balance, quality flags.

**How it works**: Computes coefficient of variation to find high-variability columns. Checks category balance (most vs least common values). All observations backed by computed numbers.

**Example**:
- Q: "Which measurements show the most variability in Gold Roses?"
- A: "In the gold_roses table, analytical response shows high variability with a coefficient of variation of 0.42. Values range from 0.12 to 0.97 with a mean of 0.72 and standard deviation of 0.30."
- Q: "Is the Gold Roses data balanced across material type categories?"
- A: "The gold_roses data shows imbalance across material type: the most common value 'oxide' has 847 records, while 'sulfide' has only 23."

**Why it matters**: Teaches the model to surface insights about data patterns without making unsupported claims.

---

### 6. Data Quality (`data_quality`)

**What**: Missing value analysis, outlier detection, and sanity check recommendations.

**How it works**:
- **Nulls**: Counts null values per column, reports exact counts and percentages.
- **Outliers**: Uses the IQR method (1.5x interquartile range) on numeric columns. Reports Q1, Q3, IQR, bounds, and count of flagged values.
- **Sanity checks**: Identifies constant columns (all same value), dominant categories (one value >80%), and worst null offenders.

**Example**:
- Q: "Are there outliers in temperature in Gold Roses?"
- A: "Using the IQR method, there are 12 potential outliers in temperature in the gold_roses table. Q1=245.3, Q3=387.1, IQR=141.8. Values below 32.6 or above 599.8 are flagged. This represents 0.8% of 1500 records."
- Q: "What sanity checks should I run on the Gold Roses data before analysis?"
- A: "For the gold_roses table (1500 records), key checks include: check analytical response for missing values (45 nulls, 3.0%); review material type — value 'oxide' dominates at 85% of records."

**Why it matters**: Teaches the model to perform and communicate data quality assessments — a critical step before any scientific analysis.

---

### 7. Schema Documentation (`schema_documentation`)

**What**: Formal column definitions with observed ranges, skewness detection, and unit consistency checks across tables.

**How it works**:
- **Schema definitions**: Lists every column with its type and observed range/values.
- **Skewness**: Compares mean to midpoint of range. If significantly offset, flags as skewed with transformation recommendation.
- **Unit consistency**: Checks if the same column name appears in multiple tables with wildly different ranges (ratio >100x), which suggests different units.

**Example**:
- Q: "Create a schema definition for the Gold Roses table with column meanings and ranges."
- A: "Schema definition for gold_roses (1500 records): temperature (DoubleType): range [198.4 to 412.1], mean 305.2; pressure (DoubleType): range [0.1 to 2.8], mean 1.15; ..."
- Q: "Is temperature measured consistently across tables?"
- A: "Warning: temperature shows very different ranges across tables, which may indicate different units or scales. Details: gold_roses has range [198.4 to 412.1], mean 305.2; thermal_analysis has range [0.2 to 0.4], mean 0.31. Verify that the same units are used before joining."

**Why it matters**: Teaches the model to generate documentation and catch data integration issues (unit mismatches are a real problem in multi-table scientific datasets).

---

### 8. Uncertainty (`uncertainty`)

**What**: Measurement precision, confidence intervals, and coefficient of variation.

**How it works**:
- **CV**: Computes stddev/mean for each numeric column. Classifies as high (<10%), moderate (10-30%), or low (>30%) precision.
- **Confidence intervals**: Computes 95% CI for the mean using standard error (stddev / sqrt(n)).

**Example**:
- Q: "How precise are the temperature measurements in Gold Roses?"
- A: "The temperature measurements in gold_roses have moderate precision with a coefficient of variation (CV) of 0.1842 (18.4%). Mean: 305.2, standard deviation: 56.24, based on 1500 measurements."
- Q: "What is the 95% confidence interval for the mean temperature in Gold Roses?"
- A: "The 95% confidence interval for the mean temperature in gold_roses is [302.4, 308.1]. This is based on 1500 measurements with mean 305.2 and standard error 1.45."

**Why it matters**: Teaches the model to communicate measurement uncertainty correctly — essential for any scientific data discussion.

---

### 9. Reproducibility (`reproducibility`)

**What**: Metadata completeness checks and reproducibility gap analysis.

**How it works**:
- **Completeness**: Counts how many columns have zero nulls vs partial data. Reports percentage.
- **Metadata gaps**: Checks column names for indicators of key reproducibility metadata: timestamps, operator/analyst IDs, batch/run IDs, instrument IDs, and units. Reports what is present and what is missing.

**Example**:
- Q: "How complete is the metadata in the Gold Roses table?"
- A: "The gold_roses table has 85% column completeness. 17 of 20 columns have no missing values. Columns with gaps: analytical response (97% complete), calibration ref (72% complete)."
- Q: "What metadata is missing from Gold Roses for reproducibility?"
- A: "Missing metadata that would improve reproducibility: timestamp or date of measurement, operator identifier, instrument identifier. Consider adding calibration references and SOP versions."

**Why it matters**: Teaches the model to evaluate whether a dataset has sufficient metadata for another scientist to reproduce the experiment.

---

### 10. Summary (`summary`)

**What**: Concise factual overviews of the schema and individual tables.

**How it works**: Pulls together row counts, column counts, key measurement means, and top categories into short summary paragraphs. One overall summary plus one per table.

**Example**:
- Q: "Give me a quick summary of the gold_roses data for a lab meeting."
- A: "The gold_roses schema contains 32 tables with 45,000 total records. There are 87 numeric measurement columns and 23 categorical columns across all tables. The largest table is analytical_results (12,000 records) and the smallest is calibration_refs (50 records)."
- Q: "Summarize the Gold Roses data."
- A: "The gold_roses table contains 1500 records with 20 columns. Key measurements: temperature (mean: 305.2); pressure (mean: 1.15); analytical response (mean: 0.72). Categories: material type: oxide, sulfide, carbonate."

**Why it matters**: Teaches the model to produce concise, presentation-ready summaries that a scientist can use directly in a lab meeting or report.

---

## Limits and Controls

| Setting | Default | Purpose |
|---|---|---|
| `MAX_ROWS_PER_TABLE` | 20 | Rows sampled per table for row-level Q&A |
| `MAX_NUMERIC_COLS_PER_TABLE` | 10 | Numeric columns analyzed per table |
| `MAX_COMPARISONS_PER_TABLE` | 5 | Category comparisons per table |
| `MAX_TOTAL_PAIRS` | 2000 | Hard ceiling on total Q&A pairs |

When the hard ceiling is hit, schema, reasoning, data quality, uncertainty, reproducibility, and summary pairs are kept in full (they are fewer and higher value). Row-level and aggregation pairs are sampled down.

## Expected Output Volume

With 32 tables, estimated pair counts:
- Row-level: ~400-600
- Aggregation: ~200-400
- Comparison: ~100-160
- Schema: ~70-80
- Reasoning: ~60-90
- Data quality: ~200-300
- Schema documentation: ~50-100
- Uncertainty: ~200-300
- Reproducibility: ~64
- Summary: ~35

**Total estimated: 1,400-2,000 Q&A pairs**

## Key Design Decisions

1. **Everything is computed from real data** — no answers are inferred, assumed, or hallucinated. If a statistic can't be computed (e.g., null values, constant columns), the question is skipped.

2. **Row-level questions are sampled, not exhaustive** — with potentially millions of rows, generating a Q&A per row would be wasteful. 20 rows per table gives enough variety.

3. **Aggregation answers use all rows** — even though we sample rows for row-level questions, the stats (mean, stddev, etc.) are computed over the full dataset for accuracy.

4. **IQR method for outliers** — chosen because it doesn't assume normal distribution, which is appropriate for scientific data that may be skewed.

5. **Unit consistency uses a 100x ratio threshold** — if the same column name has averages that differ by more than 100x across tables, it's flagged as a likely unit mismatch. This is conservative to avoid false positives.

6. **Reproducibility checks use column name patterns** — we look for keywords like "date", "operator", "batch", "instrument" in column names. This works for well-named columns but may miss abbreviated names.

## How to Use This Data

After generation:
1. Review samples in the notebook preview (Step 15-16)
2. Search for specific terms to spot-check accuracy
3. If satisfied, proceed to `train_sft_mistral.py` which reads from `dev_europa.gold_roses.sft_training_data`
4. The SFT training formats each pair as `[INST] {instruction} [/INST] {response}` for Mistral
