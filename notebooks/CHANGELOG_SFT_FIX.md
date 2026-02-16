# SFT Data Generation Fix — Column Type Detection

**What changed:** Fixed column type detection in `notebooks/generate_sft_data.py` that was causing empty Q&A pair categories (comparison, aggregation, uncertainty all generating 0 pairs).

**Root cause 1 — Numeric type detection was too strict:**
- Old code: exact match against `"DecimalType(38,18)"` — missed any other Decimal precision like `DecimalType(10,2)`, `DecimalType(18,6)`, etc.
- Fix: Changed to substring matching using `any(t in col_type.lower() for t in ["double", "float", "integer", "long", "decimal", "short", "byte"]))`
- Added helper function `is_numeric_type()` used in both Step 1 (discovery) and Step 4 (row-level Q&A generation)

**Root cause 2 — Categorical column classification was too aggressive:**
- Old code: any string column with "id", "name", "code", "key", or "sample" in the column name was put into `id_cols` only, removed from `string_cols`
- This meant `string_cols` was likely empty across all 32 tables
- Empty `string_cols` → no categorical stats computed in Step 2 → no comparison Q&A (Step 6), no category-based aggregation Q&A (Step 5), no balance/reasoning Q&A (Step 8)
- Fix: Only "id" and "key" are pure identifiers now. Columns with "name", "code", "sample" go into both `string_cols` and `id_cols`. Also added `BooleanType` as categorical.

**Added debug output in Step 1:**
- Prints which columns are detected as numeric vs categorical per table
- Shows raw Spark types per table
- Warns if total numeric or categorical columns across all tables is 0

**Files changed:** `notebooks/generate_sft_data.py`
**Commit:** `76fb792` on `main`
