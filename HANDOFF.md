# Handoff Document: Local → Databricks GPU Migration

## Overview

This document summarizes all changes made to migrate the Continual Pretraining (CPT) pipeline from a local execution environment to Databricks GPU clusters. These changes enable the pipeline to read training data directly from Unity Catalog, leverage Databricks-managed MLflow, and run training on GPU-enabled clusters (V100).

---

## Architecture: Before vs After

### Before (Local Pipeline)
```
data_prep.py → train.py → evaluate.py → mlflow_query.py
     │              │            │              │
     ▼              ▼            ▼              ▼
 Databricks    Local JSONL   Local model    Local MLflow
 SQL export    files on      artifacts in   tracking in
 to local      disk          ./outputs/     ./mlruns/
 JSONL
```

- `data_prep.py` connects to Databricks via SQL connector, exports data to local `data/train.jsonl` and `data/val.jsonl`
- `train.py` reads those local JSONL files, trains with QLoRA, saves model to `./outputs/cpt_model/`
- MLflow tracking is manual — requires `MLFLOW_TRACKING_URI` env var and `.env` file
- Authentication handled via `.env` file with `DATABRICKS_HOST` and `DATABRICKS_TOKEN`

### After (Databricks Pipeline)
```
notebooks/train_cpt.py → evaluate.py → mlflow_query.py
        │                      │              │
        ▼                      ▼              ▼
 Unity Catalog           DBFS model      Databricks
 direct read via         artifacts in    MLflow (auto-
 spark.table()           /dbfs/mnt/      configured)
```

- `data_prep.py` is no longer needed — training reads directly from Unity Catalog tables via `spark.table()`
- `train.py` detects if it's running on Databricks and adapts behavior automatically
- MLflow is auto-configured by the Databricks runtime — no manual URI setup needed
- No `.env` file needed — Databricks handles authentication natively
- New `notebooks/train_cpt.py` provides a self-contained Databricks notebook for end-to-end training

---

## What Was Changed and Why

### 1. `train.py` — Modified

**Why:** The training script needed to support reading data from Unity Catalog instead of local JSONL files, and to leverage Databricks' auto-configured MLflow instead of requiring manual setup.

**What changed:**

#### a) Removed `dotenv` dependency
```python
# REMOVED:
from dotenv import load_dotenv
load_dotenv()

# ADDED:
def is_databricks():
    """Check if running on Databricks."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
```
**Why:** On Databricks, environment variables and authentication are managed by the runtime. The `dotenv` pattern is only needed locally. The `is_databricks()` helper lets the code adapt its behavior based on where it's running.

#### b) Modified `setup_mlflow()` (line 42)
```python
# BEFORE: Always set tracking URI manually
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

# AFTER: Skip on Databricks, manual only when local
if is_databricks():
    print("Databricks detected — MLflow tracking auto-configured")
else:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
```
**Why:** Databricks Runtime ML pre-configures MLflow to point at the workspace's managed MLflow instance. Calling `set_tracking_uri()` on Databricks would override this and break the integration.

#### c) Added `load_from_unity_catalog()` function (line 129)
```python
def load_from_unity_catalog(table_name: str, text_column: str = "text"):
    from pyspark.sql import SparkSession
    from datasets import Dataset
    spark = SparkSession.builder.getOrCreate()
    df = spark.table(table_name)
    pdf = df.select(text_column).toPandas()
    pdf = pdf.rename(columns={text_column: "text"})
    pdf = pdf.dropna(subset=["text"])
    return Dataset.from_pandas(pdf)
```
**Why:** This is the bridge between Databricks (Spark/Unity Catalog) and HuggingFace (Datasets). It reads a Unity Catalog table via Spark, converts to pandas, then to a HuggingFace Dataset that the Trainer can consume. The `pyspark` import is deferred (inside the function) so the script doesn't fail when running locally where PySpark isn't installed.

#### d) Modified `prepare_datasets()` (line 154)
```python
# ADDED: Check for Unity Catalog tables first
train_table = data_config.get("train_table")
val_table = data_config.get("val_table")

if train_table and val_table and is_databricks():
    domain_train = load_from_unity_catalog(train_table, text_column)
    domain_val = load_from_unity_catalog(val_table, text_column)
else:
    # Original JSONL loading (unchanged)
    domain_train = load_dataset("json", data_files=data_config["train_file"], split="train")
    domain_val = load_dataset("json", data_files=data_config["val_file"], split="train")
```
**Why:** This is the key data loading change. When `train_table` and `val_table` are set in the config AND the code is running on Databricks, it reads directly from Unity Catalog. Otherwise, it falls back to the original JSONL file loading. This makes the script backwards-compatible — it works both locally and on Databricks without any code changes.

#### e) Updated default config path (line 310, 401)
```python
# BEFORE:
def train(config_path: str = "configs/cpt_config.yaml"):
# AFTER:
def train(config_path: str = "cpt_config.yaml"):
```
**Why:** The project uses a flat directory structure (config file is at root level, not in a `configs/` subdirectory). This was a bug fix to match the actual file layout.

**What stayed the same in train.py:**
- `load_base_model()` — QLoRA/model loading logic is identical
- `create_training_arguments()` — TrainingArguments construction is identical
- `train()` main function — Trainer setup, MLflow logging, model saving all identical
- Replay data mixing logic — unchanged
- All hyperparameter handling — unchanged

---

### 2. `cpt_config.yaml` — Modified

**Why:** The config needed new fields to specify Unity Catalog table paths as data sources.

**What changed:**

```yaml
data:
  # ADDED: Unity Catalog tables (used on Databricks)
  train_table: null  # e.g. "my_catalog.my_schema.train_documents"
  val_table: null    # e.g. "my_catalog.my_schema.val_documents"

  # KEPT: Local file paths (fallback when tables are not set)
  train_file: "data/train.jsonl"
  val_file: "data/val.jsonl"

output:
  # ADDED comment showing DBFS path option:
  # Local: "./outputs/cpt_model"
  # Databricks: "/dbfs/mnt/models/cpt_model"
  output_dir: "./outputs/cpt_model"
```

**Why:**
- `train_table` / `val_table` set to `null` by default so the local JSONL fallback is used unless explicitly configured
- The DBFS output path is shown as a comment so the user knows what to change when running on Databricks
- All other config (model, training, qlora, mlflow) is unchanged

**Action needed:** Before running on Databricks, update `train_table` and `val_table` with actual Unity Catalog table paths (e.g., `"my_catalog.my_schema.train_documents"`) and change `output_dir` to a DBFS path.

---

### 3. `notebooks/train_cpt.py` — New File

**Why:** Databricks notebooks are the standard way to run workloads on Databricks clusters. This provides a self-contained, cell-by-cell notebook that can be uploaded to a Databricks workspace and executed on a GPU cluster.

**What it contains:**

| Cell | Purpose |
|------|---------|
| 1 | Markdown header with cluster requirements |
| 2 | `%pip install peft bitsandbytes accelerate` |
| 3 | `dbutils.library.restartPython()` (required after pip install on Databricks) |
| 4 | Markdown: configuration section |
| 5 | Full config dict (inline, not YAML) — **user must update `train_table` and `val_table` here** |
| 6 | Markdown: data loading section |
| 7 | Load data from Unity Catalog via `spark.table()`, convert to HuggingFace Dataset |
| 8 | Markdown: model loading section |
| 9 | Load base model with QLoRA quantization (4-bit, LoRA adapters) |
| 10 | Markdown: tokenization section |
| 11 | Tokenize train/val datasets |
| 12 | Markdown: training section |
| 13 | Set up MLflow, TrainingArguments, Trainer — run training, save model, log to MLflow |
| 14 | Markdown: verification section |
| 15 | List saved model files as a sanity check |

**Why inline config instead of importing from YAML:** On Databricks, file paths can be tricky (workspace files vs repos vs DBFS). Inlining the config makes the notebook fully self-contained — no dependencies on file layout. The config values match `cpt_config.yaml` exactly.

**Why not just import `train.py`:** While possible (via Databricks Repos), a self-contained notebook is easier to debug cell-by-cell, modify on-the-fly, and share with team members who may not have the full repo cloned.

**Cluster requirements:**
- Runtime: Databricks Runtime ML 14.x+
- Instance: GPU-enabled (`Standard_NC6s_v3` = 1x V100 16GB)
- Libraries installed by notebook: `peft>=0.7.0`, `bitsandbytes>=0.41.0`, `accelerate>=0.24.0`

---

### 4. `sample_outputs/` — New Directory (3 files)

**Why:** These mock outputs show exactly what the pipeline produces when run end-to-end, useful for:
- Understanding the expected output format before running real training
- Frontend/app development (the future standalone app can be built against these schemas)
- Demos and stakeholder presentations

#### a) `sample_outputs/training_metrics.json`
Mock MLflow training run data including:
- Run metadata (run_id, experiment_name, timestamps, duration)
- Training parameters (model, learning rate, batch size, QLoRA config)
- Training log (loss at each logging step, learning rate schedule, eval metrics at checkpoints)
- Final metrics (train_loss: 1.491, eval_perplexity: 4.66, trainable params: 1.04%)
- Artifact paths (DBFS model path, MLflow artifact URI, checkpoint paths)

#### b) `sample_outputs/evaluation_results.json`
Mock evaluation comparing base model vs CPT model:
- Domain perplexity: base=8.42, CPT=4.66, improvement=44.7%
- 5 side-by-side generation comparisons showing:
  - Base model gives generic, vague responses
  - CPT model gives domain-specific responses with concrete metrics, criteria, and actionable recommendations
- Summary verdict: "SUCCESS: Significant domain adaptation achieved"

#### c) `sample_outputs/recommendation.json`
Mock model recommendation output:
- Task: neutron flux prediction for reactor safety
- 3 models evaluated (XGBoost, Neural Network, Linear — from the existing sample data in `mlflow_query.py`)
- Recommendation: Neural Network model with detailed justification, acceptance criteria met, and risk considerations
- Matches the output schema of `mlflow_query.py`'s `recommend_model()` function

---

## Files NOT Changed

| File | Why No Changes Needed |
|------|----------------------|
| `data_prep.py` | No longer needed on Databricks (Unity Catalog replaces the export step). Kept in repo for local development/testing. |
| `evaluate.py` | Already path-agnostic — just pass `--cpt-model /dbfs/mnt/models/cpt_model` and `--eval-file` pointing to DBFS data. No code changes required. |
| `mlflow_query.py` | Already has demo mode with sample data. Already supports configurable MLflow URI. The `recommend_model()` function works unchanged — on Databricks, just omit `--mlflow-uri` flag and it defaults to `"databricks"`. |
| `requirements.txt` | Already includes all needed dependencies (`peft`, `bitsandbytes`, `accelerate`, `databricks-sdk`, `mlflow`). |
| `README.md` | Not updated in this change — should be updated separately to reflect the Databricks workflow. |
| `.env.example` | Kept for local development reference, but not needed on Databricks. |

---

## How to Run on Databricks

### Option A: Using the Notebook (Recommended)
1. Upload `notebooks/train_cpt.py` to your Databricks workspace
2. Attach to a GPU cluster (Runtime ML 14.x+, Standard_NC6s_v3)
3. Update the `config` dict in cell 5:
   - Set `train_table` to your Unity Catalog training table path
   - Set `val_table` to your Unity Catalog validation table path
4. Run All cells

### Option B: Using train.py as a Databricks Job
1. Upload the repo to Databricks via Repos (connect GitHub repo)
2. Update `cpt_config.yaml`:
   - Set `train_table` and `val_table` to Unity Catalog paths
   - Set `output_dir` to `/dbfs/mnt/models/cpt_model`
3. Create a job that runs `python train.py --config cpt_config.yaml`
4. Attach to a GPU cluster with `peft`, `bitsandbytes`, `accelerate` installed

### Post-Training
1. Run evaluation: `python evaluate.py --cpt-model /dbfs/mnt/models/cpt_model`
2. Run recommendation: `python mlflow_query.py --demo` (or without `--demo` if MLflow registry is populated)

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────┐
│  Unity Catalog                               │
│  my_catalog.my_schema.train_documents        │
│  my_catalog.my_schema.val_documents          │
└──────────────┬───────────────────────────────┘
               │ spark.table()
               ▼
┌──────────────────────────────────────────────┐
│  train.py / notebooks/train_cpt.py           │
│                                              │
│  Spark DataFrame → pandas → HF Dataset       │
│  QLoRA (4-bit quantization + LoRA adapters)  │
│  HuggingFace Trainer                         │
│  MLflow logging (auto-configured)            │
└──────┬───────────────┬───────────────────────┘
       │               │
       ▼               ▼
┌─────────────┐  ┌─────────────────────────────┐
│ DBFS         │  │ MLflow                       │
│ /dbfs/mnt/   │  │ Experiment: continual-       │
│ models/      │  │ pretraining-poc              │
│ cpt_model/   │  │ Artifacts: model, metrics    │
└──────┬───────┘  └──────────────┬──────────────┘
       │                         │
       ▼                         ▼
┌─────────────┐  ┌─────────────────────────────┐
│ evaluate.py  │  │ mlflow_query.py              │
│ Perplexity   │  │ Query registry → build       │
│ comparison   │  │ prompt → CPT model →         │
│ Base vs CPT  │  │ recommendation               │
└─────────────┘  └─────────────────────────────┘
                         │
                         ▼
                 ┌─────────────────────────────┐
                 │ Future: Standalone App       │
                 │ Natural language interface   │
                 │ Calls serving endpoint       │
                 └─────────────────────────────┘
```

---

## Future Work (Not in This Change)

1. **Standalone App** — Build a FastAPI/Streamlit app that wraps `mlflow_query.py` logic as an API, with a natural language chat interface
2. **Model Serving Endpoint** — Deploy the CPT model as a Databricks serving endpoint so the app can call it via REST API instead of loading it into memory
3. **Automated Retraining** — Set up `train.py` as a scheduled Databricks job that retrains when new data arrives in Unity Catalog
4. **Update README.md** — Reflect the new Databricks workflow and notebook instructions
