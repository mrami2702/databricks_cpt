# Scientific AI Assistant — CPT + SFT Pipeline on Databricks

A pipeline for building a domain-specific scientific AI assistant by training Mistral-7B on experimental/scientific data from Unity Catalog. The model learns your data through two training phases, then answers questions like a scientist who has studied every table in your catalog.

Think of it like hiring a new lab assistant: CPT is teaching them to read your lab notebooks, and SFT is teaching them how to answer questions about what they read.

## Pipeline Overview

The pipeline has 4 phases, run in order:

```
Phase 1: CPT Training          Phase 2: SFT Data Generation
(teach domain vocabulary)       (create Q&A pairs from data)
        |                               |
        v                               v
Phase 3: SFT Training          Phase 4: Interactive Chat
(teach Q&A behavior)            (ask the model questions)
```

| Phase | Notebook | What It Does | Time Estimate |
|-------|----------|-------------|---------------|
| 1 | `train_cpt_mistral.py` | Trains Mistral-7B on raw scientific text from Unity Catalog | ~13 hours on V100 |
| 2a | `generate_sft_data.py` | Generates Q&A pairs from your scientific tables | ~15-30 min |
| 2b | `generate_sft_mlflow.py` | Generates Q&A pairs from MLflow model registry | ~5-10 min |
| 3 | `train_sft_mistral.py` | Fine-tunes the CPT model on Q&A pairs | ~30 min - 2 hours |
| 4 | `interactive_chat_demo.py` | Gradio chat UI to interact with the trained model | Instant |

## Cluster Requirements

- **Runtime:** Databricks Runtime ML 14.x+
- **Instance:** GPU-enabled with V100 16GB (e.g., `Standard_NC6s_v3`)
- **Model:** `mistralai/Mistral-7B-v0.3` (7.2B params, ~1% trained via QLoRA)

## Project Structure

```
databricks_cpt/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── cpt_config.yaml                    # CPT training hyperparameters (local dev)
├── .env.example                       # Databricks connection template (local dev)
│
├── notebooks/                         # === DATABRICKS NOTEBOOKS (main pipeline) ===
│   ├── train_cpt_mistral.py           # Phase 1: Continual pretraining
│   ├── generate_sft_data.py           # Phase 2a: Scientific Q&A generation (16 categories)
│   ├── generate_sft_mlflow.py         # Phase 2b: MLflow model selection Q&A generation
│   ├── train_sft_mistral.py           # Phase 3: SFT training on Q&A pairs
│   ├── interactive_chat_demo.py       # Phase 4: Gradio chat interface
│   ├── train_cpt.py                   # Earlier CPT notebook (generic, kept for reference)
│   ├── prepare_training_data.py       # Data prep helper (standalone)
│   ├── recommend.py                   # Model recommendation helper
│   ├── CHANGELOG_SFT_FIX.md           # Fix: numeric type detection
│   ├── CHANGELOG_CATEGORICAL_FIX.md   # Fix: categorical type detection
│   └── CHANGELOG_SUBGROUP_FIX.md      # Fix: subgroup NoneType formatting
│
├── data_prep.py                       # Local data extraction from Databricks
├── train.py                           # Local training script
├── evaluate.py                        # Model evaluation (base vs CPT)
├── mlflow_query.py                    # MLflow model querying
├── compare_models.py                  # Model comparison utilities
├── api_server.py                      # API server for model serving
├── local_chat.py                      # Local chat interface
├── sample_outputs/                    # Mock outputs showing expected results
│   ├── training_metrics.json
│   ├── evaluation_results.json
│   └── recommendation.json
├── demo_app.html                      # Demo application
├── training_databricks_workflow.html   # Visual workflow diagram
└── workflow_diagram.html              # Pipeline flow diagram
```

## Phase-by-Phase Guide

### Phase 1: CPT (Continual Pretraining)

**Notebook:** `notebooks/train_cpt_mistral.py`

CPT is like teaching the model a new language — it reads all your scientific data as natural language text and learns the vocabulary, patterns, and relationships. This is next-token prediction (no Q&A yet).

**What it does:**
1. Reads all tables from Unity Catalog (`dev_europa.gold_roses`)
2. Converts tabular data into natural language paragraphs
3. Trains Mistral-7B with QLoRA (4-bit quantization + LoRA adapters)
4. Logs training metrics to MLflow

**Configuration (fill in the blanks at the top of the notebook):**
- Source catalog/schema for your data
- Output path for the trained model (e.g., `/dbfs/mnt/models/cpt_model_mistral`)
- MLflow experiment name

**Output:** A CPT model saved to DBFS that understands your domain vocabulary.

---

### Phase 2a: Scientific SFT Data Generation

**Notebook:** `notebooks/generate_sft_data.py`

This notebook reads your scientific tables and automatically generates question-answer pairs — like creating a study guide from a textbook. No model is needed for this step; it's pure data processing.

**16 Q&A Categories Generated:**

| Category | What It Asks | Example |
|----------|-------------|---------|
| `row_level` | Specific record lookups | "What are the measurements for sample X?" |
| `aggregation` | Ranges, distributions, value counts | "What is the range of pH values?" |
| `comparison` | Cross-category comparisons | "How does yield compare across catalysts?" |
| `schema` | Table structure and relationships | "What tables are available?" |
| `reasoning` | Data interpretation, variability | "Which measurements show the most variability?" |
| `data_quality` | Nulls, outliers, sanity checks | "Are there outliers in temperature?" |
| `uncertainty` | Confidence intervals, precision (CV) | "What is the 95% CI for the mean?" |
| `reproducibility` | Metadata completeness | "What metadata is missing for reproducibility?" |
| `correlation` | Variable relationships (Pearson r) | "Which variables are correlated?" |
| `cross_table` | How tables relate via shared columns | "How are these two tables related?" |
| `subgroup` | Category-level behavior differences | "Which catalyst has the highest yield?" |
| `anomaly` | Unusual patterns, bimodality, multi-outliers | "Which samples are unusual across multiple measurements?" |
| `interpretation` | Scientific narrative, key relationships | "What story does this data tell?" |

**Configuration:**
- `SOURCE_CATALOG` / `SOURCE_SCHEMA`: where your data lives
- `MAX_ROWS_PER_TABLE`: rows sampled per table (default 20)
- `MAX_TOTAL_PAIRS`: hard ceiling on Q&A pairs (default 2500)

**Output:** Table saved to `{DEST_CATALOG}.{DEST_SCHEMA}.sft_training_data` with columns: `instruction`, `response`, `category`

---

### Phase 2b: MLflow Model Selection Q&A

**Notebook:** `notebooks/generate_sft_mlflow.py`

Generates Q&A pairs about the ML models in your MLflow registry — which model is best, how to compare them, when to use each one. Think of it as teaching the assistant to be your model selection advisor.

**Categories:**
- `model_info` — individual model descriptions, metrics, parameters
- `model_comparison` — pairwise and ranking comparisons
- `recommendation` — which model to deploy, latency vs accuracy
- `deployment` — production status, model counts
- `trade_off` — accuracy vs speed, version progression, risk assessment
- `registry_overview` — tracked metrics/params, comparable models
- `scientific_reliability` — how reliable for scientific decisions, confidence levels
- `scientific_reproducibility` — can results be reproduced, what's logged
- `scientific_fitness` — which model for which type of experiment

**Model Filtering:**
The notebook automatically excludes:
- Your CPT/SFT models (listed in `EXCLUDE_MODEL_NAMES`)
- Databricks foundation model suite — llama, gpt, claude, dbrx, etc. (matched via `EXCLUDE_PATTERNS`)
- Only user-created/trained models are analyzed

**Output:** Table saved to `{DEST_CATALOG}.{DEST_SCHEMA}.sft_mlflow_data`

---

### Phase 3: SFT Training

**Notebook:** `notebooks/train_sft_mistral.py`

SFT is like giving the model a quiz prep sheet after it's read all the textbooks (CPT). It learns to answer questions in a structured way using the Mistral `[INST]...[/INST]` format.

**Key details:**
- Loads the CPT model as a starting point (does NOT modify the CPT model)
- Applies fresh LoRA adapters for SFT
- Uses instruction masking: loss is only computed on response tokens, not question tokens
- Learning rate is 20x lower than CPT (1e-5 vs 2e-4) to preserve domain knowledge
- Optionally combines scientific Q&A + MLflow Q&A data

**Configuration (fill in the blanks):**
- `cpt_model_path`: path to your CPT model
- `catalog`, `schema`, `sft_table`: Unity Catalog location of Q&A data
- `output_dir`: where to save the SFT model
- `experiment_name`: MLflow experiment path
- `mlflow_table`: set to `"sft_mlflow_data"` when MLflow Q&A is ready

**Re-training:** SFT can be rerun as many times as needed. Each run loads the original CPT model fresh, so you never accumulate mistakes. You can also save multiple versions to different output paths.

**Output:** SFT model saved to DBFS

---

### Phase 4: Interactive Chat

**Notebook:** `notebooks/interactive_chat_demo.py`

Launches a Gradio chat interface to interact with the trained model. Ask it questions about your scientific data and see how it responds.

## Training Architecture

```
Base Mistral-7B
    │
    ▼ (CPT: next-token prediction on domain text)
CPT Model ──── saved to /dbfs/mnt/models/cpt_model_mistral
    │                    (untouched by SFT)
    ▼ (SFT: instruction tuning on Q&A pairs)
SFT Model ──── saved to /dbfs/mnt/models/sft_model_mistral
```

- **QLoRA:** 4-bit quantization + LoRA adapters (~1% of params trained)
- **V100 16GB** is sufficient for both CPT and SFT
- CPT and SFT models are saved separately — you always have both

## Key Technical Details

### Spark Type Detection
The pipeline detects column types using substring matching on Spark's type strings:
- **Numeric:** `double`, `float`, `integer`, `long`, `decimal`, `short`, `byte`
- **Categorical:** `string`, `varchar`, `char`, `text`, `boolean`, `date`, `timestamp`
- **Unknown types** default to categorical with a printed warning

### Instruction Format
SFT uses Mistral's instruction template:
```
<s>[INST] What is the average pH in the samples? [/INST] The average pH is 7.2 with a standard deviation of 0.3...</s>
```
The `[INST]` tokens are masked during training so the model only learns to generate responses.

### Known Fixes Applied
These were bugs encountered during development — documented in `notebooks/CHANGELOG_*.md`:
1. **Numeric type detection:** Exact match `DecimalType(38,18)` missed other decimal precisions. Fixed to substring matching.
2. **Categorical type detection:** Exact match `StringType` missed `VarcharType(255)`, `CharType(50)`, etc. Fixed to substring matching.
3. **datetime.date in string joins:** Date/timestamp columns treated as categorical had `datetime.date` objects that can't be joined as strings. Fixed by wrapping with `str()`.
4. **Subgroup NoneType formatting:** Groups with 1 sample have `None` for stddev. Fixed by extracting values with None guards before f-string formatting.
5. **MLflow fmt_metric None:** Metric values could be None. Added explicit None handling to return "N/A".

## How to Run (Step by Step)

1. **Set up cluster:** Create a Databricks GPU cluster with ML Runtime 14.x+ and a V100
2. **Upload notebooks:** Import all files from `notebooks/` into your Databricks workspace
3. **Run Phase 1:** Open `train_cpt_mistral.py`, fill in paths, run all cells (~13 hours)
4. **Run Phase 2a:** Open `generate_sft_data.py`, fill in catalog/schema, run all cells (~15-30 min)
5. **Run Phase 2b (optional):** Open `generate_sft_mlflow.py`, fill in config, run all cells (~5-10 min)
6. **Review Q&A data:** Check the preview output in Steps 18-19 of `generate_sft_data.py`
7. **Run Phase 3:** Open `train_sft_mistral.py`, fill in 6 blank paths, run all cells (~30 min - 2 hours)
8. **Test:** Run `interactive_chat_demo.py` to chat with your model

## Re-training and Iteration

| Scenario | What to Do |
|----------|-----------|
| Unhappy with Q&A quality | Edit `generate_sft_data.py` templates, rerun Phase 2a, then Phase 3 |
| Want to add MLflow Q&A | Run Phase 2b, set `mlflow_table` in Phase 3 config, rerun Phase 3 |
| Want to try different hyperparameters | Change config in Phase 3, rerun (loads fresh CPT each time) |
| Got new data in Unity Catalog | Rerun Phase 1 (CPT) from scratch, then Phase 2 + 3 |
| Want to keep multiple SFT versions | Change `output_dir` to a new path before each Phase 3 run |

## Local Development Files

These files support local development and testing outside Databricks:
- `data_prep.py` — Extract data from Databricks via SDK
- `train.py` — Local training script with MLflow logging
- `evaluate.py` — Compare base model vs CPT model
- `mlflow_query.py` — Query MLflow registry (has demo mode)
- `cpt_config.yaml` — Training hyperparameters for local runs
- `.env.example` — Template for Databricks connection credentials

## Hardware

| Phase | GPU | VRAM | Time |
|-------|-----|------|------|
| CPT Training | V100 | 16GB | ~13 hours |
| SFT Data Generation | None needed | CPU only | ~15-30 min |
| SFT Training | V100 | 16GB | ~30 min - 2 hours |
| Chat Demo | V100 | 16GB | Instant |
