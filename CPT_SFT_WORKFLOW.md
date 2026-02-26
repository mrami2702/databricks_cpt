# CPT + SFT Training Workflow

## Overview

This pipeline trains a domain-specific AI assistant in two phases using scientific data from Databricks Unity Catalog. The end result is a model that understands your experimental data AND can answer questions about it in natural language.

---

## The Problem

A base LLM (like Mistral-7B) knows nothing about your specific scientific data — your tables, columns, measurements, or relationships. If you ask it "What was the analytical response for sample S-001?" it will hallucinate an answer.

We fix this in two steps:

1. **CPT** — teach it your domain knowledge (read the textbook)
2. **SFT** — teach it to answer questions (do practice exams)

---

## Phase 1: Continual Pretraining (CPT)

### What It Does
Takes your tabular scientific data from Unity Catalog and teaches the model your domain vocabulary, data patterns, and table relationships.

### How It Works
- Reads all tables from `dev_europa.gold_roses`
- Converts tabular rows into natural language passages
- Model learns by predicting the next word in those passages (causal language modeling)
- No question/answer pairs — just raw text

### Data Format (input to model)
```
Record from Gold Roses (gold_roses): Identification — sample id: S-001.
Measurements — temperature: 312.5; pressure: 1.2; analytical response: 0.847.
Properties — material type: oxide.
```

### What the Model Learns
- Column names and what they mean
- Typical value ranges for each measurement
- Relationships between tables (shared columns)
- Domain-specific vocabulary
- How different properties relate to each other

### What the Model Does NOT Learn
- How to answer questions
- How to follow instructions
- How to format structured responses

### Training Details
- Base model: `mistralai/Mistral-7B-v0.3`
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Trainable parameters: ~1% of total
- Hardware: V100 GPU (16GB)
- Batch size: 1 (gradient accumulation: 16, effective batch: 16)
- Output: `/dbfs/mnt/models/cpt_model_mistral`

### Notebook
`notebooks/train_cpt_mistral.py` — includes data preparation and training in one file.

### After CPT, the model can:
- Generate text that sounds like your domain
- Recall column names and table structures
- Complete sentences about your data patterns

### After CPT, the model cannot:
- Answer direct questions reliably
- Provide structured recommendations
- Follow specific instructions

---

## Phase 2: SFT Data Generation

### What It Does
Takes the same source data from Unity Catalog and generates question-answer pairs at multiple levels of complexity.

### How It Works
- Reads all tables from `dev_europa.gold_roses`
- Computes statistics (min, max, avg, counts, distributions)
- Auto-generates Q&A pairs using templates
- Saves to Unity Catalog for review before training

### Q&A Levels

#### Level 1: Row-Level Questions
Questions about specific records. Answers are direct lookups.
```
Q: What was the analytical response for sample S-001?
A: Sample S-001 had an analytical response of 0.847, measured at a
   temperature of 312.5 and pressure of 1.2. It is an oxide material.
```

#### Level 2: Aggregation Questions
Questions that require computing across multiple rows.
```
Q: What is the average temperature for oxide samples?
A: Oxide samples have an average temperature of 305.2 across 47
   measurements, ranging from 198.4 to 412.1.
```

#### Level 3: Comparison Questions
Questions that compare groups or categories.
```
Q: How does analytical response differ between oxide and sulfide materials?
A: Oxide materials show a higher mean analytical response (0.82) compared
   to sulfide materials (0.61). This difference is consistent across
   temperature ranges.
```

#### Level 4: Schema-Level Questions
Questions about data structure and table relationships.
```
Q: What tables are available in the gold_roses schema?
A: The gold_roses schema contains 5 tables: sample_properties,
   analytical_results, ... They share the sample_id column which
   links samples to their measurements.
```

#### Level 5: Reasoning Questions
Questions that require domain judgment and recommendation.
```
Q: Which model should I deploy for predicting analytical response?
A: I recommend the neural network model (v2) because it achieves the
   lowest RMSE (0.0198) and highest R2 (0.9521). While inference latency
   is higher at 45.2ms, the accuracy improvement justifies the trade-off.
```

#### Level 6: Data Quality & QC
Outlier detection, null rates, and sanity checks — all computed from the data.
```
Q: Are there outliers in temperature in Gold Roses?
A: Using the IQR method (1.5x interquartile range), there are 12
   potential outliers in temperature in the gold_roses table.
   Q1=245.3, Q3=387.1, IQR=141.8. Values below 32.6 or above
   599.8 are flagged. This represents 0.8% of 1500 records.
```

#### Level 7: Data Transformation & Schema Documentation
Column definitions with observed ranges, skewness detection, and unit consistency across tables.
```
Q: Create a schema definition for the Gold Roses table with column meanings and ranges.
A: Schema definition for gold_roses (1500 records):
   - sample id (StringType)
   - temperature (DoubleType): range [198.4 to 412.1], mean 305.2
   - pressure (DoubleType): range [0.1 to 2.8], mean 1.15
   - analytical response (DoubleType): range [0.12 to 0.97], mean 0.72

Q: Is temperature measured consistently across tables?
A: Warning: temperature shows very different ranges across tables,
   which may indicate different units or scales...
```

#### Level 8: Uncertainty & Precision
Confidence intervals, coefficient of variation, and measurement precision.
```
Q: How precise are the temperature measurements in Gold Roses?
A: The temperature measurements in gold_roses have moderate precision
   with a coefficient of variation (CV) of 0.1842 (18.4%). Mean: 305.2,
   standard deviation: 56.24, based on 1500 measurements.

Q: What is the 95% confidence interval for the mean temperature?
A: The 95% CI for mean temperature in gold_roses is [302.4, 308.1],
   based on 1500 measurements with standard error 1.45.
```

#### Level 9: Metadata Completeness & Reproducibility
What metadata exists, what is missing, and documentation checklists.
```
Q: What metadata is missing from Gold Roses for reproducibility?
A: Missing metadata that would improve reproducibility: timestamp or
   date of measurement, operator identifier, instrument identifier.
   Consider adding calibration references and SOP versions.
```

#### Level 10: Data Summaries
Concise factual summaries for lab meetings or quick overviews.
```
Q: Give me a quick summary of the gold_roses data for a lab meeting.
A: The gold_roses schema contains 32 tables with 45,000 total records.
   There are 87 numeric measurement columns and 23 categorical columns.
   The largest table is analytical_results (12,000 records)...
```

### Review Step
After generation, the notebook displays a summary and samples so you can verify:
- Questions make sense for your domain
- Answers are factually correct (computed from real data, not made up)
- Format and phrasing look right

If anything is off, tweak templates and re-run before training.

### Output
Saved as `dev_europa.gold_roses.sft_training_data` with columns:
- `instruction` — the question
- `response` — the answer
- `category` — which level (row_level, aggregation, comparison, schema, reasoning, data_quality, schema_documentation, uncertainty, reproducibility, summary)

### Notebook
`notebooks/generate_sft_data.py`

---

## Phase 3: Supervised Fine-Tuning (SFT)

### What It Does
Takes the CPT model (which already knows your domain) and teaches it to answer questions in a structured way.

### How It Works
- Loads the CPT model from Phase 1 as the starting point
- Applies a fresh LoRA adapter on top
- Trains on the Q&A pairs from Phase 2
- Uses Mistral's instruction format with special tokens

### Data Format (input to model)
```
<s>[INST] What was the analytical response for sample S-001? [/INST]
Sample S-001 had an analytical response of 0.847, measured at a
temperature of 312.5 and pressure of 1.2. It is an oxide material.</s>
```

### Key Differences from CPT

| | CPT (Phase 1) | SFT (Phase 3) |
|---|---|---|
| Starting model | Base Mistral-7B | CPT-trained model |
| Data format | Raw text passages | [INST] question [/INST] answer pairs |
| What it learns | Domain vocabulary and patterns | How to answer questions |
| Loss computed on | Every token | Only response tokens (question is masked) |
| Learning rate | Higher (~2e-4) | Lower (~1e-5) to preserve CPT knowledge |
| Training steps | More | Fewer |

### Why Lower Learning Rate?
The CPT model already has domain knowledge baked in. A high learning rate would overwrite that knowledge. SFT uses a gentler learning rate to teach the Q&A format while preserving what was learned in CPT.

### Why Mask the Question?
During SFT, we only compute loss on the response tokens. The model already knows how to read questions — we want it to learn how to generate good answers. Masking the instruction portion focuses the training signal on response quality.

### Training Details
- Starting model: `/dbfs/mnt/models/cpt_model_mistral` (output from Phase 1)
- Method: QLoRA (same as Phase 1)
- Learning rate: ~1e-5 (10-20x lower than CPT)
- Fewer training steps (the model is already close to what we want)
- Output: `/dbfs/mnt/models/sft_model_mistral`

### Notebook
`notebooks/train_sft_mistral.py` (to be created)

---

## Full Pipeline Summary

```
Unity Catalog (dev_europa.gold_roses)
    |
    |  All tables: tabular scientific data
    |
    v
+-------------------------------------------+
|  Phase 1: CPT Training                    |
|  notebooks/train_cpt_mistral.py           |
|                                           |
|  1. Discover tables in schema             |
|  2. Convert rows to natural language      |
|  3. Train Mistral-7B with QLoRA           |
|                                           |
|  Output: /dbfs/mnt/models/cpt_model_mistral
+-------------------------------------------+
    |
    |  Model now understands your domain
    |
    v
+-------------------------------------------+
|  Phase 2: Generate SFT Data               |
|  notebooks/generate_sft_data.py           |
|                                           |
|  1. Read same tables                      |
|  2. Compute statistics                    |
|  3. Generate Q&A pairs (5 levels)         |
|  4. Save to Unity Catalog for review      |
|                                           |
|  Output: dev_europa.gold_roses.sft_training_data
+-------------------------------------------+
    |
    |  You review Q&A pairs, confirm quality
    |
    v
+-------------------------------------------+
|  Phase 3: SFT Training                    |
|  notebooks/train_sft_mistral.py           |
|                                           |
|  1. Load CPT model from Phase 1           |
|  2. Train on Q&A pairs from Phase 2       |
|  3. Lower learning rate to preserve       |
|     domain knowledge                      |
|                                           |
|  Output: /dbfs/mnt/models/sft_model_mistral
+-------------------------------------------+
    |
    |  Final model: knows domain + answers questions
    |
    v
+-------------------------------------------+
|  Inference                                |
|  notebooks/recommend.py                   |
|  run_recommendation.py (local)            |
|                                           |
|  User asks natural language questions     |
|  Model responds with domain-aware answers |
+-------------------------------------------+
```

---

## What You Run (Step by Step)

1. **Run `train_cpt_mistral.py` on Databricks GPU cluster**
   - Copy notebook contents into Databricks
   - Attach to GPU cluster (V100)
   - Run all cells
   - Wait for training to complete
   - Verify model saved to DBFS

2. **Run `generate_sft_data.py` on Databricks**
   - Can run on any cluster (no GPU needed for this step)
   - Review the generated Q&A pairs
   - Re-run with adjustments if needed

3. **Run `train_sft_mistral.py` on Databricks GPU cluster**
   - Same GPU cluster as step 1
   - Loads CPT model, trains on Q&A pairs
   - Produces final model

4. **Test with `recommend.py` or `run_recommendation.py`**
   - Ask questions in natural language
   - Get domain-aware answers

---

## Current Status

| Component | Status |
|---|---|
| train_cpt_mistral.py | Created, ready to run |
| generate_sft_data.py | Not yet created |
| train_sft_mistral.py | Not yet created |
| recommend.py (Databricks) | Created, needs model path update |
| run_recommendation.py (local) | Created, needs model path update |

---

## Notes

- Both training phases use QLoRA on V100 GPU (16GB)
- All data comes from the same Unity Catalog source
- MLflow tracks both training runs automatically on Databricks
- The CPT model is a prerequisite for SFT — you cannot skip Phase 1
- Phase 2 (data generation) does not require a GPU
- After SFT, update recommend.py to point at `/dbfs/mnt/models/sft_model_mistral`
