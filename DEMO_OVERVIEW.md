# CPT Pipeline Demo Overview

## The Problem

You have ML models in production (XGBoost, neural nets, random forests, etc.) tracked in MLflow. When someone asks "which model should I deploy for X?" — today that requires a senior data scientist to manually review metrics, compare models, and apply domain-specific judgment.

## The Solution

We train an LLM on our domain-specific scientific knowledge so it can do that reasoning automatically. A user asks a question in plain English, and the system responds with a recommendation backed by real metrics and domain expertise.

## How It Works (3 Parts)

### Part 1: Training the LLM (Databricks GPU)

We take a base LLM (Microsoft Phi-3, 3.8B parameters) and teach it our domain knowledge through Continual Pretraining (CPT).

**Where does the knowledge come from?**
All the text data in our Unity Catalog — specifically everything in `dev_europa.gold_roses`. This includes scientific documents, evaluation criteria, model selection guidelines, and domain expertise that we want the LLM to learn.

**How do we train efficiently?**
We use QLoRA — a technique that only trains 1% of the model's parameters while keeping the rest frozen. This means:
- We can train on a single V100 GPU (16GB) instead of needing a massive cluster
- Training takes hours, not days
- The model keeps its general language abilities while gaining our domain expertise

**What happens during training?**
1. The notebook reads all tables from `dev_europa.gold_roses`
2. Auto-detects text columns in each table
3. Combines everything into one training dataset
4. Fine-tunes the LLM with a very low learning rate (to avoid "forgetting" general knowledge)
5. Saves the trained model to DBFS and logs everything to MLflow

### Part 2: Getting a Recommendation (The Core Value)

After training, we have an LLM that understands our domain. When we want a model recommendation:

1. **Pull model metrics from MLflow** — the system queries MLflow Model Registry and gets metrics for all candidate models (accuracy, latency, F1 scores, etc.)

2. **Build a prompt** — the metrics get formatted into a structured prompt along with the user's question ("Which model should I deploy for neutron flux prediction?")

3. **Ask the trained LLM** — the prompt goes to our CPT-trained model, which now has domain-specific reasoning ability

4. **Get a natural language response** — the LLM responds with a recommendation, justification referencing specific metrics, acceptance criteria, and risk considerations

**Example interaction:**

> **User:** Which model should I deploy for rare earth mineral classification?
>
> **AI:** Recommendation: rare_earth_extraction_rf_v3
>
> This model achieves 94.2% classification accuracy for lithium-bearing pegmatite identification with an F1 score of 0.91 on minority deposit types. The feature importance aligns with USGS critical minerals criteria, correctly weighting geochemical signatures (Li/Cs ratios, Ta concentrations) over lithological proxies.
>
> Why not the neural network? While the NN scores slightly higher on accuracy (95.1%), its inference latency of 340ms exceeds the 100ms requirement for real-time field analysis. The random forest delivers comparable accuracy at 12ms latency.

The key difference from a generic LLM: our model doesn't just compare numbers — it understands **why** certain metrics matter for our specific scientific use cases.

### Part 3: The App (Future)

The trained model gets deployed as a Databricks serving endpoint (REST API). A standalone app with a chat interface calls that endpoint. Users ask questions in natural language, the app pulls model metrics from MLflow, sends everything to the serving endpoint, and displays the recommendation.

```
User types question
    → App pulls ML model metrics from MLflow
    → Builds prompt with metrics + question
    → Calls Databricks serving endpoint (our trained LLM)
    → Displays natural language recommendation
```

## What Makes This Different From Just Using ChatGPT?

A generic LLM can compare numbers, but it doesn't know:
- Our specific acceptance criteria for production deployment
- Why certain metrics matter more for our scientific use cases
- Our model selection best practices and historical decisions
- Domain-specific tradeoffs (e.g., accuracy vs latency for field analysis)

Our CPT model learned all of this from the training data. It reasons like a domain expert, not a generalist.

## The Files

| File | What It Does | Where It Runs |
|------|-------------|---------------|
| `notebooks/train_cpt.py` | Trains the LLM on our domain knowledge | Databricks GPU notebook |
| `train.py` | Same training logic, runnable as a job | Databricks job (alternative) |
| `mlflow_query.py` | Queries MLflow + calls the trained LLM for recommendations | Databricks / becomes app backend |
| `evaluate.py` | Tests if training actually improved the model | Databricks (optional) |
| `cpt_config.yaml` | All training settings (model, learning rate, data source) | Config file |

## The Pipeline

```
STEP 1 — TRAIN (run once or periodically)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unity Catalog              Databricks GPU            DBFS + MLflow
dev_europa.gold_roses  →   QLoRA fine-tuning    →   Trained CPT model
(all tables, all text)     (V100, ~2-3 hours)       (saved + logged)


STEP 2 — TEST (verify it works)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Load trained model → Ask it domain questions → Confirm domain-aware responses


STEP 3 — RECOMMEND (the actual use case)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLflow Registry          Trained LLM              Natural Language
(model metrics)    →     (domain reasoning)   →   Recommendation
"rmse: 0.02,             "Given safety             "Use Model B because
 r2: 0.95..."            requirements..."          it meets criteria X, Y, Z"


STEP 4 — APP (future)
━━━━━━━━━━━━━━━━━━━━━━
Serving Endpoint → Standalone App → User chats in natural language
```

## Key Technical Details (If Asked)

- **Base model:** Microsoft Phi-3 Mini (3.8B parameters)
- **Training method:** QLoRA — 4-bit quantization + LoRA adapters, only 1.04% of parameters trained
- **Data source:** All tables in `dev_europa.gold_roses` (auto-detected text columns)
- **Cluster:** Databricks Runtime ML 14.x+, 1x V100 GPU (16GB)
- **Tracking:** MLflow auto-configured on Databricks, all metrics and model artifacts logged
- **Training prevents catastrophic forgetting:** Low learning rate (2e-5) + cosine decay + optional replay data mixing
