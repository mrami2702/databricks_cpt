# Existing Assets — What We Have and How It Maps

This file catalogs everything already built and how each piece
slots into the closed-loop architecture. Nothing gets thrown away.

---

## Asset Inventory

### Agents (agents/agents/)

| Agent | Status | Role in Closed Loop |
|---|---|---|
| `root_agent.py` | Live | Coordinator — gains new routing rule for research sessions |
| `databricks_catalog_agent.py` | Live | Search Context Layer — validates columns, checks distributions |
| `databricks_job_agent.py` | Live | Execution backbone — runs notebooks, manages clusters |
| `automl_agent.py` | Live | **Inner Loop** — the experiment execution engine |
| `google_scholar_agent.py` | Live | Search Context Layer — literature grounding before experiments |
| `edx_data_discovery_agent.py` | Live | External data — surfaces joinable datasets for richer experiments |
| `ScientificAdvisorAgent` | Planned | Analysis Agent — interprets results with domain knowledge |

**All existing agents keep their current tools unchanged.**
The closed-loop system is additive — new agents and tools wrap around existing ones.

---

### Tools (agents/tools/)

| Tool File | Status | Closed-Loop Role |
|---|---|---|
| `catalog_tools.py` | Live | Column validation in Hypothesis Parser; distribution checks in context layer |
| `job_tools.py` | Live | Experiment execution; notebook dispatch from Loop Orchestrator |
| `eda_report_tools.py` | Live | Pre-experiment data profiling for context package |
| `automl_tools.py` | Live | Inner loop — called by AutoML Agent |
| `automl_report_tools.py` | Live | Feeds Analysis Agent with structured AutoML results |
| `scholar_tools.py` | Live | Literature context before each experiment iteration |
| `edx_tools.py` | Live | External dataset discovery when local data is insufficient |
| `hypothesis_tools.py` | **NEW** | NL → structured ExperimentConfig translation |
| `acquisition_tools.py` | **NEW** | "What to try next" — Bayesian + LLM hybrid |
| `memory_tools.py` | **NEW** | Scientific memory read/write |
| `analysis_tools.py` | **NEW** | Result interpretation → plain English findings |

---

### Clients (agents/clients/)

| Client | Status | Closed-Loop Role |
|---|---|---|
| `databricks_sql.py` | Live | Memory store queries (if Delta table); result fetching |
| `databricks_sdk_client.py` | Live | Job dispatch, MLflow API calls |
| `edx_client.py` | Live | External dataset search |
| `serper_client.py` | Live | Scholar search for context layer |

No new clients needed for MVP. The MLflow client will be added as part of
the ScientificAdvisorAgent build (already specced in TODO_scientific_advisor.md).

---

### Notebooks (notebooks/)

| Notebook | Status | Closed-Loop Role |
|---|---|---|
| `train_cpt_mistral.py` | Live | Produces domain-aware base model for Analysis Agent |
| `train_sft_mistral.py` | Live | Produces fine-tuned Mistral for domain interpretation |
| `generate_sft_data.py` | Live | Gold-standard Q&A data for training Analysis Agent |
| `generate_sft_mlflow.py` | Live | MLflow-aware Q&A data |
| `interactive_chat_demo.py` | Live | Prototype for conversational interface |

---

### Infrastructure

| Asset | Status | Closed-Loop Role |
|---|---|---|
| MLflow (Databricks workspace) | Live | Experiment tracking store — all AutoML runs logged here |
| Unity Catalog (`dev_europa.gold_roses`) | Live | Primary data source for experiments |
| V100 GPU cluster | Live | AutoML training runs |
| SQL Warehouse | Live | Memory store queries and data access |
| GCP Vertex AI | Live | LLM calls (Claude) for Hypothesis Parser, Acquisition Function |
| Mistral-7B SFT endpoint | Planned | Analysis Agent domain interpretation |

---

## Gap Analysis

What the closed loop needs that doesn't exist yet:

### New Agents
1. **LoopOrchestratorAgent** — state machine managing the outer loop
   - File: `agents/agents/loop_agent.py`
   - Calls: HypothesisParser, SearchContextLayer, AutoMLAgent, AnalysisAgent, AcquisitionFunction

### New Tool Modules
2. **hypothesis_tools.py** — NL → ExperimentConfig
   - Key function: `parse_hypothesis(user_text, available_columns) → ExperimentConfig`

3. **acquisition_tools.py** — "What to try next?"
   - Key function: `propose_next_experiment(run_history, memory, literature_context) → Proposal`

4. **memory_tools.py** — Scientific memory read/write
   - Key functions: `add_finding()`, `get_relevant_findings()`, `get_dead_ends()`

5. **analysis_tools.py** — Result interpretation
   - Key function: `interpret_automl_results(results, hypothesis, context) → Finding`

### New Data Store
6. **Scientific Memory table** in Unity Catalog
   - Table: `dev_europa.gold_roses.research_findings` (or separate schema)
   - Schema: finding_text, hypothesis, confidence, run_ids, created_at, tags

### Orchestration Layer
7. **Loop state management** — conversation-persistent state across turns
   - Extends current `InMemorySessionService` or uses Delta table for persistence

---

## What Changes in Existing Files

| File | Change Required |
|---|---|
| `agents/agents/root_agent.py` | Add LoopOrchestratorAgent as 6th sub-agent; add routing rule for research sessions |
| `agents/main.py` | Minor: pass loop state context through conversation turns |
| `agents/config.py` | Add: MEMORY_TABLE env var, LOOP_EXPLORE_RATIO env var |
| `agents/tools/__init__.py` | Import new tool modules |
| `agents/agents/__init__.py` | Export LoopOrchestratorAgent |

All other existing files: **no changes.**

---

## Reuse Highlights

These existing capabilities can be directly reused with no modification:

**DatabricksCatalogAgent.profile_dataset()** → perfect for pre-experiment data validation
in the context layer. Already returns distributions, null rates, type detection.

**AutoMLAgent** → drop-in inner loop. ExperimentConfig just needs to map to its
existing input format (table, target, features, problem_type).

**GoogleScholarAgent.search_papers()** → direct call with hypothesis terms for
literature grounding before each iteration.

**generate_eda_report()** → surfaces data quality issues before experiment runs,
can flag problems back to user before wasting compute.

**MLflow run tracking** → AutoML Agent already logs to MLflow. Analysis Agent
queries these logs. The infrastructure is already wired.

**Fine-tuned Mistral-7B** → once deployed, becomes the domain-grounded core of
the Analysis Agent, replacing general LLM for domain-specific interpretation.
