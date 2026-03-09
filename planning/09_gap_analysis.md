# Gap Analysis — What We Have vs What We Need

Honest inventory of every component in the closed-loop system.
Maps directly to the 9-step technical architecture in 01_architecture.md.

---

## Step 1 — User Input & Routing

| Component | Status | Notes |
|---|---|---|
| `agents/main.py` REPL | **HAVE** | Entry point works |
| Root agent | **HAVE** | Routes to sub-agents |
| Research session routing rule | **NEED** | One new rule in root_agent.py |
| LoopOrchestratorAgent to route TO | **NEED** | Doesn't exist yet |

---

## Step 2 — Hypothesis Parsing

| Component | Status | Notes |
|---|---|---|
| Column validation (catalog schema) | **HAVE** | `DatabricksCatalogAgent.get_dataset_schema()` |
| NL → ExperimentConfig translator | **NEED** | `hypothesis_tools.py` — entirely new |
| Column fuzzy matching | **NEED** | "temperature" → "temp_c" resolution logic |
| ExperimentConfig dataclass | **NEED** | The structured object itself |

---

## Step 3 — Context Loading

| Component | Status | Notes |
|---|---|---|
| Literature search | **HAVE** | `GoogleScholarAgent.search_papers()` |
| Data profiling | **HAVE** | `DatabricksCatalogAgent.profile_dataset()` |
| Memory retrieval | **NEED** | `memory_tools.get_relevant_findings()` |
| Research findings Delta table | **NEED** | DDL + creation in Unity Catalog |
| Parallel dispatch | **NEED** | Fire all 3 context calls simultaneously |
| ContextPackage assembler | **NEED** | Combines the 3 outputs into one object |

---

## Step 4 — AutoML Execution

| Component | Status | Notes |
|---|---|---|
| AutoMLAgent | **HAVE** | Runs experiments |
| `automl_tools.py` | **HAVE** | Tool implementations |
| `automl_report_tools.py` | **HAVE** | Parses results from MLflow |
| MLflow logging | **HAVE** | Already integrated |
| ExperimentConfig → AutoML translation | **NEED** | Small bridge function |
| MLflow loop session tagging | **NEED** | Tag runs with iteration + session ID |
| AutoMLResult dataclass | **NEED** | Structured return object |

**Verdict: AutoML is the closest to done. Mostly needs connecting, not building.**

---

## Step 5 — Analysis

| Component | Status | Notes |
|---|---|---|
| `automl_report_tools.py` | **HAVE (partial)** | Reads MLflow, formats output — but no interpretation |
| Plain English interpretation | **NEED** | `analysis_tools.py` — LLM call on results |
| Quality checks | **NEED** | Leakage detection, weak signal flags, anomaly detection |
| Finding dataclass | **NEED** | Structured interpretation object |
| Memory write after analysis | **NEED** | `memory_tools.add_finding()` call |
| Fine-tuned Mistral interpreter | **NEED (Phase 3)** | Endpoint not deployed yet — Claude fallback for now |

---

## Step 6 — Acquisition Function

| Component | Status | Notes |
|---|---|---|
| Anything for this | **NOTHING** | Entirely new territory |
| Phase 1: LLM-based proposal | **NEED** | `acquisition_tools.py` — Claude prompt over run history |
| Phase 2: scikit-optimize | **NEED** | New dependency + GP surrogate integration |
| Domain constraint enforcement | **NEED** | Hard/soft rules from CM2US domain priors |
| Proposal dataclass | **NEED** | Top-N candidates + justifications |

---

## Step 7 — User Gate

| Component | Status | Notes |
|---|---|---|
| Basic REPL | **HAVE** | `main.py` handles input/output |
| Iteration report renderer | **NEED** | Formatted output with findings + proposals |
| Research session command parser | **NEED** | "yes", "use #2", "stop", "show me 5" |
| Loop state across turns | **NEED** | Session persists between user messages |

---

## Step 8 — Loop State Machine

| Component | Status | Notes |
|---|---|---|
| Anything for this | **NOTHING** | Entirely new |
| `loop_agent.py` state machine | **NEED** | IDLE → PARSING → RUNNING → PROPOSING → AWAITING |
| LoopState / LoopIteration dataclasses | **NEED** | Track full iteration history |
| Session management | **NEED** | Persist state across conversation turns |

---

## Step 9 — Report Generation

| Component | Status | Notes |
|---|---|---|
| Anything for this | **NOTHING** | Entirely new |
| `report_tools.py` | **NEED** | Markdown + notebook generator |
| Markdown summary report | **NEED** | Shareable research summary (.md) |
| Databricks notebook generator | **NEED** | Reproducible .py notebook with all results |
| Session summary NL generator | **NEED** | Plain English full session wrap-up |

---

## Infrastructure

| Component | Status | Notes |
|---|---|---|
| Unity Catalog (`dev_europa.gold_roses`) | **HAVE** | 32 tables, data access working |
| MLflow | **HAVE** | Experiment tracking live |
| Claude via GCP Vertex AI | **HAVE** | LLM calls working |
| Databricks SQL connector | **HAVE** | Data access working |
| `research_findings` Delta table | **NEED** | DDL + create in Unity Catalog |
| `research_sessions` Delta table | **NEED** | DDL + create in Unity Catalog |
| `research_dead_ends` Delta table | **NEED** | DDL + create in Unity Catalog |
| Memory seeds (literature priors) | **NEED** | Pre-populate with CM2US known facts |
| `scikit-optimize` package | **NEED (Phase 2)** | Add to `requirements_agents.txt` |

---

## Summary View

```
WHAT WE HAVE (the raw ingredients):
────────────────────────────────────────────────────────────────
✓  All sub-agents (Catalog, Job, AutoML, Scholar, EDX)
✓  All data infrastructure (Unity Catalog, MLflow, SQL)
✓  LLM infrastructure (Claude via Vertex AI)
✓  Entry point + basic routing (main.py, root_agent.py)
✓  Data profiling + schema tools
✓  Literature search
✓  AutoML runs + MLflow logging
✓  automl_report_tools (partial analysis — reads results, no interpretation)


WHAT WE NEED TO BUILD (the recipe):
────────────────────────────────────────────────────────────────
NEW FILES (6):
  agents/agents/loop_agent.py        ← state machine, the backbone
  agents/tools/hypothesis_tools.py   ← NL → ExperimentConfig
  agents/tools/analysis_tools.py     ← results → plain English Finding
  agents/tools/acquisition_tools.py  ← propose next experiment (LLM → BO hybrid)
  agents/tools/memory_tools.py       ← scientific memory read/write
  agents/tools/report_tools.py       ← markdown report + Databricks notebook output

MODIFIED FILES (4):
  agents/agents/root_agent.py        ← add loop routing rule + LoopOrchestratorAgent
  agents/config.py                   ← add MEMORY_TABLE, SESSION_TABLE, EXPLORE_RATIO
  agents/tools/__init__.py           ← import new tool modules
  agents/requirements_agents.txt     ← add scikit-optimize (Phase 2)

NEW INFRASTRUCTURE (3 tables + seeds):
  dev_europa.gold_roses.research_findings    ← scientific memory
  dev_europa.gold_roses.research_sessions    ← session tracking
  dev_europa.gold_roses.research_dead_ends   ← known bad regions
  Memory seed data (CM2US literature priors baked in from Day 1)
```

---

## What Each New File Does (One Line Each)

| File | What it does |
|---|---|
| `loop_agent.py` | State machine that runs the outer research loop and coordinates all other components |
| `hypothesis_tools.py` | Translates user NL input into a validated, structured ExperimentConfig |
| `analysis_tools.py` | Interprets AutoML results in plain English and writes findings to memory |
| `acquisition_tools.py` | Proposes the next experiment to run using LLM reasoning (Phase 1) + Bayesian optimization (Phase 2) |
| `memory_tools.py` | Reads and writes scientific findings to Delta table — the system's persistent knowledge |
| `report_tools.py` | Generates shareable Markdown summary report and reproducible Databricks notebook at session end |

---

## Build Order (dependency sequence)

```
1. Dataclasses + config vars        no deps — define data shapes first
        ↓
2. memory_tools.py                  no deps — pure storage layer
        ↓
3. hypothesis_tools.py              deps: catalog tools for column validation
        ↓
4. analysis_tools.py                deps: automl_report_tools for result parsing
        ↓
5. acquisition_tools.py             deps: memory_tools + analysis output shapes
        ↓
6. report_tools.py                  deps: all iteration data shapes defined
        ↓
7. loop_agent.py                    deps: ALL tools above + existing sub-agents
        ↓
8. root_agent.py update             deps: loop_agent.py exists
        ↓
9. Delta tables                     can be created any time before first run
        ↓
10. Memory seeds                    after Delta tables exist
```

---

## Difficulty Assessment

| Component | Difficulty | Why |
|---|---|---|
| `memory_tools.py` | Low | Read/write Delta table — straightforward SQL |
| `report_tools.py` | Low | String formatting + template rendering |
| `root_agent.py` update | Low | One new routing rule + one new AgentTool import |
| `analysis_tools.py` | Medium | LLM prompt engineering + quality check logic |
| `hypothesis_tools.py` | Medium-High | Fuzzy column matching + LLM structured output + validation |
| `acquisition_tools.py` | Medium-High | Phase 1 is simple; Phase 2 requires MLflow model extraction + scikit-optimize |
| `loop_agent.py` | High | State machine across conversation turns in ADK is the trickiest engineering piece |

---

## The Acquisition Function Clarified

The acquisition function (`acquisition_tools.py`) is NOT AutoML.
They play different roles:

```
AutoML (HAVE):
  Takes historical experiment data
  Trains the best ML model to PREDICT outcomes from conditions
  Returns: model + feature importances
  Answers: "which variables drive recovery?"

Acquisition Function (NEED):
  Takes the AutoML-trained model
  Mathematically searches the parameter space for the highest-value next point
  Returns: specific conditions to physically test next
  Answers: "what exact conditions should I run in the lab next?"

Together:
  AutoML builds the map.
  Acquisition function finds the best destination on that map.
  Researcher physically travels there (runs the experiment).
  New result updates the map. Loop repeats.
```

**Phase 1 implementation**: Pure LLM — Claude reasons over run history and proposes next conditions (~40 lines of Python).

**Phase 2 implementation**: scikit-optimize generates mathematically optimal candidates via Gaussian Process + Expected Improvement, then Claude filters using domain knowledge (~100 lines of Python).

The researcher never sees this distinction. They just receive:
```
"Based on 3 experiments, I recommend testing HCl=3.2M, Temp=67°C next.
 Predicted Nd recovery: 85%. This region has the highest expected improvement
 over your current best (83%). Proceed?"
```

---

## Report Output Spec

Two artifacts generated at session end (Step 9):

### A — Markdown Summary Report
```
agents/output/session_[id]_report.md

Contents:
  Executive Summary       (2-3 paragraphs, plain English)
  Research Hypothesis     (original NL + parsed config)
  Method                  (dataset, parameters varied, metrics)
  Iteration Log           (table: conditions → result → finding per run)
  Key Findings            (bulleted, plain English)
  Optimal Conditions      (best parameter set found)
  Recommended Next Steps  (what to investigate next)
  Supporting Data         (MLflow run IDs, literature references)
```

### B — Databricks Notebook
```
agents/output/session_[id]_notebook.py

Contents:
  Markdown cells with hypothesis, method, findings
  Code cells to reload the data used
  Code cells to reproduce AutoML results from MLflow run IDs
  Code cells for feature importance plots
  Code cells to query optimal conditions from best model
  Final markdown cell with recommendations

Format: Databricks .py notebook
        (# Databricks notebook source + # COMMAND ---------- markers)
        Ready to upload and run in Databricks workspace
```

The notebook is fully reproducible — anyone who receives it can open it
in Databricks, run all cells, and arrive at the same findings independently.
