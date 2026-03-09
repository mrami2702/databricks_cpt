# Master Handoff Document — Closed-Loop Autonomous Research System
## CM2US Critical Minerals Use Case

**Purpose**: This document is the single source of truth for an agent picking up this
project. It synthesizes all planning discussions, architectural decisions, conceptual
clarifications, and the immediate build plan. Read this first. Reference the numbered
planning files for deeper detail on each topic.

---

## 1. What We Are Building and Why

### The Project
A closed-loop autonomous research system that allows a scientist to state a hypothesis
in plain English and receive interpreted experimental results, with the system
proposing the next experiment to run — all conversationally, without touching
dashboards, configs, or SQL.

### The Strategic Context (CM2US)
This is being built for the **CM2US (Critical Minerals and Materials to Unlock Supply)**
use case — a DOE initiative led by 12 national laboratories (Ames Lab as lead) under
the Genesis Mission. The US is nearly 100% import-dependent on many critical minerals,
and China controls ~80% of global REE processing. CM2US is the national response.

The specific research problem: optimizing **rare earth element (REE) recovery** from
coal-based feedstocks (fly ash, acid mine drainage, coal refuse). Researchers run
leaching experiments varying parameters like acid concentration, temperature, and
leaching time to maximize REE recovery yield. This is exactly the kind of
multi-parameter optimization problem a closed-loop system accelerates.

### The Core Value Proposition
```
BEFORE: Researcher manually configures experiments, reads dashboards,
        decides next steps from intuition. Weeks between iterations.

AFTER:  Researcher states hypothesis in one sentence. System runs
        AutoML, interprets results in plain English, proposes the
        next experiment to run. Hours between iterations.
        Each run is the most informative experiment possible.
```

---

## 2. The Architecture — Two Nested Loops

The system has two nested loops operating at different levels:

```
╔══════════════════════════════════════════════════════════════════════╗
║  OUTER LOOP — The Science Loop (what we're building)                 ║
║                                                                      ║
║  User NL hypothesis                                                  ║
║       ↓                                                              ║
║  [Step 2]  Hypothesis Parser     → ExperimentConfig                 ║
║       ↓                                                              ║
║  [Step 3]  Context Loading       → ContextPackage                   ║
║            (literature + memory + data profile — in parallel)        ║
║       ↓                                                              ║
║  [Step 4]  AutoML Execution      → Surrogate Model + Importances    ║
║       ↓                                                              ║
║  [Step 5]  Analysis Agent        → Plain English Finding            ║
║       ↓                                                              ║
║  [Step 6]  Acquisition Function  → Proposed Next Conditions         ║
║       ↓                                                              ║
║  [Step 7]  User Gate             → approve / redirect / stop        ║
║       ↓                                                              ║
║  [Step 8]  Loop Decision         → repeat or generate report        ║
║       ↓                                                              ║
║  [Step 9]  Report Generation     → Markdown + Databricks Notebook   ║
╚══════════════════════════════════════════════════════════════════════╝
```

**The inner loop** (AutoML) runs inside Step 4 — it tries multiple ML algorithms
on the historical data, picks the best, and returns a trained surrogate model.
It does NOT run physical experiments. It trains a model that PREDICTS outcomes.

**The outer loop** runs the full sequence above, with each iteration representing
one physical experiment conducted by the researcher in the lab.

---

## 3. Critical Conceptual Clarifications

These distinctions were worked out carefully in planning discussions.
Getting them right is essential to building the system correctly.

### AutoML vs Acquisition Function — NOT the same thing

This is the most important distinction in the whole system:

```
AutoML (ALREADY BUILT — agents/agents/automl_agent.py):
  Input:   Historical experiment data table (N rows of past results)
  Process: Tries Random Forest, XGBoost, LightGBM, etc. on that data
  Output:  Best trained ML model + feature importances + R² score
  Answers: "Which variables drive the outcome? How well can we predict it?"
  Role:    SURROGATE MODEL BUILDER

Acquisition Function (TO BE BUILT — agents/tools/acquisition_tools.py):
  Input:   The trained surrogate model from AutoML
  Process: Queries that model for hundreds of thousands of candidate
           parameter combinations, scores each by Expected Improvement
  Output:  The specific parameter conditions to physically test next
  Answers: "What exact conditions should the researcher run in the lab?"
  Role:    PARAMETER SPACE SEARCHER

Together:
  AutoML builds the map of the parameter space.
  The acquisition function finds the best destination on that map.
  The researcher physically travels there (runs the experiment).
  New result updates the map. Loop repeats.
```

### The Surrogate Model Concept

A surrogate model is a cheap mathematical copy of an expensive real-world process.
Instead of running a 3-hour lab experiment to find out what 100,000 parameter
combinations produce, you query the surrogate model — which answers in milliseconds.

AutoML's job is to train the best possible surrogate model from whatever historical
data exists. As more experiments are run and added to the dataset, AutoML retrains
and the surrogate improves.

### The Acquisition Function Mechanics

The acquisition function balances exploitation (refine what's working) vs exploration
(investigate unknown territory). It uses Expected Improvement (EI):

```
EI = (How much better than current best) × (How confident the model is)

High EI candidates:
  → Model predicts improvement AND is confident (exploit)
  → Model is very uncertain (explore — might be hiding something great)

Low EI candidates:
  → Model predicts worse outcome AND is confident (skip)
```

**Implementation**: Phase 1 is a pure LLM prompt (~40 lines of Python). Claude
reasons over run history and proposes next conditions. Phase 2 adds scikit-optimize
for mathematical rigor. The researcher never sees this distinction.

### The Hybrid LLM + Math Approach

The acquisition function has two layers:
1. **Math** (scikit-optimize / Gaussian Process): generates candidates with highest EI
2. **LLM** (Claude): filters candidates using domain knowledge the math can't know
   (e.g., "pH > 4 causes REE precipitation — remove this candidate regardless of EI")

This is a chain, not a loop. Math runs once, LLM filters once, output returned.
The outer science loop is the only thing that actually repeats.

### Physical Experiments Are Human-Executed

The system does NOT control lab equipment. The loop represents:
- **System side**: AutoML trains model → acquisition function proposes conditions
- **Human side**: Researcher physically runs that experiment in the lab
- **Handoff**: Results logged to Unity Catalog → system picks up → next iteration

In future: robotic lab integration would automate the human side. For now,
the value is in telling the researcher WHAT to test next, not in running it.

---

## 4. The CM2US Research Domain

### Feedstocks
Coal fly ash (~38M tons/year, 481 mg/kg median REE), coal refuse,
acid mine drainage (AMD) precipitates, coal bottom ash.

### Target Minerals
Heavy REEs: Dysprosium (Dy), Terbium (Tb), Erbium (Er), Yttrium (Y)
Light REEs: Neodymium (Nd), Europium (Eu), Cerium (Ce), Lanthanum (La)
Others: Cobalt, Lithium, Scandium

### The Leaching Variable Space (Phase 1 focus)

**Independent variables (what researcher controls):**
```
hcl_concentration_M    0.5 – 6.0M     Most important single factor
temperature_C          25 – 95°C      Nonlinear effect, plateau ~65-80°C
leaching_time_min      30 – 360 min   Diminishing returns past optimum
ph_final               0.5 – 4.0      HARD CONSTRAINT: >4.0 = REE precipitation loss
liquid_solid_ratio     5:1 – 50:1     Dilution vs recovery tradeoff
particle_size_um       <75 – >250μm   Surface area effect
```

**Dependent variables (what researcher measures):**
```
nd_recovery_pct        % Nd extracted   Primary target for Phase 1
dy_recovery_pct        % Dy extracted   High-value HREE
total_ree_recovery     % total REE      Overall yield
al_coextraction_ppm    ppm Al in leach  Purity constraint: keep < 500ppm
purity_pct             REE/total solids Quality metric
```

### Domain Priors (encode in acquisition function constraints)
```
HARD CONSTRAINTS (never propose):
  ph_final > 4.5          REE precipitation — guaranteed yield loss
  temperature_C > 95      Equipment limits + energy cost cliff
  hcl_concentration_M > 6 Diminishing returns + safety

SOFT CONSTRAINTS (penalize):
  temperature_C > 80      Energy cost spikes
  hcl_concentration_M > 4.5  Al co-extraction spike risk

PRIORITY REGIONS (literature-confirmed sweet spots):
  hcl: 2.5-4.0M AND temp: 55-75°C AND time: 90-180min

BENCHMARK (literature):
  3M HCl, 65°C, 270min → Nd=70.8%, Eu=88.0%, Dy=73.4%
  Target to beat: >80% individual element recovery
```

### Phase 1 Test Hypothesis
```
"I want to optimize Neodymium recovery from our coal fly ash samples.
 I believe acid concentration is the primary driver. Maximize Nd recovery
 while keeping Al co-extraction below 500 ppm."

Parsed config:
  target:    nd_recovery_pct
  features:  [hcl_conc, temp_c, leaching_time_min, ph_final, ls_ratio]
  table:     dev_europa.gold_roses.[leaching_results_table]
  constraint: al_coextraction_ppm < 500
  success:   nd_recovery_pct > 80
```

---

## 5. Existing Stack — What's Already Built

### Infrastructure (all live)
```
Unity Catalog:    dev_europa.gold_roses (32 tables, ~45K rows of scientific data)
MLflow:           Databricks workspace, experiment tracking live
LLM:              Claude via GCP Vertex AI (anthropic[vertex] package)
SQL:              Databricks SQL Warehouse connector
Entry point:      agents/main.py (async REPL)
```

### Agents (all live, in agents/agents/)
```
root_agent.py               Coordinator, routes to sub-agents via AgentTool
databricks_catalog_agent.py 15 tools: schema, query, profile, null check, etc.
databricks_job_agent.py     12 tools: cluster mgmt, job dispatch, EDA notebooks
automl_agent.py             AutoML experiments on Unity Catalog tables
google_scholar_agent.py     5 tools: academic paper search via Serper API
edx_data_discovery_agent.py 9 tools: NETL EDX portal (17K+ energy datasets)
```

### Tools (all live, in agents/tools/)
```
catalog_tools.py        Column validation, schema fetch, data profiling
job_tools.py            Job dispatch, cluster management
eda_report_tools.py     EDA profiling reports
automl_tools.py         AutoML experiment execution
automl_report_tools.py  Parse AutoML results from MLflow ← key dependency
scholar_tools.py        Literature search
edx_tools.py            External dataset discovery
```

### Key Existing Capabilities Used by the New System
- `DatabricksCatalogAgent.profile_dataset()` → pre-experiment data validation
- `DatabricksCatalogAgent.get_dataset_schema()` → column validation in hypothesis parser
- `GoogleScholarAgent.search_papers()` → literature grounding before each iteration
- `AutoMLAgent` → inner loop, trains surrogate model
- `automl_report_tools` → parses AutoML results, feeds analysis agent
- `EDXDataDiscoveryAgent` → pulls NETL REE datasets for benchmarking

---

## 6. What Needs to Be Built — Complete List

### New Files (6 files, ~310 lines total)

```
agents/tools/analysis_tools.py        ~50 lines
  Interprets AutoML results in plain English using LLM.
  Quality checks (leakage, weak signal, few trials).
  Returns Finding dataclass.
  DEPENDENCY: automl_report_tools.py (existing)

agents/tools/hypothesis_tools.py      ~70 lines
  Translates NL hypothesis → validated ExperimentConfig.
  Fuzzy column matching (user says "temperature" → finds "temp_c").
  LLM call for structured extraction, catalog validation.
  Returns ExperimentConfig dataclass or clarification request.

agents/tools/research_session_tools.py  ~60 lines
  Single-pass glue: NL → parse → AutoML → interpret.
  Not a loop — just one research pass end-to-end.
  Foundation that the loop wraps around later.

agents/tools/memory_tools.py          ~50 lines
  Phase 1: JSON file backend (agents/output/research_memory.json)
  Phase 2: Delta table backend (dev_europa.gold_roses.research_findings)
  Functions: add_finding(), get_relevant_findings(), add_dead_end()

agents/tools/report_tools.py          ~80 lines
  Generates two artifacts at session end:
    A) Markdown summary report (agents/output/session_[id]_report.md)
    B) Databricks notebook (agents/output/session_[id]_notebook.py)
  Notebook is in proper Databricks .py format, ready to upload and run.

agents/agents/loop_agent.py           PHASE 2 — build after above 5 are working
  State machine: IDLE → PARSING → RUNNING → ANALYZING → PROPOSING → AWAITING
  Manages iteration state across conversation turns.
  Calls all tools in sequence, handles user gate.
```

### New Tool (Phase 2)
```
agents/tools/acquisition_tools.py     PHASE 2
  Phase 1 version: LLM prompt over run history → propose next conditions
  Phase 2 version: scikit-optimize GP + EI → LLM domain filter → proposal
  Returns Proposal with top-N ranked candidates + justifications
  n_results parameter: researcher can request 1, 3, or 5 options
```

### Modified Files (minor changes)
```
agents/agents/root_agent.py    Add research session routing rule (~15 lines)
agents/config.py               Add MEMORY_FILE, MEMORY_TABLE, EXPLORE_RATIO vars
agents/tools/__init__.py       Import new tool modules
agents/requirements_agents.txt Add scikit-optimize (Phase 2 only)
```

### New Infrastructure (Phase 2)
```sql
-- Scientific memory (replaces JSON file in Phase 2)
CREATE TABLE dev_europa.gold_roses.research_findings (
  finding_id STRING, session_id STRING, hypothesis STRING,
  finding_text STRING, finding_type STRING, confidence DOUBLE,
  variables ARRAY<STRING>, run_ids ARRAY<STRING>,
  tags ARRAY<STRING>, created_at TIMESTAMP
) USING DELTA;

CREATE TABLE dev_europa.gold_roses.research_sessions (
  session_id STRING, start_time TIMESTAMP, end_time TIMESTAMP,
  hypothesis STRING, iterations INT, status STRING, summary STRING
) USING DELTA;

CREATE TABLE dev_europa.gold_roses.research_dead_ends (
  dead_end_id STRING, region STRING, reason STRING,
  run_ids ARRAY<STRING>, created_at TIMESTAMP
) USING DELTA;
```

---

## 7. Build Order and Immediate Delivery Plan

### Philosophy
Build inward from what works. Each week delivers something a researcher can use.
The complex pieces (loop_agent, acquisition function) come last — they wrap around
a working foundation, not the other way around.

### Week 1 — Immediate Value, Zero New Infrastructure
```
1. analysis_tools.py
   Works on EXISTING MLflow runs right now.
   "Interpret run aaa112" → plain English findings immediately.

2. hypothesis_tools.py
   NL hypothesis → validated ExperimentConfig.
   Researcher never has to know column names.
```

### Week 2 — Single-Pass Research Assistant
```
3. research_session_tools.py
   Wire steps 1+2 together: NL → AutoML → interpreted results in one shot.

4. memory_tools.py (JSON version)
   Findings persist across sessions. No infrastructure needed.
```

### Week 3 — Shareable Output
```
5. report_tools.py
   Markdown summary + Databricks notebook generated at session end.

6. root_agent.py update
   15 lines: hook new tools into existing agent routing.
```

### Week 4+ — The Full Loop
```
7. loop_agent.py          State machine wrapping the single-pass
8. acquisition_tools.py   Phase 1 LLM, Phase 2 scikit-optimize
9. Delta tables            Replace JSON memory with proper storage
10. Memory seeds           Pre-populate with CM2US literature priors
```

### Demonstration After Week 2
```
User:   "I want to optimize Neodymium recovery from coal fly ash.
         Acid concentration is the primary driver."

System: Parsing hypothesis...
        Found table: leaching_experiments (247 rows)
        Matched columns: hcl_conc, temp_c, leaching_time_min, ph_final
        Running AutoML...

        ━━━ RESULTS ━━━
        Acid concentration confirmed as dominant predictor (importance=0.52, R²=0.81).
        Temperature is second (0.21). pH and leaching time have minor effects.
        Hypothesis: SUPPORTED.

        Open questions:
        • Temperature effect may plateau above 72°C — worth testing
        • Interaction between temperature and time unexplored

        Finding saved to memory.
```

---

## 8. Key Dataclasses (Data Shapes to Know)

```python
@dataclass
class ExperimentConfig:
    target_variable: str          # "nd_recovery_pct"
    independent_variables: list   # ["hcl_conc", "temp_c", ...]
    table_name: str               # "dev_europa.gold_roses.leaching_experiments"
    problem_type: str             # "regression"
    primary_metric: str           # "r2"
    constraints: dict             # {"ph_final": "<4.0"}
    max_runtime_minutes: int      # 20
    raw_hypothesis: str           # original user text
    confidence_in_parsing: float  # 0-1, low = ask clarification

@dataclass
class AutoMLResult:
    run_id: str                   # MLflow run ID
    best_metric_value: float      # R² = 0.81
    model_type: str               # "XGBoost"
    feature_importances: dict     # {"hcl_conc": 0.52, "temp_c": 0.21, ...}
    model_artifact_path: str      # "dbfs:/mlflow/aaa112/model"
    n_trials: int                 # 12
    warnings: list                # ["WEAK_SIGNAL", ...]

@dataclass
class Finding:
    plain_english: str            # 3-5 sentence interpretation
    hypothesis_status: str        # SUPPORTED | PARTIAL | REJECTED | ANOMALY
    key_findings: list[str]       # bullet points
    anomalies: list[str]          # unexpected patterns
    open_questions: list[str]     # what to investigate next
    confidence: float             # 0-1
    run_id: str                   # MLflow run ID for traceability
    feature_importances: dict     # mapped to user vocabulary
    top_predictor: str            # "hcl_conc"
    top_predictor_importance: float  # 0.52

@dataclass
class Proposal:                   # Phase 2 — acquisition function output
    candidates: list[dict]        # top-N parameter combinations to test
    justification: str            # plain English explanation
    confidence: float
    should_stop: bool             # system recommends stopping
    stop_reason: str | None
```

---

## 9. The Conversational Interface

The researcher interacts entirely in natural language. They never see configs,
MLflow run IDs (during a session), or SQL.

### Typical Session Flow
```
User:   "Test whether temperature drives yield on our samples."
System: [parses, runs AutoML, returns findings]
        "Temperature confirmed (importance=0.67). Proposed next:
         test temperature × time interaction. Proceed?"
User:   "Yes"
System: [runs next iteration automatically]
        "Interaction found. Yield degrades sharply above 85°C — threshold effect.
         Recommend investigating this threshold more precisely. Proceed?"
User:   "Stop and give me a report."
System: [generates session_abc_report.md + session_abc_notebook.py]
        "Report and notebook saved. Ready to share."
```

### Inline Control Commands (recognized mid-session)
```
"Show me 3 options"          → n_results=3 in acquisition function
"Be more exploratory"        → explore_ratio=0.5
"Play it safe"               → explore_ratio=0.05
"Why did you propose that?"  → system explains EI reasoning
"Stop"                       → generate report and close session
"What have we learned?"      → summarize all findings to date
```

### Error Handling (user-facing)
```
Column not found:   "I couldn't find 'reactor temperature'. Did you mean
                    temp_c or temperature_setpoint?"
Weak signal:        "The model found very weak signal (R²=0.02). The hypothesis
                    may not hold in this data, or we need more samples."
Anomaly detected:   "Unexpected: yield spiked in sample group A despite low
                    acid concentration. Pausing to flag this before continuing."
```

---

## 10. Report Output Spec

Two artifacts generated at end of every session:

### A — Markdown Summary (agents/output/session_[id]_report.md)
```
# Research Session Report
Date, hypothesis, dataset info

## Executive Summary        2-3 paragraph plain English
## Method                   Table, target, features
## Iteration Log            Table: conditions | result | key finding per run
## Key Findings             Bulleted plain English
## Optimal Conditions       Best parameter set found
## Recommended Next Steps   What to investigate next
## Supporting Data          MLflow run IDs, literature references
```

### B — Databricks Notebook (agents/output/session_[id]_notebook.py)
```
Databricks .py format (# Databricks notebook source + # COMMAND ----------)
Cells include:
  - Load source data from Unity Catalog (spark.table())
  - Reload AutoML model from MLflow by run_id
  - Feature importance plot
  - Predict on optimal conditions
  - Markdown cells with findings and recommendations
Fully reproducible: anyone can upload and re-run to verify findings
```

---

## 11. Scientific Memory Design

Memory makes the system smarter over time. It stores semantic findings,
not just numerical results.

### Phase 1: JSON File
```json
// agents/output/research_memory.json
{
  "findings": [
    {
      "finding_id": "abc123",
      "finding_text": "HCl concentration is the dominant predictor (importance=0.52)",
      "hypothesis": "acid concentration drives Nd recovery",
      "variables": ["hcl_conc", "nd_recovery_pct"],
      "confidence": 0.85,
      "run_ids": ["aaa112"],
      "tags": ["predictor", "hcl_dominant"],
      "created_at": "2026-03-09T14:00:00"
    }
  ],
  "dead_ends": []
}
```

### Phase 2: Delta Table (dev_europa.gold_roses.research_findings)
Migrate from JSON when ready. Same data, queryable at scale.

### Literature Seeds (pre-populate before first run)
```python
SEED_FINDINGS = [
    "HCl 2.5-4M is optimal range for coal fly ash REE leaching",
    "pH above 4 causes REE co-precipitation with iron hydroxides — hard constraint",
    "Temperature effect plateaus above 65-80°C for most fly ash feedstocks",
    "HREE and LREE respond differently to same leaching conditions",
    "Benchmark: 3M HCl, 65°C, 270min → Nd=70.8%, Dy=73.4%"
]
```

---

## 12. File Map — Where Everything Lives

```
agents/
├── main.py                     Entry point (REPL) — minor update Week 3
├── config.py                   Env vars — add MEMORY_FILE, EXPLORE_RATIO
├── auth.py                     GCP Vertex AI + Databricks auth — no change
│
├── agents/
│   ├── root_agent.py           Add research routing rule — Week 3
│   ├── databricks_catalog_agent.py   EXISTING — used for column validation
│   ├── databricks_job_agent.py       EXISTING — no changes
│   ├── automl_agent.py               EXISTING — inner loop, no changes
│   ├── google_scholar_agent.py       EXISTING — literature context
│   ├── edx_data_discovery_agent.py   EXISTING — NETL datasets
│   └── loop_agent.py                 NEW — Phase 2 only
│
├── tools/
│   ├── catalog_tools.py        EXISTING — column validation
│   ├── automl_tools.py         EXISTING — run AutoML
│   ├── automl_report_tools.py  EXISTING — parse MLflow results ← key dependency
│   ├── scholar_tools.py        EXISTING — literature search
│   ├── edx_tools.py            EXISTING — external datasets
│   ├── analysis_tools.py       NEW — Week 1 (interpret results)
│   ├── hypothesis_tools.py     NEW — Week 1 (parse NL hypothesis)
│   ├── research_session_tools.py  NEW — Week 2 (single-pass glue)
│   ├── memory_tools.py         NEW — Week 2 (JSON → Delta table)
│   ├── report_tools.py         NEW — Week 3 (Markdown + notebook)
│   └── acquisition_tools.py    NEW — Phase 2 (propose next experiment)
│
├── clients/
│   ├── databricks_sql.py       EXISTING — SQL queries
│   ├── databricks_sdk_client.py EXISTING — SDK, MLflow
│   ├── serper_client.py        EXISTING — Scholar search
│   └── edx_client.py           EXISTING — EDX portal
│
└── output/
    ├── research_memory.json    AUTO-CREATED — Phase 1 memory store
    ├── session_[id]_report.md  AUTO-GENERATED — shareable summary
    └── session_[id]_notebook.py AUTO-GENERATED — Databricks notebook

planning/                       All planning docs live here
notebooks/                      Databricks training pipeline (CPT, SFT)
```

---

## 13. Planning Files Reference

| File | Read for |
|---|---|
| `00_overview.md` | Vision, the two loops, success criteria |
| `01_architecture.md` | Full component map, step-by-step data flow |
| `02_existing_assets.md` | Complete inventory of existing code |
| `03_new_components.md` | Detailed design specs for new components |
| `04_implementation_phases.md` | Phase 0 → Phase 3 roadmap |
| `05_conversational_interface.md` | Full UX design — every user interaction |
| `06_scientific_memory.md` | Memory data model, retrieval strategy |
| `07_automl_integration.md` | AutoML role clarified — surrogate model builder |
| `08_cm2us_use_case.md` | Full CM2US domain deep dive |
| `09_gap_analysis.md` | Precise have vs need inventory |
| `10_build_now.md` | **Immediate build plan with code stubs** — start here |

---

## 14. Start Building

**The first file to create is `agents/tools/analysis_tools.py`.**

It adds value to existing MLflow runs immediately, requires no new infrastructure,
and is the foundation everything else builds on.

Full code spec with stubs is in `10_build_now.md` — Week 1, Step 1.

After that: `hypothesis_tools.py` → `research_session_tools.py` → `memory_tools.py`
→ `report_tools.py` → `root_agent.py` update → (then loop + acquisition function).

**The test hypothesis for end-to-end validation:**
```
"I want to optimize Neodymium recovery from our coal fly ash leaching
 experiments. I believe acid concentration is the primary driver.
 Keep Al co-extraction below 500 ppm."
```

This exercises: hypothesis parsing, column matching, AutoML dispatch,
plain English interpretation, memory write, and (at session end) report generation.
