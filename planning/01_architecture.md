# System Architecture — Closed-Loop Research

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATIONAL INTERFACE                             │
│                  (natural language in / natural language out)               │
│                           agents/main.py  ←→  user                         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOOP ORCHESTRATOR                                  │
│                      agents/agents/loop_agent.py  (NEW)                    │
│                                                                             │
│  Manages the outer loop state machine:                                      │
│  IDLE → HYPOTHESIS_PARSED → CONTEXT_LOADED → RUNNING →                     │
│  ANALYZING → PROPOSING → AWAITING_USER → [repeat or DONE]                  │
└──┬───────────────┬──────────────┬─────────────────┬────────────────┬────────┘
   │               │              │                 │                │
   ▼               ▼              ▼                 ▼                ▼
┌──────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
│Hypothesis│  │ Search  │  │ AutoML   │  │  Analysis    │  │ Acquisition  │
│ Parser   │  │ Context │  │ Agent    │  │  Agent       │  │ Function     │
│  (NEW)   │  │ Layer   │  │(EXISTING)│  │  (NEW)       │  │  (NEW)       │
└──────────┘  └─────────┘  └──────────┘  └──────────────┘  └──────────────┘
   │               │              │                 │                │
   ▼               ▼              ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SHARED DATA LAYER                                  │
│                                                                             │
│  MLflow (experiment runs)  |  Unity Catalog (source data)                  │
│  Scientific Memory Store   |  Hypothesis Log                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components in Detail

### 1. Conversational Interface (agents/main.py — extend existing)

Entry point for the user. Handles:
- Multi-turn conversation context
- Routing user input to Loop Orchestrator vs one-off queries
- Formatting Loop Orchestrator output into readable NL responses
- Interrupt handling ("stop this run", "change direction")

**Key change from current**: current main.py routes to agents directly.
Extended version detects "research session" intent and routes to Loop Orchestrator.

---

### 2. Loop Orchestrator (NEW — agents/agents/loop_agent.py)

The state machine that runs the outer loop. Responsibilities:
- Maintains loop state across turns (hypothesis, run history, current iteration)
- Calls Hypothesis Parser to translate user input
- Dispatches to Search Context Layer for grounding
- Triggers AutoML Agent for experiment execution
- Dispatches to Analysis Agent for result interpretation
- Calls Acquisition Function for next-step proposal
- Returns NL summary + proposal to user
- Accepts user decision (approve / redirect / stop) and continues

**State:**
```python
LoopState:
  hypothesis: str                    # current research hypothesis
  iteration: int                     # which run we're on
  run_history: List[RunSummary]      # all runs so far
  current_config: ExperimentConfig   # what's being run now
  status: LoopStatus                 # IDLE | RUNNING | AWAITING_USER | DONE
  memory_context: str                # retrieved from Scientific Memory
```

---

### 3. Hypothesis Parser (NEW — agents/tools/hypothesis_tools.py)

Translates user NL input into a structured experiment configuration.

**Input**: "I think temperature affects yield more than catalyst concentration.
Focus on samples with high measurement uncertainty."

**Output:**
```python
ExperimentConfig:
  target_variable: str               # "yield"
  independent_variables: List[str]   # ["temperature", "catalyst_concentration"]
  hypothesis_direction: str          # "temperature_effect > concentration_effect"
  sample_filter: str                 # "high measurement uncertainty"
  metrics: List[str]                 # ["effect_size", "feature_importance", "r2"]
  constraints: Dict                  # {"min_samples": 30}
  success_criteria: str              # "temperature feature importance > 0.5"
  table_hint: str                    # inferred from vocabulary match
```

**Implementation**: LLM call (Claude via existing Vertex AI setup) with structured
output prompt. Uses catalog tool to validate that proposed columns actually exist.

---

### 4. Search Context Layer (extend existing agents)

Before any experiment runs, the system gathers context:

**a) Literature context** — GoogleScholarAgent
- "Find papers on [target_variable] and [independent_variables]"
- Returns: known effects, expected ranges, prior contradictions

**b) Prior runs context** — Scientific Memory + MLflow query
- "What have we already tried for this hypothesis?"
- Returns: explored regions, known dead ends, best results so far

**c) Data context** — DatabricksCatalogAgent
- Confirm columns exist, check data distributions, flag class imbalance
- Returns: actual column names, row counts, null rates

This layer informs both the experiment design and the acquisition function.
It prevents re-exploring known territory and grounds proposals in reality.

---

### 5. AutoML Agent (EXISTING — agents/agents/automl_agent.py)

The inner loop. No changes needed to the agent itself.

**How it fits:**
- Loop Orchestrator passes ExperimentConfig to AutoML Agent
- AutoML Agent runs the sweep, logs to MLflow
- Returns: best trial, metrics, feature importances, run_id

**Key connection**: the structured ExperimentConfig from Hypothesis Parser
must map to AutoML Agent's expected input format (target column, feature columns,
ML problem type, dataset table name).

---

### 6. Analysis Agent (NEW — extend ScientificAdvisorAgent)

Interprets AutoML results semantically, not just numerically.

**Input**: AutoML results dict + original hypothesis + memory context

**Output:**
```
"Temperature was the most important predictor (importance=0.67), confirming
the hypothesis. However, yield degraded sharply above 85°C — a threshold
effect not captured in the linear model. The catalyst concentration had
negligible effect (importance=0.08). Measurement uncertainty was highest
in samples from table 'analytical_results' — we may want to filter those."
```

**Implementation**: Uses fine-tuned Mistral (when available) for domain-grounded
interpretation, falls back to Claude for general analysis.

Outputs:
- `plain_english_summary: str`
- `hypothesis_supported: bool | None`
- `anomalies: List[str]`
- `confidence: float`
- `key_findings: List[str]`
- `open_questions: List[str]`

---

### 7. Acquisition Function (NEW — agents/tools/acquisition_tools.py)

Decides what experiment to run next. The "what should we try next?" brain.

**Strategy: Hybrid Bayesian + LLM**

Step 1 — Numerical: Bayesian optimization over the result landscape.
Which region of the parameter space has highest expected improvement?

Step 2 — LLM grounding: Given the BO proposal + literature context + memory,
does this make scientific sense? Are there domain reasons to prefer a different direction?

Step 3 — Output: Ranked list of next-experiment proposals with justification

**Exploration vs exploitation switch:**
- Default: 80% exploit (refine best region), 20% explore (try novel territory)
- User can shift: "be more creative" → 50/50, "focus on what's working" → 95/5

**Termination triggers:**
- Success criteria met
- Diminishing returns (Δmetric < threshold for N consecutive runs)
- Exploration exhausted (all regions sampled)
- User says stop

---

### 8. Scientific Memory (NEW — agents/tools/memory_tools.py + storage)

Persistent semantic store that grows across runs and sessions.

**What it stores:**
```
Finding:
  hypothesis: str
  finding: str                        # "temperature > 85°C degrades yield"
  confidence: float
  supporting_runs: List[str]          # MLflow run IDs
  table: str                          # where the data came from
  created_at: datetime
  tags: List[str]

DeadEnd:
  region: Dict                        # {"temperature": ">85", "pH": "3-4"}
  reason: str                         # "yield consistently < 0.1 in this region"
  runs: List[str]
```

**Storage options** (in priority order):
1. Delta table in Unity Catalog (preferred — queryable, persistent, Databricks-native)
2. MLflow tags + notes on runs (simpler, already in stack)
3. Local JSON file (MVP fallback)

**Query interface:**
- `get_relevant_findings(hypothesis)` → semantic search over findings
- `get_dead_ends(parameter_space)` → avoid known bad regions
- `add_finding(finding)` → called by Analysis Agent after each run
- `get_full_history(table)` → all findings for a given data table

---

## Data Flow Diagram

```
User: "I think temperature drives yield. Test this on gold_roses."
                    │
                    ▼
         Hypothesis Parser
         ─────────────────
         target: "yield"
         IVs: ["temperature"]
         table: "gold_roses"
                    │
         ┌──────────┴──────────────┐
         ▼                         ▼
  Scholar Search           Memory + Catalog
  "temperature +           "prior runs on gold_roses"
   yield papers"           "columns: temp, yield_pct..."
         │                         │
         └──────────┬──────────────┘
                    │  context package
                    ▼
             AutoML Agent
             ─────────────
             runs sweep on gold_roses
             target: yield_pct
             features: [temp, catalyst, pH...]
             logs to MLflow
                    │
                    ▼  results
             Analysis Agent
             ─────────────
             "temperature = 0.67 importance"
             "threshold at 85°C"
             "hypothesis SUPPORTED"
                    │
         ┌──────────┴──────────────┐
         ▼                         ▼
  Memory Store              Acquisition Function
  adds finding:             "next: test temp×time
  "temp > 85 = bad"          interaction"
                    │
                    ▼
  User: "Temperature confirmed. Proceed with temp×time interaction?"
```

---

## Integration with Existing Agent Architecture

The Loop Orchestrator plugs into the existing ADK root agent as a new sub-agent.
The root agent gains a new routing rule:

```
"RESEARCH SESSION" → LoopOrchestratorAgent
```

One-off queries still route to existing agents unchanged.
Research sessions route to the orchestrator which manages the loop internally.

This preserves backwards compatibility — the existing agent system continues
to work exactly as before for non-loop queries.
