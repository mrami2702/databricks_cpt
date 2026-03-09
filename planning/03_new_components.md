# New Components — Design Specs

Detailed specs for everything that needs to be built.
Each section includes: purpose, interface, implementation approach, and open questions.

---

## 1. Loop Orchestrator Agent

**File**: `agents/agents/loop_agent.py`
**Type**: ADK `LlmAgent` (same pattern as existing agents)

### Purpose
Manages the outer research loop as a state machine. Maintains iteration context
across conversation turns. Routes to the right tools at each stage.

### State Machine

```
IDLE
  │  (user submits hypothesis)
  ▼
PARSING
  │  (hypothesis_tools.parse_hypothesis called)
  ▼
LOADING_CONTEXT
  │  (scholar + memory + catalog called in parallel)
  ▼
RUNNING
  │  (automl_agent dispatched)
  ▼
ANALYZING
  │  (analysis_tools.interpret_results called)
  ▼
PROPOSING
  │  (acquisition_tools.propose_next called)
  ▼
AWAITING_USER
  │  (NL summary + proposal sent to user)
  │
  ├──→ "yes / proceed" → back to LOADING_CONTEXT (iteration+1)
  ├──→ "change direction" → back to PARSING (new hypothesis)
  ├──→ "stop" → DONE
  └──→ "tell me more" → stays in AWAITING_USER (answers question)

DONE
  (summary of all findings generated, session closed)
```

### Key Methods / Tools the Orchestrator Uses
```python
# From hypothesis_tools.py
parse_hypothesis(user_text, available_columns) → ExperimentConfig

# From acquisition_tools.py
propose_next_experiment(run_history, memory_context, literature) → Proposal

# From memory_tools.py
get_relevant_findings(hypothesis) → List[Finding]
add_finding(finding) → None

# From analysis_tools.py
interpret_automl_results(results, hypothesis, memory) → Finding

# From existing agents (via AgentTool)
automl_agent.run_experiment(config) → AutoMLResult
catalog_agent.profile_dataset(table) → DataProfile
scholar_agent.search_papers(query) → List[Paper]
```

### Loop Iteration Object
```python
@dataclass
class LoopIteration:
    iteration_id: int
    hypothesis: str
    experiment_config: ExperimentConfig
    context: ContextPackage            # literature + memory + data profile
    automl_result: AutoMLResult
    analysis: Finding
    proposal: Proposal
    user_decision: str                 # "proceed" | "redirect" | "stop"
    timestamp: datetime
```

### Conversation Interface
The orchestrator outputs two things per iteration:
1. **Summary** (what just happened, what was learned)
2. **Proposal** (what to do next, with justification)

Format:
```
━━━ ITERATION 2 COMPLETE ━━━

What we found:
[plain English analysis — 3-5 sentences]

Key findings:
• [finding 1]
• [finding 2]

Hypothesis status: SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED / ANOMALY

━━━ PROPOSED NEXT STEP ━━━

[proposal in plain English]
Justification: [why this direction makes sense]
Estimated compute: [rough AutoML runtime estimate]

Proceed? [yes / change direction / stop]
```

---

## 2. Hypothesis Parser

**File**: `agents/tools/hypothesis_tools.py`

### Purpose
Translate free-form user text into a structured ExperimentConfig that the
AutoML Agent can consume. Also validates that proposed columns exist in the catalog.

### Interface
```python
def parse_hypothesis(
    user_text: str,
    available_tables: List[str],
    available_columns: Dict[str, List[str]],   # table → columns
    prior_config: Optional[ExperimentConfig]   # for refinement turns
) -> ExperimentConfig:
```

### ExperimentConfig Schema
```python
@dataclass
class ExperimentConfig:
    # Core
    target_variable: str               # what we're trying to predict/explain
    independent_variables: List[str]   # what we think drives it
    table_name: str                    # Unity Catalog table to run on

    # Problem framing
    problem_type: str                  # "regression" | "classification" | "correlation"
    hypothesis_direction: Optional[str]  # "temp_effect > pH_effect"

    # Filters
    sample_filter: Optional[str]       # SQL WHERE clause fragment
    min_samples: int                   # minimum rows required

    # Metrics
    primary_metric: str                # what to optimize (r2, accuracy, f1...)
    secondary_metrics: List[str]       # also track these
    success_criteria: Optional[str]    # "feature_importance[temp] > 0.5"

    # Exploration
    explore_interactions: bool         # test feature interaction terms?
    max_runtime_minutes: int           # AutoML budget

    # Source
    raw_hypothesis: str                # original user text, preserved
    confidence_in_parsing: float       # 0-1, low = ask user for clarification
```

### Implementation Approach
1. LLM call (Claude) with structured output prompt
2. Prompt includes: user text + available columns + catalog schema
3. Validate output: do proposed columns exist? Is target numeric/categorical?
4. If `confidence_in_parsing < 0.7`: return clarification request instead of config
5. Column fuzzy matching: "temp" → "temperature_celsius" (Levenshtein or embedding)

### Column Fuzzy Matching
Critical: users say "temperature", catalog has "temp_celsius". The parser must
resolve this using the actual catalog schema (DatabricksCatalogAgent.get_dataset_schema).

---

## 3. Acquisition Function

**File**: `agents/tools/acquisition_tools.py`

### Purpose
Given everything we've learned so far, propose the most scientifically valuable
next experiment. Balances exploration of new territory with exploitation of
promising findings.

### Interface
```python
def propose_next_experiment(
    original_hypothesis: str,
    run_history: List[LoopIteration],
    memory_context: List[Finding],
    literature_context: List[Paper],
    explore_ratio: float = 0.2           # 0=pure exploit, 1=pure explore
) -> Proposal:

@dataclass
class Proposal:
    experiment_config: ExperimentConfig  # the proposed next run
    justification: str                   # plain English explanation
    confidence: float                    # how confident in this direction
    alternatives: List[ExperimentConfig] # 2 other options, ranked
    should_stop: bool                    # system recommends stopping
    stop_reason: Optional[str]           # if should_stop=True
```

### Decision Logic (in priority order)

1. **Anomaly follow-up**: If last run found an anomaly, investigate it first.
   "Yield dropped at 85°C — run a focused experiment around that threshold."

2. **Hypothesis refinement**: If hypothesis partially supported, narrow scope.
   "Temperature confirmed. Now test temp × time interaction."

3. **Dead end avoidance**: Filter out regions known to be unproductive (from memory).

4. **Literature guidance**: If scholar found relevant papers, use their findings
   to prioritize directions. "Papers suggest pH effect is nonlinear — test wider range."

5. **Exploration injection**: With probability=explore_ratio, propose something
   orthogonal to current trajectory. "We haven't tested catalyst type at all."

6. **Termination check**:
   - Success criteria met → recommend stop
   - Δmetric < 1% for 3 consecutive runs → recommend stop
   - All independent variables tested → recommend stop

### Implementation Approach
- Phase 1 (MVP): Pure LLM — prompt with run history + memory + literature
- Phase 2: Add Gaussian Process surrogate model over result landscape
- Phase 3: Bayesian optimization with LLM as prior (full hybrid)

The MVP LLM approach is sufficient to demonstrate the loop and get user feedback.
Gaussian Process adds mathematical rigor once the system is validated end-to-end.

---

## 4. Analysis Agent Tools

**File**: `agents/tools/analysis_tools.py`
**Integrates with**: ScientificAdvisorAgent (when deployed)

### Purpose
Transform AutoML result dicts into human-readable scientific findings.
Identify anomalies. Assess hypothesis support. Flag uncertainty.

### Interface
```python
def interpret_automl_results(
    automl_results: dict,              # from automl_report_tools
    hypothesis: str,
    experiment_config: ExperimentConfig,
    memory_context: List[Finding],
    literature_context: List[Paper]
) -> Finding:

@dataclass
class Finding:
    plain_english: str                 # 3-5 sentence summary
    hypothesis_status: str            # SUPPORTED | PARTIAL | REJECTED | ANOMALY
    key_findings: List[str]           # bullet points
    anomalies: List[str]              # unexpected patterns
    open_questions: List[str]         # what this run raised but didn't answer
    confidence: float
    run_id: str                        # MLflow run ID for traceability
    feature_importances: Dict[str, float]
    supports_memory_update: bool      # should this be added to memory?
    memory_entry: Optional[str]       # the finding to store if yes
```

### Interpretation Logic
1. Map feature importances to user-facing variable names
2. Compare top features to hypothesis (does it match what user predicted?)
3. Check for threshold effects (is the metric relationship nonlinear?)
4. Flag if key metrics are near baseline (model didn't learn anything useful)
5. Cross-reference with memory: does this contradict known findings?
6. Cross-reference with literature: surprising finding? cite the contradiction

### Model Used
- Primary: Fine-tuned Mistral-7B (domain-specific, knows gold_roses vocabulary)
- Fallback: Claude via Vertex AI (general but effective)

---

## 5. Scientific Memory

**File**: `agents/tools/memory_tools.py`
**Storage**: Unity Catalog Delta table (primary) or JSON file (MVP fallback)

### Purpose
Persistent semantic store of research findings that persists across sessions.
This is what makes the system get smarter over time.

### Schema (Delta Table)
```sql
CREATE TABLE dev_europa.gold_roses.research_findings (
  finding_id    STRING,           -- UUID
  hypothesis    STRING,           -- the research question
  finding_text  STRING,           -- plain English finding
  confidence    DOUBLE,           -- 0-1
  run_ids       ARRAY<STRING>,    -- MLflow run IDs supporting this
  source_table  STRING,           -- Unity Catalog table
  variables     ARRAY<STRING>,    -- columns involved
  tags          ARRAY<STRING>,    -- ["threshold_effect", "temperature", ...]
  created_at    TIMESTAMP,
  session_id    STRING            -- which research session produced this
);

CREATE TABLE dev_europa.gold_roses.research_dead_ends (
  dead_end_id   STRING,
  hypothesis    STRING,
  region        STRING,           -- JSON: {"temperature": ">85", "pH": "3-4"}
  reason        STRING,
  run_ids       ARRAY<STRING>,
  created_at    TIMESTAMP
);
```

### Interface
```python
def add_finding(finding: Finding) → str:              # returns finding_id
def get_relevant_findings(hypothesis: str, k: int=5) → List[Finding]:
def get_dead_ends(parameter_space: Dict) → List[DeadEnd]:
def get_session_history(session_id: str) → List[Finding]:
def get_all_findings_for_table(table: str) → List[Finding]:
```

### Retrieval Approach
- Phase 1 (MVP): SQL LIKE matching on hypothesis text + tag overlap
- Phase 2: Embedding-based semantic search (embed findings, cosine similarity)

The MVP SQL approach works for small memory stores.
As findings accumulate, embedding search becomes necessary for relevance.

### MVP Fallback (no Delta table access)
Store findings as a local JSON file at `agents/output/research_memory.json`.
This works for single-machine development and demos.

---

## 6. Root Agent Updates

**File**: `agents/agents/root_agent.py` (modify existing)

### Changes Required

1. Add `LoopOrchestratorAgent` import and `AgentTool` wrapper
2. Add routing rule:

```python
# New routing rule to add to system prompt:
"""
RESEARCH SESSION: When the user wants to run a closed-loop research experiment,
test a hypothesis systematically, or start an automated research workflow,
route to LoopOrchestratorAgent. Examples:
- "I want to test whether temperature drives yield"
- "Run a closed-loop experiment on [hypothesis]"
- "Start a research session on [topic]"
- "Systematically investigate [question]"
"""
```

3. Pass session state (loop status, iteration count) in context

### Backward Compatibility
All existing routing rules stay exactly as-is.
The new rule is additive — it only fires for explicit research session intent.

---

## Build Order (Dependencies)

```
1. memory_tools.py         (no deps — pure storage)
2. hypothesis_tools.py     (deps: catalog_tools for column validation)
3. analysis_tools.py       (deps: automl_report_tools for result parsing)
4. acquisition_tools.py    (deps: memory_tools, analysis output)
5. loop_agent.py           (deps: all above + existing agents)
6. root_agent.py update    (deps: loop_agent)
7. main.py update          (deps: loop state context)
```

This order ensures each component can be built and tested independently
before integrating into the full loop.
