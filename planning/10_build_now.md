# Build Now — Immediate Delivery Plan

Start here. This file tells you exactly what to build first, in what order,
with what it delivers at each step. No loop, no state machine, no acquisition
function yet — just the pieces that add real value to what already exists.

**Philosophy**: Build inward from what works. Add value at every step.
Ship something useful before building the complex pieces.

---

## The Delivery Sequence

```
Week 1:  analysis_tools.py + hypothesis_tools.py
Week 2:  Single-pass glue + memory_tools.py (JSON)
Week 3:  report_tools.py + root_agent.py hook
Week 4+: loop_agent.py + acquisition_tools.py + Delta tables
```

Each week is independently useful. You can stop after any week and have
something a researcher can actually use.

---

## Week 1 — Step 1: `analysis_tools.py`

**File**: `agents/tools/analysis_tools.py`
**Size**: ~50 lines
**Dependencies**: automl_report_tools.py (already exists), Claude via Vertex AI
**New infrastructure needed**: None

### What it does
Interprets existing AutoML/MLflow results in plain English.
Works on runs that already exist in MLflow right now — zero new experiments needed.

### Before vs After
```
BEFORE: Researcher opens MLflow dashboard, reads feature importance numbers manually
AFTER:  "Interpret run aaa112" → plain English findings in 30 seconds
```

### What to build

```python
# agents/tools/analysis_tools.py

from dataclasses import dataclass
from typing import Optional
from agents.clients.databricks_sdk_client import get_mlflow_client

@dataclass
class Finding:
    plain_english: str
    hypothesis_status: str        # SUPPORTED | PARTIAL | REJECTED | ANOMALY
    key_findings: list[str]
    anomalies: list[str]
    open_questions: list[str]
    confidence: float
    run_id: str
    feature_importances: dict[str, float]
    top_predictor: str
    top_predictor_importance: float


def quality_checks(automl_results: dict) -> list[str]:
    """Flag data quality issues before interpreting."""
    warnings = []
    r2 = automl_results.get("best_metric_value", 0)
    importances = automl_results.get("feature_importances", {})

    if r2 < 0.05:
        warnings.append("WEAK_SIGNAL: R² < 0.05 — model barely outperforms baseline")
    if importances and max(importances.values()) > 0.95:
        warnings.append("LEAKAGE_RISK: Single feature dominates (>95%) — check for data leakage")
    if automl_results.get("n_trials", 10) < 5:
        warnings.append("FEW_TRIALS: Less than 5 trials — results may not be optimal")
    return warnings


def interpret_automl_results(
    automl_results: dict,
    hypothesis: str,
    memory_context: list = None,
    literature_context: list = None
) -> Finding:
    """
    Translate AutoML results into a plain English Finding.

    automl_results: dict from automl_report_tools (feature importances, R², run_id)
    hypothesis: the original user hypothesis string
    memory_context: prior findings from memory_tools (optional)
    literature_context: papers from GoogleScholarAgent (optional)
    """
    warnings = quality_checks(automl_results)
    importances = automl_results.get("feature_importances", {})
    top_predictor = max(importances, key=importances.get) if importances else "unknown"
    r2 = automl_results.get("best_metric_value", 0)

    memory_text = "\n".join([f"- {f}" for f in (memory_context or [])])
    literature_text = "\n".join([f"- {p}" for p in (literature_context or [])])
    warnings_text = "\n".join(warnings) if warnings else "None"

    prompt = f"""
You are a scientific data analyst interpreting machine learning results
for a critical minerals researcher.

Original hypothesis: {hypothesis}

AutoML Results:
- Best model: {automl_results.get("model_type", "unknown")}
- R² score: {r2:.3f}
- Feature importances: {importances}
- Number of trials: {automl_results.get("n_trials", "unknown")}
- Warnings: {warnings_text}

Prior knowledge from memory:
{memory_text or "None"}

Relevant literature:
{literature_text or "None"}

Provide a scientific interpretation. Be specific about:
1. Whether the hypothesis is supported by the results
2. Which variables matter most and how much
3. Any unexpected patterns or anomalies
4. What questions this raises for the next experiment

Return JSON with these exact keys:
{{
  "plain_english": "3-5 sentence summary a researcher would understand",
  "hypothesis_status": "SUPPORTED | PARTIAL | REJECTED | ANOMALY",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "anomalies": ["anomaly 1"] or [],
  "open_questions": ["question 1", "question 2"],
  "confidence": 0.0-1.0
}}
"""

    # LLM call via existing Vertex AI setup
    from agents.auth import get_vertex_client
    response = get_vertex_client().call(prompt)
    parsed = parse_json_response(response)

    return Finding(
        plain_english=parsed["plain_english"],
        hypothesis_status=parsed["hypothesis_status"],
        key_findings=parsed["key_findings"],
        anomalies=parsed["anomalies"],
        open_questions=parsed["open_questions"],
        confidence=parsed["confidence"],
        run_id=automl_results.get("run_id", ""),
        feature_importances=importances,
        top_predictor=top_predictor,
        top_predictor_importance=importances.get(top_predictor, 0)
    )
```

### ADK Tool Wrapper
```python
def analyze_automl_run(run_id: str, hypothesis: str) -> dict:
    """
    ADK-compatible tool. Fetches AutoML results from MLflow by run_id
    and returns a plain English interpretation.

    Args:
        run_id: MLflow run ID to interpret
        hypothesis: the research hypothesis being investigated

    Returns:
        dict with finding, hypothesis_status, key_findings, top_predictor
    """
    from agents.tools.automl_report_tools import get_automl_run_results
    automl_results = get_automl_run_results(run_id)
    finding = interpret_automl_results(automl_results, hypothesis)

    return {
        "plain_english": finding.plain_english,
        "hypothesis_status": finding.hypothesis_status,
        "key_findings": finding.key_findings,
        "anomalies": finding.anomalies,
        "open_questions": finding.open_questions,
        "top_predictor": finding.top_predictor,
        "top_predictor_importance": finding.top_predictor_importance,
        "confidence": finding.confidence,
        "run_id": run_id
    }
```

### Test It
```
User: "Interpret AutoML run aaa112. My hypothesis was that acid
       concentration drives Nd recovery."

Expected output:
  "Acid concentration confirmed as the dominant predictor (importance=0.52,
   R²=0.81). Temperature is second most important (0.21). Leaching time and
   pH have minor effects. The model fit is strong — hypothesis supported.
   One open question: temperature effect may be nonlinear above 72°C,
   worth investigating in the next experiment."
```

---

## Week 1 — Step 2: `hypothesis_tools.py`

**File**: `agents/tools/hypothesis_tools.py`
**Size**: ~70 lines
**Dependencies**: DatabricksCatalogAgent.get_dataset_schema() (already exists)
**New infrastructure needed**: None

### What it does
Translates a user's natural language hypothesis into a validated, structured
ExperimentConfig. Researcher never has to know column names or configure AutoML manually.

### Before vs After
```
BEFORE: Researcher manually specifies table, target column, feature columns in AutoML
AFTER:  "Optimize Nd recovery from coal fly ash" → system resolves columns and config
```

### What to build

```python
# agents/tools/hypothesis_tools.py

from dataclasses import dataclass, field
from typing import Optional
from difflib import get_close_matches


@dataclass
class ExperimentConfig:
    # Core
    target_variable: str               # column name to predict
    target_variable_user: str          # what the user called it
    independent_variables: list[str]   # column names to use as features
    table_name: str                    # fully qualified Unity Catalog table

    # Problem framing
    problem_type: str                  # "regression" | "classification"
    primary_metric: str                # "r2" | "accuracy" | "f1"

    # Filters
    sample_filter: Optional[str]       # SQL WHERE clause fragment
    min_samples: int = 30

    # Constraints
    constraints: dict = field(default_factory=dict)   # {"ph_final": "<4.0"}

    # Optimization goal
    success_criteria: Optional[str] = None
    max_runtime_minutes: int = 20

    # Source
    raw_hypothesis: str = ""
    confidence_in_parsing: float = 1.0


def fuzzy_match_column(user_term: str, available_columns: list[str]) -> Optional[str]:
    """
    Match user vocabulary to actual column names.
    "temperature" → "temp_c"
    "acid concentration" → "hcl_conc"
    """
    # Exact match first
    if user_term in available_columns:
        return user_term

    # Fuzzy match using difflib
    matches = get_close_matches(user_term.lower(),
                                 [c.lower() for c in available_columns],
                                 n=1, cutoff=0.6)
    if matches:
        idx = [c.lower() for c in available_columns].index(matches[0])
        return available_columns[idx]

    # Substring match as fallback
    for col in available_columns:
        if user_term.lower() in col.lower() or col.lower() in user_term.lower():
            return col

    return None


def parse_hypothesis(
    user_text: str,
    available_tables: list[str],
    available_columns: dict[str, list[str]]   # table_name → [column_names]
) -> ExperimentConfig | dict:
    """
    Translate NL hypothesis into ExperimentConfig.
    Returns ExperimentConfig if confident, or dict with clarification_needed=True.
    """
    # Build column context for LLM
    columns_context = "\n".join([
        f"Table: {table}\n  Columns: {', '.join(cols)}"
        for table, cols in available_columns.items()
    ])

    prompt = f"""
You are parsing a scientific research hypothesis into a structured experiment config.

User hypothesis: "{user_text}"

Available tables and columns:
{columns_context}

Extract the following and return as JSON:
{{
  "target_variable_user": "what the user called the outcome variable",
  "target_column_guess": "closest matching column name from available columns",
  "feature_columns_guess": ["column1", "column2", ...],
  "table_name_guess": "most likely table name",
  "problem_type": "regression or classification",
  "primary_metric": "r2 or accuracy or f1",
  "constraints": {{"column": "condition"}} or {{}},
  "confidence": 0.0-1.0,
  "clarification_needed": false,
  "clarification_question": null
}}

If confidence < 0.7, set clarification_needed to true and write a specific
clarification question. Be precise about column names.
"""

    from agents.auth import get_vertex_client
    response = get_vertex_client().call(prompt)
    parsed = parse_json_response(response)

    # Return clarification request if uncertain
    if parsed.get("clarification_needed") or parsed.get("confidence", 1.0) < 0.7:
        return {
            "clarification_needed": True,
            "question": parsed.get("clarification_question",
                                    "Could you clarify which columns you're referring to?")
        }

    # Fuzzy match columns to validate they exist
    table = parsed["table_name_guess"]
    cols = available_columns.get(table, [])

    target = fuzzy_match_column(parsed["target_column_guess"], cols)
    features = [
        fuzzy_match_column(f, cols) for f in parsed["feature_columns_guess"]
    ]
    features = [f for f in features if f is not None]  # drop unresolved

    if not target:
        return {
            "clarification_needed": True,
            "question": f"I couldn't find '{parsed['target_column_guess']}' in {table}. "
                       f"Available columns: {', '.join(cols[:10])}"
        }

    return ExperimentConfig(
        target_variable=target,
        target_variable_user=parsed["target_variable_user"],
        independent_variables=features,
        table_name=f"dev_europa.gold_roses.{table}",
        problem_type=parsed.get("problem_type", "regression"),
        primary_metric=parsed.get("primary_metric", "r2"),
        constraints=parsed.get("constraints", {}),
        max_runtime_minutes=20,
        raw_hypothesis=user_text,
        confidence_in_parsing=parsed.get("confidence", 0.9)
    )
```

### ADK Tool Wrapper
```python
def parse_research_hypothesis(
    hypothesis: str,
    table_name: str = None
) -> dict:
    """
    ADK-compatible tool. Parse a natural language research hypothesis
    into a structured experiment configuration.

    Args:
        hypothesis: natural language research hypothesis
        table_name: optional specific table to use (otherwise auto-detected)

    Returns:
        ExperimentConfig as dict, or clarification request
    """
    from agents.tools.catalog_tools import list_tables, get_dataset_schema

    # Get available tables and columns from catalog
    tables = list_tables("dev_europa", "gold_roses")
    columns = {t: get_dataset_schema(f"dev_europa.gold_roses.{t}") for t in tables}

    result = parse_hypothesis(hypothesis, tables, columns)

    if isinstance(result, dict) and result.get("clarification_needed"):
        return result

    return {
        "target_variable": result.target_variable,
        "features": result.independent_variables,
        "table": result.table_name,
        "problem_type": result.problem_type,
        "primary_metric": result.primary_metric,
        "constraints": result.constraints,
        "confidence": result.confidence_in_parsing,
        "raw_hypothesis": result.raw_hypothesis
    }
```

---

## Week 2 — Step 3: Single-Pass Glue

**File**: `agents/tools/research_session_tools.py`
**Size**: ~60 lines
**Dependencies**: hypothesis_tools.py + analysis_tools.py + AutoMLAgent
**New infrastructure needed**: None

### What it does
Chains the two Week 1 files together with the existing AutoML agent into
a single end-to-end research pass. Not a loop yet — just one hypothesis
in, interpreted results out.

```python
def run_research_pass(hypothesis: str, n_proposals: int = 1) -> dict:
    """
    Single-pass research session:
      NL hypothesis → parse → AutoML → interpret → return findings

    This is the foundation of the loop. The loop is just this,
    repeated with the acquisition function choosing what to run next.
    """
    # Step 1: Parse hypothesis
    config = parse_research_hypothesis(hypothesis)
    if config.get("clarification_needed"):
        return config  # ask user for clarification

    # Step 2: Run AutoML
    from agents.tools.automl_tools import run_automl_experiment
    automl_result = run_automl_experiment(
        table=config["table"],
        target_col=config["target_variable"],
        feature_cols=config["features"],
        problem_type=config["problem_type"],
        primary_metric=config["primary_metric"],
        timeout_minutes=20
    )

    # Step 3: Interpret results
    finding = interpret_automl_results(
        automl_results=automl_result,
        hypothesis=hypothesis
    )

    return {
        "hypothesis": hypothesis,
        "config": config,
        "finding": {
            "plain_english": finding.plain_english,
            "hypothesis_status": finding.hypothesis_status,
            "key_findings": finding.key_findings,
            "top_predictor": finding.top_predictor,
            "open_questions": finding.open_questions
        },
        "mlflow_run_id": automl_result.get("run_id")
    }
```

### What researcher can do at this point
```
"Run a research analysis: I think acid concentration drives Nd recovery
 from our fly ash samples."

→ System parses, runs AutoML, returns plain English interpretation.
   No manual config. No dashboard reading. One natural language sentence in.
```

---

## Week 2 — Step 4: `memory_tools.py` (JSON version)

**File**: `agents/tools/memory_tools.py`
**Size**: ~50 lines
**Dependencies**: None
**New infrastructure needed**: None (local JSON file)

### What it does
Saves findings from each research pass so the system builds knowledge
across sessions. Starts as a simple JSON file — migrates to Delta table in Phase 2.

```python
# agents/tools/memory_tools.py

import json
import uuid
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path("agents/output/research_memory.json")

def _load() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"findings": [], "dead_ends": []}

def _save(data: dict):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, indent=2, default=str))


def add_finding(
    finding_text: str,
    hypothesis: str,
    variables: list[str],
    confidence: float,
    run_id: str,
    tags: list[str] = None
) -> str:
    """Save a new finding to memory. Returns finding_id."""
    data = _load()
    finding_id = str(uuid.uuid4())[:8]
    data["findings"].append({
        "finding_id": finding_id,
        "finding_text": finding_text,
        "hypothesis": hypothesis,
        "variables": variables,
        "confidence": confidence,
        "run_ids": [run_id],
        "tags": tags or [],
        "created_at": datetime.now().isoformat()
    })
    _save(data)
    return finding_id


def get_relevant_findings(hypothesis: str, variables: list[str] = None, k: int = 5) -> list[dict]:
    """
    Retrieve findings relevant to a hypothesis.
    Simple match: variables overlap + recency.
    """
    data = _load()
    findings = data["findings"]

    if variables:
        # Score by variable overlap
        scored = []
        for f in findings:
            overlap = len(set(f.get("variables", [])) & set(variables))
            scored.append((overlap, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        findings = [f for _, f in scored if _ > 0]

    return findings[:k]


def add_dead_end(region: dict, reason: str, run_id: str):
    """Record a parameter region that consistently performs poorly."""
    data = _load()
    data["dead_ends"].append({
        "dead_end_id": str(uuid.uuid4())[:8],
        "region": region,
        "reason": reason,
        "run_ids": [run_id],
        "created_at": datetime.now().isoformat()
    })
    _save(data)


def get_dead_ends() -> list[dict]:
    return _load()["dead_ends"]


def get_all_findings() -> list[dict]:
    return _load()["findings"]
```

---

## Week 3 — Step 5: `report_tools.py`

**File**: `agents/tools/report_tools.py`
**Size**: ~80 lines
**Dependencies**: All dataclasses from steps 1-4
**New infrastructure needed**: None (writes local files)

### What it does
Generates two shareable artifacts at the end of any research session:
1. Markdown summary report
2. Databricks notebook (.py format, ready to upload)

### Markdown Report Template
```python
def generate_markdown_report(
    hypothesis: str,
    findings: list[dict],
    config: dict,
    session_id: str
) -> str:
    """Generate a shareable Markdown research summary."""

    top_finding = findings[-1] if findings else {}
    all_findings_text = "\n".join([
        f"| {i+1} | {f.get('top_predictor','?')} | "
        f"{f.get('hypothesis_status','?')} | "
        f"{f.get('plain_english','')[:80]}... |"
        for i, f in enumerate(findings)
    ])

    return f"""# Research Session Report
**Session ID**: {session_id}
**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Status**: COMPLETE

---

## Hypothesis
{hypothesis}

## Dataset
- Table: `{config.get('table', 'unknown')}`
- Target: `{config.get('target_variable', 'unknown')}`
- Features: {', '.join(f'`{f}`' for f in config.get('features', []))}

---

## Results

{top_finding.get('plain_english', '')}

### Key Findings
{chr(10).join(f'- {kf}' for kf in top_finding.get('key_findings', []))}

### Hypothesis Status: {top_finding.get('hypothesis_status', 'UNKNOWN')}

---

## Iteration Log

| # | Top Predictor | Status | Summary |
|---|---|---|---|
{all_findings_text}

---

## Open Questions
{chr(10).join(f'- {q}' for q in top_finding.get('open_questions', []))}

---

## Supporting Data
- MLflow runs: {', '.join(f.get('mlflow_run_id', '') for f in findings)}
- Memory findings saved: {len(findings)}

*Generated by CM2US Closed-Loop Research System*
"""
```

### Databricks Notebook Generator
```python
def generate_databricks_notebook(
    hypothesis: str,
    findings: list[dict],
    config: dict,
    session_id: str
) -> str:
    """Generate a reproducible Databricks notebook (.py format)."""

    run_ids = [f.get('mlflow_run_id', '') for f in findings if f.get('mlflow_run_id')]

    return f"""# Databricks notebook source
# MAGIC %md
# MAGIC # Research Session: {hypothesis[:60]}
# MAGIC **Session**: {session_id} | **Date**: {datetime.now().strftime('%Y-%m-%d')}

# COMMAND ----------
# MAGIC %md ## Setup

# COMMAND ----------
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------
# MAGIC %md ## Source Data

# COMMAND ----------
df = spark.table("{config.get('table', '')}")
print(f"Rows: {{df.count()}}")
display(df.select({str(config.get('features', []) + [config.get('target_variable', '')])}).describe())

# COMMAND ----------
# MAGIC %md ## AutoML Results

# COMMAND ----------
# Load best model from session
run_id = "{run_ids[-1] if run_ids else ''}"
model = mlflow.pyfunc.load_model(f"runs:/{{run_id}}/model")
run = mlflow.get_run(run_id)
print("Metrics:", run.data.metrics)
print("Params:", run.data.params)

# COMMAND ----------
# Feature importance
importances = {str(findings[-1].get('feature_importances', {}) if findings else {})}
pd.Series(importances).sort_values().plot(kind='barh', title='Feature Importances')
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md ## Key Findings
# MAGIC
# MAGIC {chr(10).join(f'# MAGIC - {kf}' for kf in (findings[-1].get('key_findings', []) if findings else []))}

# COMMAND ----------
# MAGIC %md ## Recommended Next Steps
# MAGIC
# MAGIC {chr(10).join(f'# MAGIC - {q}' for q in (findings[-1].get('open_questions', []) if findings else []))}
"""
```

---

## Week 3 — Step 6: Hook Into Root Agent

**File**: `agents/agents/root_agent.py` (modify)
**Changes**: ~15 lines

Add one new routing rule and register the new tools:

```python
# Add to system prompt routing section:
"""
RESEARCH ANALYSIS: When the user wants to analyze a hypothesis, interpret
AutoML results, run a single research pass, or get plain English interpretation
of ML results, use the research session tools:
- parse_research_hypothesis: parse NL hypothesis into experiment config
- analyze_automl_run: interpret an existing MLflow run in plain English
- run_research_pass: end-to-end single research pass (hypothesis → AutoML → interpretation)
- generate_research_report: generate shareable report from session findings
"""
```

---

## What You Can Show After Each Week

### After Week 1
```
Researcher: "Interpret AutoML run aaa112. Hypothesis: acid concentration
             drives Nd recovery."
System:     [plain English interpretation of existing MLflow run]
```

### After Week 2
```
Researcher: "Optimize Nd recovery from coal fly ash.
             I think acid concentration is the primary driver."
System:     [parses NL → runs AutoML → returns plain English findings]
            [saves finding to memory]
```

### After Week 3
```
Researcher: [after 3 research passes]
            "Generate a report for this session."
System:     [writes session_abc_report.md]
            [writes session_abc_notebook.py — ready to upload to Databricks]
            "Report saved. Notebook ready to share."
```

---

## What Comes After (Phase 2 — The Loop)

Once the 4 steps above are working and tested, the loop is just:
1. Run `run_research_pass()` — same as Week 2 Step 3
2. Call acquisition function — "what should we test next?"
3. Present to user — "proceed?"
4. Repeat

The loop adds `loop_agent.py` and `acquisition_tools.py` on top of an
already-working foundation. Nothing from Weeks 1-3 changes — it just starts repeating.

```
WEEK 1-3 (now):   Single pass → useful immediately
WEEK 4+  (loop):  Wrap in repeat + add acquisition function → fully autonomous
```

---

## Files Checklist

```
Week 1:
  [ ] agents/tools/analysis_tools.py        ~50 lines
  [ ] agents/tools/hypothesis_tools.py      ~70 lines

Week 2:
  [ ] agents/tools/research_session_tools.py  ~60 lines
  [ ] agents/tools/memory_tools.py            ~50 lines
  [ ] agents/output/research_memory.json      (auto-created on first write)

Week 3:
  [ ] agents/tools/report_tools.py            ~80 lines
  [ ] agents/agents/root_agent.py             ~15 lines changed

Total new code: ~310 lines across 6 files
Total changed code: ~15 lines in 1 file
```

---

## CM2US Phase 1 Test Hypothesis

Use this to validate the full Week 1-3 build end-to-end:

```
"I want to understand what drives Neodymium recovery yield in our
 leaching experiments. I believe acid concentration is the most
 important factor. Test this on our fly ash data."

Expected flow:
  → hypothesis_tools parses: target=nd_recovery_pct, features=[hcl_conc, temp_c, ...]
  → AutoML runs on leaching_experiments table
  → analysis_tools interprets: "hcl_conc dominant (0.52), R²=0.81, SUPPORTED"
  → memory_tools saves finding
  → report_tools generates session_[id]_report.md + notebook
```
