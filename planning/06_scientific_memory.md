# Scientific Memory Design

The memory layer is what makes the system get smarter over time.
This doc specifies the data model, retrieval strategy, and evolution path.

---

## Why Memory Matters

Standard MLflow tracks *what happened*:
> "Run 47: R²=0.84, lr=1e-3, batch_size=32, features=[temp, pH, catalyst]"

Scientific Memory tracks *what was learned*:
> "Temperature is the dominant predictor of yield in gold_roses.
>  Effect saturates above 85°C (threshold finding, high confidence).
>  pH has negligible effect below pH 7.
>  Catalyst type was not predictive in 3 independent tests."

This semantic layer is what prevents Run 50 from re-discovering what Run 5 already knew.
It's also what allows the acquisition function to make scientifically meaningful proposals
rather than just mathematical ones.

---

## Data Model

### Table: research_findings

```sql
CREATE TABLE dev_europa.gold_roses.research_findings (
  finding_id     STRING NOT NULL,     -- UUID, primary key
  session_id     STRING,              -- research session that produced this
  hypothesis     STRING,              -- the research question being investigated
  finding_text   STRING,              -- plain English finding (1-3 sentences)
  finding_type   STRING,              -- "predictor" | "threshold" | "null_effect" | "anomaly" | "interaction"
  confidence     DOUBLE,              -- 0.0 to 1.0
  effect_size    DOUBLE,              -- feature importance or correlation coefficient
  direction      STRING,              -- "positive" | "negative" | "nonlinear" | "threshold"
  source_table   STRING,              -- Unity Catalog table (e.g. "gold_roses.analytical_results")
  variables      ARRAY<STRING>,       -- columns involved
  conditions     MAP<STRING, STRING>, -- context: {"sample_type": "oxide", "pH": "<7"}
  run_ids        ARRAY<STRING>,       -- MLflow run IDs supporting this finding
  iteration      INT,                 -- which loop iteration produced this
  created_at     TIMESTAMP,
  tags           ARRAY<STRING>        -- semantic tags for retrieval
)
USING DELTA;
```

### Table: research_dead_ends

```sql
CREATE TABLE dev_europa.gold_roses.research_dead_ends (
  dead_end_id    STRING NOT NULL,
  session_id     STRING,
  hypothesis     STRING,
  region         STRING,              -- JSON: {"temperature": ">85", "pH": "3-4"}
  variables      ARRAY<STRING>,
  reason         STRING,              -- "yield consistently < 0.05 in this region"
  run_ids        ARRAY<STRING>,
  created_at     TIMESTAMP
)
USING DELTA;
```

### Table: research_sessions

```sql
CREATE TABLE dev_europa.gold_roses.research_sessions (
  session_id     STRING NOT NULL,
  start_time     TIMESTAMP,
  end_time       TIMESTAMP,
  hypothesis     STRING,
  iterations     INT,
  status         STRING,              -- "active" | "completed" | "abandoned"
  summary        STRING,              -- final NL summary
  finding_ids    ARRAY<STRING>        -- all findings from this session
)
USING DELTA;
```

---

## Finding Types

| Type | When added | Example |
|---|---|---|
| `predictor` | Feature importance > 0.2 | "Temperature is a strong predictor of yield (importance=0.67)" |
| `threshold` | Nonlinear effect detected | "Yield degrades sharply above 85°C" |
| `null_effect` | Importance < 0.05 in 2+ runs | "pH has no meaningful effect on yield in tested range" |
| `anomaly` | Outlier group or unexpected pattern | "Sample group A shows yield spike not explained by features" |
| `interaction` | Interaction term important | "Temperature × time interaction is significant (combined importance=0.42)" |

---

## Retrieval Strategy

### Phase 1 (MVP): SQL-based

```python
def get_relevant_findings(hypothesis: str, k: int = 5) -> List[Finding]:
    """
    Simple retrieval: match on variables mentioned in hypothesis,
    plus recency weighting.
    """
    # Extract variable names from hypothesis using LLM
    variables = extract_variables(hypothesis)

    # SQL: find findings that involve any of these variables
    sql = f"""
    SELECT * FROM dev_europa.gold_roses.research_findings
    WHERE ARRAYS_OVERLAP(variables, ARRAY{variables})
    ORDER BY confidence DESC, created_at DESC
    LIMIT {k}
    """
    return query(sql)
```

### Phase 2: Embedding-based

```python
def get_relevant_findings(hypothesis: str, k: int = 5) -> List[Finding]:
    """
    Semantic retrieval: embed hypothesis, cosine search over finding embeddings.
    Better for paraphrase matching: "temperature drives yield" ~= "heat affects output".
    """
    query_embedding = embed(hypothesis)  # via Databricks embedding model or OpenAI

    sql = f"""
    SELECT *, VECTOR_SIMILARITY(embedding, {query_embedding}) as score
    FROM dev_europa.gold_roses.research_findings_with_embeddings
    ORDER BY score DESC
    LIMIT {k}
    """
    return query(sql)
```

Databricks has native vector search (Mosaic AI Vector Search) which can host
the embeddings table and handle ANN retrieval at scale.

---

## Memory Lifecycle

### Adding a Finding (after each iteration)
```
Analysis Agent produces Finding object
  ↓
finding.supports_memory_update == True?
  ↓ YES
Loop Orchestrator calls memory_tools.add_finding()
  ↓
Finding stored with tags, run_ids, confidence
  ↓
If finding is "null_effect" or region consistently bad:
  → also add to research_dead_ends
```

### Reading Memory (before each iteration)
```
Loop Orchestrator calls get_relevant_findings(current_hypothesis)
  ↓
Returns top-k most relevant prior findings
  ↓
Passed to Acquisition Function as memory_context
  ↓
Acquisition Function uses findings to:
  - Avoid dead ends
  - Build on confirmed predictors
  - Investigate open questions
```

### Confidence Updates
When a new run confirms an existing finding, confidence increases:
```python
if new_finding confirms existing_finding:
    existing_finding.confidence = min(1.0, existing_finding.confidence + 0.15)
    existing_finding.run_ids.append(new_run_id)

if new_finding contradicts existing_finding:
    existing_finding.confidence -= 0.20
    flag_contradiction(existing_finding, new_finding)
    # → surfaces to user: "New run contradicts finding from session 3. Review?"
```

---

## What Memory Enables Over Time

**After 5 sessions on gold_roses:**
- Acquisition function knows which variable combinations have been tested
- Dead end map prevents re-running known unproductive experiments
- Confirmed predictors are used as priors in new hypotheses

**After 20+ sessions:**
- The memory store becomes a structured knowledge base about gold_roses
- New researchers starting sessions benefit from prior work automatically
- The system can generate a "state of knowledge" summary: what do we know,
  what's uncertain, what hasn't been investigated

**Cross-table learning (future):**
- Findings from table A can inform hypotheses about related table B
- Requires semantic matching on variable names across tables
- Tags enable this: findings tagged "temperature" surface regardless of table

---

## MVP Fallback (JSON file)

Before Delta tables are provisioned, use a local JSON store:

```python
# agents/output/research_memory.json
{
  "findings": [
    {
      "finding_id": "uuid1",
      "hypothesis": "temperature drives yield",
      "finding_text": "Temperature is the strongest predictor (importance=0.67)",
      "confidence": 0.85,
      "variables": ["temperature_celsius", "yield_pct"],
      "run_ids": ["abc123"],
      "created_at": "2026-03-07T14:00:00"
    }
  ],
  "dead_ends": []
}
```

Retrieval: load JSON, filter by variable overlap, sort by confidence.
This is sufficient for Phase 1 / single-developer use.

Migrate to Delta tables in Phase 2 by writing a one-time migration script.
