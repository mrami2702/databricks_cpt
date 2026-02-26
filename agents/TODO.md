# Agent System — TODO

Deferred work for the multi-agent scientific assistant (agents/ directory).
Read this before adding new features — it tracks what was intentionally skipped, why,
and what's needed to implement each item. Organized by priority.

Current state: 5 agents live — RootAgent (coordinator), DatabricksCatalogAgent,
DatabricksJobAgent, GoogleScholarAgent, EDXDataDiscoveryAgent.
Primary gap: ScientificAdvisorAgent (deferred until fine-tuned Mistral endpoint is deployed).
Secondary gap: EDX agent missing ranking, data ingestion, and join advisor tools.

---

## HIGH PRIORITY: ScientificAdvisorAgent

The fifth agent in the system. Deferred until the fine-tuned Mistral-7B model is deployed
to a Databricks Model Serving endpoint.

### What It Does

Two capability layers not covered by the current 4-agent system:

1. **Domain Intelligence** — routes scientific interpretation questions to the
   fine-tuned Mistral-7B (CPT + SFT trained on `dev_europa.gold_roses` data)
2. **ML Model Management** — queries the MLflow registry and uses Mistral for
   model selection reasoning and experiment recommendation

### Files to Create (5 new files)

```
agents/
  clients/
    mistral_client.py             # query_mistral() → Databricks serving endpoint
    mlflow_client.py              # AgentMlflowClient + ModelInfo dataclass
  tools/
    advisor_domain_tools.py       # 7 scientific domain intelligence tools
    advisor_mlflow_tools.py       # 6 ML model management tools
  agents/
    scientific_advisor_agent.py   # LlmAgent wiring all 13 tools
```

### File to Modify (1)

```
agents/agents/root_agent.py       # add ScientificAdvisorAgent import + AgentTool
```

---

## Tool Specs: Domain Intelligence Tools (7 tools → advisor_domain_tools.py)

All 7 tools call `query_mistral(DOMAIN_SYSTEM_PROMPT, user_prompt)`.
When `FINE_TUNED_ENDPOINT` is not set → returns `__FALLBACK__` sentinel →
tools return the prompt as context so the Claude coordinator answers directly.

### `interpret_results`
```
Args:   data (str|dict), question (str), table_name (str)
Action: Passes tabular results + research question to Mistral for scientific interpretation
Return: {"interpretation": str, "confidence": str, "key_findings": list[str]}
```

### `suggest_next_query`
```
Args:   goal (str), queries_run (list[str]), findings (str)
Action: Asks Mistral to recommend the next SQL/analysis step given findings so far
Return: {"suggested_query": str, "rationale": str, "expected_insight": str}
```

### `recommend_experiment`
```
Args:   objective (str), data_summary (str), constraints (str)
Action: Mistral proposes an experiment design given objective and available data
Return: {"recommendation": str, "methodology": str, "expected_outcomes": list[str]}
```

### `diagnose_anomaly`
```
Args:   symptom (str), evidence (str)
Action: Mistral diagnoses what's causing the observed anomaly
Return: {"diagnosis": str, "possible_causes": list[str], "investigation_steps": list[str]}
```

### `explain_scientific_context`
```
Args:   table_name (str), column_name (str), value (str|float)
Action: Mistral explains what a data value means in the domain (typical range, significance)
Return: {"explanation": str, "typical_range": str, "significance": str}
```

### `validate_hypothesis`
```
Args:   hypothesis (str), evidence (str)
Action: Mistral evaluates whether evidence supports or refutes the hypothesis
Return: {"verdict": str, "reasoning": str, "caveats": list[str], "confidence": str}
```

### `generate_analysis_plan`
```
Args:   research_question (str), available_data (str)
Action: Mistral generates a step-by-step analysis plan
Return: {"steps": list[str], "tools_needed": list[str], "expected_timeline": str}
```

### Fallback pattern (same across all 7 tools)
```python
result = query_mistral(DOMAIN_SYSTEM_PROMPT, user_prompt)
if result == "__FALLBACK__":
    return {"interpretation": user_prompt, "confidence": "answered_by_coordinator"}
return {"interpretation": result, "confidence": "domain_model"}
```

---

## Tool Specs: ML Model Management Tools (6 tools → advisor_mlflow_tools.py)

These use `AgentMlflowClient` for model registry data. `recommend_ml_model` and
`suggest_model_training` also call Mistral for reasoning — the SFT model was
explicitly trained for this use case in `generate_sft_mlflow.py`.

### `list_ml_models`
```
Args:   (none)
Action: Returns all registered models in the MLflow workspace
Return: {"models": list[{name, latest_version, stage, description}], "count": int}
```

### `describe_ml_model`
```
Args:   model_name (str)
Action: Gets full metadata for a specific registered model
Return: {
    "name": str, "version": str, "stage": str,
    "metrics": dict, "params": dict, "tags": dict,
    "description": str, "run_id": str
}
```

### `recommend_ml_model`
```
Args:   task (str), input_features (str), constraints (str)
Action: Lists all models, passes them + task description to Mistral for recommendation
Return: {"recommended_model": str, "version": str, "justification": str, "risks": list[str]}
Note:   This is the highest-value tool — SFT model was trained specifically for this
```

### `compare_ml_models`
```
Args:   model_names (list[str])
Action: Fetches metrics for each named model, builds comparison table
Return: {"comparison": list[{model, version, metrics}], "best_by_metric": dict}
```

### `suggest_model_training`
```
Args:   task (str), available_data (str), current_models (str)
Action: Mistral suggests whether to train a new model, fine-tune, or use existing
Return: {"suggestion": str, "approach": str, "rationale": str, "estimated_effort": str}
```

### `get_model_lineage`
```
Args:   model_name (str)
Action: Traces training history — runs, data sources, parameter evolution
Return: {"training_runs": list, "data_sources": list, "param_history": list}
```

---

## Implementation Notes

### mistral_client.py — key function
```python
def query_mistral(system_prompt: str, user_prompt: str,
                  max_tokens: int = 512, temperature: float = 0.7) -> str:
    if FINE_TUNED_ENDPOINT_FALLBACK:
        return "__FALLBACK__"
    url = f"{DATABRICKS_HOST}/serving-endpoints/{FINE_TUNED_ENDPOINT}/invocations"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=get_databricks_headers(), json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

### mlflow_client.py — mirror patterns from mlflow_query.py (lines 26-37, 145-178)
- `ModelInfo` dataclass: name, version, stage, metrics, params, tags, description, run_id
- `AgentMlflowClient.get_registered_models()` → `mlflow.search_registered_models()`
- `AgentMlflowClient.get_model_info(model_name)` → `get_latest_versions` across stages
- `AgentMlflowClient.fmt_metric(value)` → None maps to "N/A" (mirrors generate_sft_mlflow.py)
- Do NOT import torch or transformers — pure mlflow + requests only

### Wiring into root_agent.py
```python
# Add import
from agents.agents.scientific_advisor_agent import scientific_advisor_agent

# Add to tools list (position 4, before closing bracket)
AgentTool(agent=scientific_advisor_agent),

# Add routing block to instruction string:
# SCIENTIFIC REASONING + ML MODEL MANAGEMENT (interpret results, diagnose anomalies,
# recommend experiments, validate hypotheses, compare or recommend ML models):
#   → ScientificAdvisorAgent
```

### Tool chaining example (from handoff)
```
User: "Why are my yields dropping?"
  → DatabricksCatalogAgent: query_data("SELECT date, yield FROM experiments ORDER BY date")
  → ScientificAdvisorAgent: interpret_results(data, "why yields dropping", "experiments")
     → Mistral: "15% decline over 2 weeks, correlates with temperature spike"
  → ScientificAdvisorAgent: suggest_next_query(goal, queries_run, findings)
     → Mistral: "Query catalyst batch info to check for degradation"
  → DatabricksCatalogAgent: query_data("SELECT batch_id, manufacture_date FROM catalysts")
  → ScientificAdvisorAgent: diagnose_anomaly("yield decline", all_evidence)
     → Mistral: "Catalyst batch #2847 shows degradation. Confidence: HIGH"
```

---

## Prerequisites Before Implementing ScientificAdvisorAgent

1. **Deploy SFT model to Databricks Model Serving**
   - Register model: `mlflow.register_model(model_uri, "mistral-sft")`
   - Create serving endpoint in Databricks UI or via SDK
   - Note endpoint name (e.g., `mistral-7b-sft`)

2. **Set env var**
   ```
   FINE_TUNED_ENDPOINT=mistral-7b-sft
   ```
   Without this, all domain tools fall back gracefully — Claude answers directly.

3. **Verify endpoint accepts OpenAI-compatible format**
   ```bash
   curl -X POST "$DATABRICKS_HOST/serving-endpoints/$FINE_TUNED_ENDPOINT/invocations" \
     -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
   ```
   Expected: `{"choices": [{"message": {"content": "..."}}]}`

4. **MLflow access** — existing `DATABRICKS_TOKEN` should cover this (same workspace)

---

## HIGH PRIORITY: EDXDataDiscoveryAgent — Missing Capabilities

### 1. Dataset Relevance Ranking Tool

**Gap:** We have no dedicated tool that scores how well a set of candidate EDX datasets
match the user's internal data. Right now:
- `find_edx_datasets_like` has `tag_overlap_count` (basic tag-based ranking — useful but narrow)
- All other search tools return results in CKAN's internal Solr relevance order, but we
  never surface that score — the agent sees an unscored list and has to reason about it
- There's no multi-dimensional match score the agent can show the user

**What to build:** `rank_edx_datasets` tool in `edx_tools.py`

```
Name:   rank_edx_datasets
Args:   dataset_names (list[str]),  # 'name' slugs from prior search results
        user_context (str)          # user's data description / schema summary
Return: {
    "rankings": [
        {
            "name": str,
            "title": str,
            "overall_score": float,           # 0-1 composite
            "tag_overlap_score": float,       # how many tags match user context keywords
            "geospatial_score": float,        # 1.0 if is_geospatial else 0.0
            "format_score": float,            # 1.0 if CSV/Shapefile/GeoJSON present
            "recency_score": float,           # normalized from publication_date
            "match_explanation": str,         # plain-English reason for score
        },
        ...
    ],
    "top_pick": str,                          # name of highest-scored dataset
}
```

**Implementation approach:**
- Call `get_package(name)` for each dataset to get full metadata
- Score each dimension independently (all 0-1):
  - `tag_overlap_score`: count how many of the dataset's tags appear as keywords in
    `user_context` (case-insensitive), normalize by total tags
  - `geospatial_score`: 1.0 if `is_geospatial` else 0.0
  - `format_score`: 1.0 if any of `['csv','shapefile','geojson']` in resource_formats else 0.5
  - `recency_score`: normalize publication_date year against range 2010-2025
- `overall_score` = weighted average (suggested: tag 40%, geo 30%, format 20%, recency 10%)
- Sort results descending by overall_score

**Also:** Expose CKAN Solr score in `_parse_dataset()` — it's already in the raw response
at `raw.get("score")` but currently discarded. Adding it to the parsed dict gives the agent
a free relevance signal with zero API cost.

---

### 2. Data Ingestion into Databricks

**Gap:** The user can discover and evaluate EDX datasets but has no way to say
"I like this one — load it into my Databricks catalog" without leaving the agent system.

**What to build:** Two new tools — one for each common case.

#### `generate_ingestion_notebook`
```
Name:   generate_ingestion_notebook
Args:   dataset_name (str),         # EDX dataset slug
        resource_url (str),         # direct download URL (CSV/Shapefile) from get_edx_dataset_details
        target_catalog (str),       # e.g., "dev_europa"
        target_schema (str),        # e.g., "gold_roses"
        target_table (str),         # desired table name
Return: {
    "notebook_code": str,           # complete PySpark notebook content, ready to paste/run
    "instructions": str,            # plain-English steps for the user
    "estimated_rows": str,          # if size_bytes known: rough estimate
}
```

Implementation: generates a PySpark notebook string that:
1. `%sh wget <resource_url> -O /tmp/<filename>`
2. Reads file with `spark.read.csv("/tmp/<filename>", header=True, inferSchema=True)`
   (or `.read.format("com.databricks.spark.csv")` for larger files)
3. Writes to `<catalog>.<schema>.<table>` as Delta: `df.write.format("delta").saveAsTable(...)`
4. Prints row count + sample

Note: This tool generates code for the user to run — it does NOT run the job itself.
That avoids needing cluster selection and keeps the tool safe and fast.
If we want auto-execution later, it should delegate to `DatabricksJobAgent.run_notebook`.

#### `suggest_ingestion_strategy`
```
Name:   suggest_ingestion_strategy
Args:   resource_formats (list[str]),  # from get_edx_dataset_details
        file_size_bytes (int | None),
        target_use (str)               # "join to existing table" / "standalone analysis" / etc.
Return: {
    "recommended_format": str,         # which resource to use (CSV vs Shapefile vs etc.)
    "ingestion_method": str,           # "spark.read.csv" / "Auto Loader" / "manual download"
    "join_ready": bool,                # True if CSV/GeoJSON (tabular)
    "caveats": list[str],              # e.g., "Shapefile requires geopandas or sedona"
    "next_step": str,                  # "Call generate_ingestion_notebook with resource_url X"
}
```

**Files to modify:** `agents/tools/edx_tools.py` (add 2 functions),
`agents/agents/edx_data_discovery_agent.py` (add to tools list + update instruction)

**Root agent routing addition:** The user saying "load this into Databricks" or
"get me that data" should trigger a two-agent chain:
```
EDXDataDiscoveryAgent.generate_ingestion_notebook → returns code
DatabricksJobAgent (optional) → run_notebook if user confirms execution
```

---

### 3. Join Advisor

**Gap:** The user can find a complementary EDX dataset but currently has no structured
help answering "how would I join this external data to my internal table?"
This is the most natural next question after discovery.

**What to build:** Two tools — one for compatibility analysis, one for code generation.

#### `analyze_join_compatibility`
```
Name:   analyze_join_compatibility
Args:   internal_schema (dict),     # from DatabricksCatalogAgent.describe_table or get_dataset_schema
        edx_dataset_name (str),     # EDX dataset slug
        edx_resource_url (str)      # direct CSV URL to sample headers from
Return: {
    "candidate_join_keys": [
        {
            "internal_column": str,
            "edx_column": str,       # inferred from CSV headers
            "join_type": str,        # "exact match", "fuzzy geo match", "spatial join"
            "confidence": str,       # "high" / "medium" / "low"
            "rationale": str,
        }
    ],
    "join_feasibility": str,         # "direct join", "needs preprocessing", "spatial join required"
    "blocking_issues": list[str],    # e.g., "EDX uses FIPS codes, your table uses county names"
    "preprocessing_steps": list[str]
}
```

Implementation:
- Fetch first 5 rows of `edx_resource_url` with `requests.get(..., stream=True)` to read
  just the CSV headers cheaply (no full download needed)
- Compare column names against `internal_schema` columns:
  - Exact name matches → high confidence join key
  - Semantic matches (e.g., `lat`/`latitude`, `state`/`state_name`) → medium confidence
  - No match → report and suggest preprocessing
- Check for geospatial join opportunity: if both have lat/lon → spatial join candidate
- If EDX dataset is a Shapefile (no headers to read): fall back to tag/description analysis

#### `generate_join_code`
```
Name:   generate_join_code
Args:   internal_table (str),        # fully qualified: catalog.schema.table
        edx_dataset_name (str),
        edx_resource_url (str),
        join_key_internal (str),     # column name in internal table
        join_key_external (str),     # column name in EDX file
        join_type (str)              # "inner" / "left" / "spatial"
Return: {
    "pyspark_code": str,             # complete, runnable PySpark notebook snippet
    "sql_alternative": str,          # same join as SQL (for users who prefer SQL)
    "notes": str,                    # caveats (type casting, deduplication, etc.)
}
```

Implementation: generates PySpark code string:
```python
# 1. Load EDX data
edx_df = spark.read.csv("<resource_url>", header=True, inferSchema=True)

# 2. Load internal table
internal_df = spark.table("<internal_table>")

# 3. Join
joined_df = internal_df.join(
    edx_df,
    internal_df["<join_key_internal>"] == edx_df["<join_key_external>"],
    how="<join_type>"
)

# 4. Preview
display(joined_df.limit(20))
```

For spatial joins, reference Apache Sedona (`ST_Join`) with a note to install
`databricks-mosaic` or Sedona if not already available.

**Files to modify:** `agents/tools/edx_tools.py` (add 2 functions),
`agents/agents/edx_data_discovery_agent.py` (add to tools list + update instruction)

**Suggested flow the agent should follow when user says "help me join these":**
```
User: "I like that REE dataset. How do I join it with my mineral_samples table?"
  1. DatabricksCatalogAgent: describe_table("dev_europa.gold_roses.mineral_samples")
     → returns schema: columns, types, sample values
  2. EDXDataDiscoveryAgent: get_edx_dataset_details("ree-wyoming-samples")
     → returns resource list with CSV URL
  3. EDXDataDiscoveryAgent: analyze_join_compatibility(schema, dataset, csv_url)
     → candidate join keys identified: lat/lon (spatial join) or sample_id (exact)
  4. EDXDataDiscoveryAgent: generate_join_code(...)
     → returns ready-to-run PySpark notebook snippet
  5. Root synthesizes: "Here's how to join them — paste this into a Databricks notebook"
```

---

---

## HIGH PRIORITY: Search Quality Improvements

These are improvements to how `EDXDataDiscoveryAgent` finds and ranks results.
Full technical details in `HOW_WE_SEARCH.md`. Listed in priority order by impact-to-effort.

### 1. Expose CKAN Solr score (quick win — one-line fix)

**File:** `agents/clients/edx_client.py` → `_parse_dataset()`

**Change:**
```python
# Add this field to the returned dict:
"solr_score": raw.get("score", 0.0),
```

The Solr BM25F relevance score is already in every `package_search` API response —
we just discard it. Surfacing it costs nothing and gives the agent a numeric signal
to reason about match quality. A result with score 14.2 vs the next at 0.3 means
the top result is overwhelmingly better; scores of 14.2 vs 13.9 mean both are
worth presenting. Currently the agent can't distinguish these cases.

---

### 2. Automatic query expansion via tag lookup

**File:** `agents/agents/edx_data_discovery_agent.py` → update instruction

**What:** Add a pre-search step to the agent instruction: before running any keyword
search, call `list_edx_tags()` and check whether the user's key terms appear as
exact EDX tags. If a better tag exists, use `search_edx_by_tag` with that tag
in addition to (or instead of) the free-text search.

**Example of what this fixes:**
```
User says: "rare earth elements"
EDX tag:   "REE"

Without expansion: search_edx_datasets("rare earth elements") — misses "REE" datasets
With expansion:    list_edx_tags() reveals "REE" → also run search_edx_by_tag("REE")
```

No new code needed — only an instruction update to the agent. High impact for low effort.

---

### 3. OR fallback for zero-result AND searches

**File:** `agents/tools/edx_tools.py` → modify `search_edx_multi_criteria`

**What:** If an AND query returns zero results, automatically retry with OR logic
and mark results as partial matches in the return dict.

```python
# Current behavior: criteria=["lithium","Wyoming","borehole"] → q="lithium AND Wyoming AND borehole"
# If count=0, retry: q="lithium OR Wyoming OR borehole"
# Return dict addition: "match_type": "full" | "partial"
```

**Impact:** Eliminates silent dead-ends where the user gets nothing back and the
agent has to manually retry with fewer terms. The `match_type` flag lets the agent
tell the user "I couldn't find datasets matching all three criteria — here are the
closest partial matches."

---

### 4. `rank_edx_datasets` tool — multi-dimensional scoring

See full spec above in "HIGH PRIORITY: EDXDataDiscoveryAgent — Missing Capabilities".
Listed again here because it is also a search quality improvement — it turns a flat
list of candidates into a ranked shortlist with per-dimension scores and explanations.

---

### 5. Semantic reranking (longer-term, requires new dependency)

**What:** After retrieving candidates via Solr, embed both the user's query and each
dataset's title + description using a small local embedding model. Rerank by cosine
similarity before returning results to the agent.

**Suggested model:** `sentence-transformers/all-MiniLM-L6-v2` (80MB, fast CPU inference)

**New dependency:** `sentence-transformers` → add to `requirements_agents.txt`

**Where:** New helper in `edx_tools.py` or a standalone `agents/clients/embeddings_client.py`

**Impact:** Catches semantic matches that pure keyword search misses:
- "subsurface samples" ↔ "borehole data"
- "REE" ↔ "rare earth elements"
- "geochemical assay" ↔ "elemental concentration data"

**When to implement:** After improvements 1–4 are in place and vocabulary mismatch
is still causing meaningful misses in real usage. Don't add the dependency until
the simpler fixes have been validated.

---

## MEDIUM PRIORITY

### Optional: run_model_prediction tool
A 14th tool (beyond the 13 above) proposed in the original handoff:
```
Name:   run_model_prediction
Args:   model_name (str), input_data (dict)
Action: Calls the fine-tuned model endpoint with structured input for inference
Return: {"prediction": str|dict, "model_name": str, "version": str, "latency_ms": float}
Note:   Only relevant once endpoint is deployed. Low complexity — calls query_mistral
        with a structured payload rather than a natural language prompt.
```

### System prompt refinement
After real user testing:
- Tune root_agent routing instruction based on observed misroutes
- Tighten DatabricksCatalogAgent instruction if it over-queries (fetches too many rows)
- Expand EDXDataDiscoveryAgent tag vocabulary if EDX searches return sparse results
- Consider adding few-shot examples to ScientificAdvisorAgent once it's live

### Agent-level logging
Currently no logging beyond Python exceptions. Consider adding:
- Tool call event hooks (which tool, args, latency)
- Failed tool call capture to a local log file
- Optional MLflow run logging for agent sessions (conversation_id, tools_used, tokens)

---

## LOW PRIORITY

### Pagination for large result sets
`search_edx_datasets`, `search_edx_by_location`, `search_edx_by_tag` all cap at 50 results.
If users need to page through larger result sets, add `start` parameter and surface
`total_count` vs. `returned_count` more explicitly so Claude knows to prompt the user
to narrow the search.

### Unit tests
No tests exist yet. Key things to cover:
- `edx_client._parse_dataset()` — verify field extraction and geospatial detection
- `catalog_tools` type detection — numeric/categorical classification with real Spark type strings
- `find_edx_datasets_like` — deduplication and ranking logic
- `mistral_client.query_mistral()` — fallback sentinel behavior when endpoint not set

### Streaming responses
`agents/main.py` currently collects the full response before printing.
For long scientific analysis tasks, streaming would improve UX.
Google ADK supports streaming via `run_async()` with `streaming=True` — evaluate once
the ScientificAdvisorAgent is live and response latency becomes noticeable.

### Multi-turn session persistence
`InMemorySessionService` resets on each `main.py` run. For long research workflows,
persistent session storage would allow users to continue conversations.
Google ADK supports custom `SessionService` implementations.
