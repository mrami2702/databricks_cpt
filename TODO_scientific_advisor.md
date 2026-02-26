# TODO: ScientificAdvisorAgent

This agent is deferred until the fine-tuned Mistral model is deployed to a Databricks serving endpoint.

---

## What It Does

The ScientificAdvisorAgent adds two capability layers on top of the existing 3-agent system:

1. **Domain Intelligence** — calls the fine-tuned Mistral-7B (CPT + SFT) for scientific interpretation
2. **ML Model Management** — queries the MLflow registry and uses Mistral for model selection reasoning

---

## Files to Create

```
agents/
  clients/
    mlflow_client.py          # AgentMlflowClient + ModelInfo dataclass
    mistral_client.py         # query_mistral() → Databricks serving endpoint
  tools/
    advisor_domain_tools.py   # 7 domain intelligence functions
    advisor_mlflow_tools.py   # 6 ML model management functions
  agents/
    scientific_advisor_agent.py  # LlmAgent with all 13 tools
```

---

## Domain Intelligence Tools (7)

All call `query_mistral(DOMAIN_SYSTEM_PROMPT, user_prompt)`.
When `FINE_TUNED_ENDPOINT` is not set → `__FALLBACK__` sentinel → return prompt as context so Claude answers directly.

| Tool | Args | Returns |
|------|------|---------|
| `interpret_results` | data, question, table_name | interpretation, confidence |
| `suggest_next_query` | goal, queries_run, findings | suggested_query, rationale, expected_insight |
| `recommend_experiment` | objective, data_summary, constraints | recommendation, methodology, expected_outcomes |
| `diagnose_anomaly` | symptom, evidence | diagnosis, possible_causes, investigation_steps |
| `explain_scientific_context` | table_name, column_name, value | explanation, typical_range, significance |
| `validate_hypothesis` | hypothesis, evidence | verdict, reasoning, caveats |
| `generate_analysis_plan` | research_question, available_data | steps, tools_needed, expected_timeline |

---

## ML Model Management Tools (6)

Use `AgentMlflowClient` for model data. `recommend_ml_model` calls Mistral for reasoning
(the SFT model was explicitly trained on this use case in `generate_sft_mlflow.py`).

| Tool | Args | Returns |
|------|------|---------|
| `list_ml_models` | — | models list with version/stage |
| `describe_ml_model` | model_name | metrics, params, tags, description |
| `recommend_ml_model` | task, input_features, constraints | recommended_model, version, justification, risks |
| `compare_ml_models` | model_names (list) | comparison table (metric per model) |
| `suggest_model_training` | task, available_data, current_models | suggestion, approach, rationale |
| `get_model_lineage` | model_name | training_runs, data_sources, param_history |

---

## Key Implementation Notes

### mistral_client.py
```python
def query_mistral(system_prompt, user_prompt, max_tokens=512, temperature=0.7) -> str:
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

### mlflow_client.py
Mirrors patterns from `mlflow_query.py` (lines 26-37, 145-178) without importing torch:
- `ModelInfo` dataclass: name, version, stage, metrics, params, tags, description, run_id
- `AgentMlflowClient.get_registered_models()` — search_registered_models()
- `AgentMlflowClient.get_model_info(model_name)` — get_latest_versions across stages
- `AgentMlflowClient.fmt_metric(value)` — None → "N/A" (mirrors generate_sft_mlflow.py)

### Fallback pattern (all 7 domain tools)
```python
result = query_mistral(DOMAIN_SYSTEM_PROMPT, user_prompt)
if result == "__FALLBACK__":
    return {"interpretation": user_prompt, "confidence": "answered_by_coordinator"}
return {"interpretation": result, "confidence": "domain_model"}
```

### Wiring into root_agent.py
```python
# In agents/agents/root_agent.py — add after google_scholar_agent:
from agents.agents.scientific_advisor_agent import scientific_advisor_agent
# ...
tools=[
    AgentTool(agent=databricks_catalog_agent),
    AgentTool(agent=databricks_job_agent),
    AgentTool(agent=google_scholar_agent),
    AgentTool(agent=scientific_advisor_agent),  # add this line
],
```

Also update root_agent instruction to include:
```
SCIENTIFIC REASONING + ML MODELS (interpret results, diagnose anomalies, recommend experiments,
validate hypotheses, compare/recommend ML models from registry):
  → ScientificAdvisorAgent
```

---

## Prerequisites Before Implementing

1. **Deploy SFT model to Databricks Model Serving**
   - Register model: `mlflow.register_model(model_uri, "mistral-sft")`
   - Create serving endpoint in Databricks UI or via SDK
   - Set `FINE_TUNED_ENDPOINT=<endpoint-name>` in `.env`

2. **Model endpoint accepts OpenAI-compatible format**
   ```json
   POST /serving-endpoints/<name>/invocations
   {"messages": [...], "max_tokens": 512, "temperature": 0.7}
   ```
   Returns: `{"choices": [{"message": {"content": "..."}}]}`

3. **MLflow access** — `DATABRICKS_TOKEN` must have access to MLflow workspace
   (same token used for data access should cover this)

---

## Tool Chaining Example (from handoff doc)

```
User: "Why are my yields dropping?"
  → query_data("SELECT date, yield FROM experiments ORDER BY date DESC")
  → interpret_results(raw_data, "why yields dropping", "experiments")
     LLM: "15% decline over 2 weeks"
  → suggest_next_query(goal, queries_run, findings)
     LLM: "Query catalyst batch info"
  → query_data("SELECT batch_id, manufacture_date FROM catalysts...")
  → diagnose_anomaly("yield decline", all_evidence)
     LLM: "Catalyst batch #2847 shows degradation. Confidence: HIGH"
```
