# Implementation Phases — Roadmap

Three phases from MVP to full system. Each phase is independently valuable
and shippable. Later phases build on earlier ones without breaking them.

---

## Phase 0 — Foundation (Prerequisites)

Before building the loop, confirm these are working:

- [ ] AutoMLAgent runs end-to-end on a Unity Catalog table
- [ ] AutoML results are logged to MLflow correctly
- [ ] `automl_report_tools.py` returns a parseable result dict
- [ ] GoogleScholarAgent returns results for domain-relevant queries
- [ ] DatabricksCatalogAgent.profile_dataset() works on gold_roses tables

**Why first**: The loop depends on these being reliable. If AutoML is flaky,
the loop will fail at the inner loop on every iteration. Validate before building on top.

**Estimated work**: 1-2 sessions of testing and minor fixes.

---

## Phase 1 — MVP: Manual-Assist Loop

**Goal**: A working loop where the user provides more structure, the system handles execution and reporting.

**What's built:**
- `hypothesis_tools.py` — basic NL → ExperimentConfig (LLM, no fuzzy column matching yet)
- `analysis_tools.py` — result interpretation using Claude (no Mistral endpoint yet)
- `memory_tools.py` — JSON file backend (no Delta table yet)
- `loop_agent.py` — orchestrator with simplified state machine
- `root_agent.py` — add loop routing rule

**What's NOT in Phase 1:**
- Acquisition function (user manually says "try X next")
- Bayesian optimization
- Embedding-based memory retrieval
- Fine-tuned Mistral for analysis

**User experience in Phase 1:**
```
User: "Test whether temperature drives yield on gold_roses"
System: [parses hypothesis, runs AutoML, returns results in plain English]
        "Temperature was the strongest predictor (0.67 importance).
         What would you like to investigate next?"
User: "Test temperature vs time interaction now"
System: [parses new direction, runs AutoML, returns results]
        ...
```

The user is still directing "what next" — the system handles "how."

**Files to create/modify:**
```
NEW:  agents/tools/hypothesis_tools.py
NEW:  agents/tools/analysis_tools.py
NEW:  agents/tools/memory_tools.py       (JSON backend)
NEW:  agents/agents/loop_agent.py
MOD:  agents/agents/root_agent.py        (add routing rule)
MOD:  agents/config.py                   (add MEMORY_FILE config)
```

**Success criteria for Phase 1:**
- [ ] User can state a hypothesis in NL and get an ExperimentConfig back
- [ ] Loop runs AutoML and returns plain English results
- [ ] Findings are saved to memory file
- [ ] User can run 3 consecutive iterations conversationally
- [ ] Results are traceable to MLflow run IDs

---

## Phase 2 — Autonomous Proposals

**Goal**: The system proposes what to try next. User approves, redirects, or stops.

**What's added:**
- `acquisition_tools.py` — LLM-based proposal generation
- Memory retrieval in proposal (avoids dead ends)
- Literature context injected before each iteration
- Anomaly detection in analysis (flags surprises before proposing next step)
- Delta table backend for memory (replaces JSON file)

**User experience in Phase 2:**
```
User: "Test whether temperature drives yield on gold_roses"
System: [runs iteration 1]
        "Temperature confirmed (0.67 importance). I propose we next test
         the temperature×time interaction — literature suggests this is
         the key unexplored direction. Proceed?"
User: "Yes"
System: [runs iteration 2 automatically]
        "Temperature×time interaction found (0.42 combined importance).
         Interestingly, yield degrades sharply above 85°C — this looks
         like a threshold effect, not linear. I recommend investigating
         this threshold more precisely. Proceed?"
User: "Yes, but also check if pH matters at that threshold"
System: [integrates user direction + system proposal, runs iteration 3]
        ...
```

**Files to create/modify:**
```
NEW:  agents/tools/acquisition_tools.py
MOD:  agents/tools/memory_tools.py       (add Delta table backend)
MOD:  agents/tools/analysis_tools.py     (add anomaly detection)
MOD:  agents/agents/loop_agent.py        (add proposal stage)
MOD:  agents/config.py                   (add MEMORY_TABLE, EXPLORE_RATIO)
```

**New Unity Catalog tables:**
```sql
dev_europa.gold_roses.research_findings
dev_europa.gold_roses.research_dead_ends
```

**Success criteria for Phase 2:**
- [ ] System proposes next experiment without user specifying it
- [ ] Proposals cite justification (literature or prior runs)
- [ ] Dead ends from prior runs are excluded from proposals
- [ ] Anomalies trigger user-confirmation gate before continuing
- [ ] Session history persists across restarts (Delta table)

---

## Phase 3 — Full Autonomy + Domain Intelligence

**Goal**: System runs multi-iteration research with minimal user intervention.
Fine-tuned Mistral provides domain-grounded analysis. Mathematical rigor added
to acquisition function.

**What's added:**
- ScientificAdvisorAgent deployed (Mistral endpoint live)
- Gaussian Process surrogate model in acquisition function
- Embedding-based memory retrieval (semantic search over findings)
- Multi-fidelity experiment scheduling (fast probe → full run pipeline)
- Session reports: full research summary at end of session
- Parallel context loading (scholar + memory + catalog in parallel)
- Configurable autonomy level (how much to self-direct vs ask user)

**User experience in Phase 3:**
```
User: "Investigate the key drivers of yield in gold_roses.
       Run until you've tested the top 5 hypotheses or you've
       found 3 strong predictors."
System: [runs 4 iterations autonomously, pausing only for anomalies]
        [at end]: "Research session complete. Summary:
         Strong predictors found: temperature (0.67), time (0.42)
         Weak/no effect: pH, catalyst type
         Threshold discovered: yield degrades >85°C
         Anomaly flagged on iteration 3: yield spike in sample group A
         All findings saved. Recommend deeper analysis of threshold effect.
         Full report: [link to MLflow experiment dashboard]"
```

**Files to create/modify:**
```
NEW:  agents/clients/mistral_client.py   (from TODO_scientific_advisor.md)
NEW:  agents/clients/mlflow_client.py    (from TODO_scientific_advisor.md)
NEW:  agents/tools/advisor_tools.py      (from TODO_scientific_advisor.md)
MOD:  agents/tools/acquisition_tools.py  (add GP surrogate model)
MOD:  agents/tools/memory_tools.py       (add embedding search)
MOD:  agents/agents/loop_agent.py        (add multi-fidelity, session report)
MOD:  agents/agents/root_agent.py        (register ScientificAdvisorAgent)
```

**Success criteria for Phase 3:**
- [ ] System runs 5 iterations with only one user interaction (hypothesis input)
- [ ] Fine-tuned Mistral provides domain-specific analysis commentary
- [ ] Acquisition function avoids re-exploring known regions mathematically
- [ ] Session summary report generated in NL at end
- [ ] Embedding search surfaces relevant prior findings across sessions

---

## Phase Comparison

| Capability | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| NL hypothesis → experiment | ✓ | ✓ | ✓ |
| AutoML execution | ✓ | ✓ | ✓ |
| Plain English results | ✓ | ✓ | ✓ |
| Memory (what we've learned) | JSON | Delta table | Delta + embeddings |
| Proposes next experiment | ✗ (user decides) | ✓ (LLM) | ✓ (LLM + GP) |
| Literature grounding | ✗ | ✓ | ✓ |
| Anomaly detection | ✗ | ✓ | ✓ |
| Domain-specific analysis | ✗ | ✗ | ✓ (Mistral) |
| Multi-fidelity experiments | ✗ | ✗ | ✓ |
| Fully autonomous runs | ✗ | partial | ✓ |
| Session summary report | ✗ | ✗ | ✓ |

---

## Per-Phase Effort Estimate

| Phase | New files | Modified files | Rough complexity |
|---|---|---|---|
| Phase 0 | 0 | 0 | Low — validation only |
| Phase 1 | 4 | 2 | Medium — core new components |
| Phase 2 | 1 | 4 | Medium — acquisition + persistence |
| Phase 3 | 3 | 4 | High — GP model + Mistral + embeddings |

---

## Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| AutoML Agent unreliable on specific tables | 0 | Test with target table before Phase 1 |
| Hypothesis Parser misinterprets column names | 1 | Add clarification gate when confidence < 0.7 |
| Loop runs too many iterations / burns compute | 2 | Hard limit: max_iterations config, runtime budget per run |
| Memory grows stale / contradictory findings | 2 | Tag findings with confidence; low-confidence ones expire |
| Mistral endpoint not deployed before Phase 3 | 3 | Claude fallback is already designed in |
| GP surrogate model overfits small run history | 3 | Minimum 5 runs before GP kicks in; LLM acquisition until then |
