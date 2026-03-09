# Closed-Loop Research System — Planning

This folder contains the comprehensive plan for building an autonomous
scientific research loop on top of the existing Databricks + multi-agent stack.

---

## Plan Files

| File | What it covers |
|---|---|
| `00_MASTER_HANDOFF.md` | **READ THIS FIRST** — complete synthesis of all planning, for agent handoff |
| `00_overview.md` | Vision, the two-loop concept, success criteria |
| `01_architecture.md` | Full system architecture with component map and data flow |
| `02_existing_assets.md` | Inventory of what's already built and how it maps to the new system |
| `03_new_components.md` | Detailed design specs for every new component to build |
| `04_implementation_phases.md` | Phased roadmap — Phase 0 (validate) → Phase 3 (full autonomy) |
| `05_conversational_interface.md` | UX design — what the user sees/says at every step |
| `06_scientific_memory.md` | Data model and retrieval strategy for cross-session knowledge |
| `07_automl_integration.md` | AutoML as surrogate model builder — role in the loop clarified |
| `08_cm2us_use_case.md` | Deep dive on CM2US critical minerals use case — variables, feedstocks, hypothesis types |
| `09_gap_analysis.md` | Honest inventory of what's built vs what needs to be built, with build order |
| `10_build_now.md` | **Start here to build** — immediate delivery plan, week by week, with full code specs |

---

## The One-Paragraph Summary

The user states a scientific hypothesis in natural language. The system parses it
into a structured experiment config, gathers context from literature and prior runs,
dispatches an AutoML experiment to train a surrogate model, interprets results in
plain English, runs an acquisition function to propose the next physical experiment,
and waits for user approval. This outer loop repeats — with each iteration informed
by a growing scientific memory of what's been learned — until the hypothesis is
resolved or the user stops it. At session end, the system generates a shareable
Markdown report and a reproducible Databricks notebook.

---

## New Files to Build (summary)

```
agents/agents/loop_agent.py          ← Loop Orchestrator (state machine)
agents/tools/hypothesis_tools.py     ← NL → ExperimentConfig
agents/tools/analysis_tools.py       ← AutoML results → plain English findings
agents/tools/acquisition_tools.py    ← Acquisition function (LLM Phase 1, BO Phase 2)
agents/tools/memory_tools.py         ← Scientific memory read/write
agents/tools/report_tools.py         ← Markdown report + Databricks notebook generator
```

**Modified files:**
```
agents/agents/root_agent.py          ← Add loop routing rule + LoopOrchestratorAgent
agents/config.py                     ← Add MEMORY_TABLE, SESSION_TABLE, EXPLORE_RATIO
agents/tools/__init__.py             ← Import new tool modules
agents/requirements_agents.txt       ← Add scikit-optimize (Phase 2)
```

**New infrastructure:**
```
dev_europa.gold_roses.research_findings    ← scientific memory Delta table
dev_europa.gold_roses.research_sessions    ← session tracking Delta table
dev_europa.gold_roses.research_dead_ends   ← known bad regions Delta table
```

---

## Key Architecture Clarification

AutoML and the Acquisition Function play different roles:

- **AutoML** = trains a surrogate model on historical experiment data.
  Answers: "which variables drive the outcome?"

- **Acquisition Function** = searches the surrogate model to find the
  single most informative next experiment to physically run.
  Answers: "what exact conditions should I test next?"

AutoML builds the map. The acquisition function finds the best destination on it.

---

## Start Here

**For a new agent picking up this project**: read `00_MASTER_HANDOFF.md` — it's the
single document that synthesizes everything.

**To start building right now**: go to `10_build_now.md` for exact files,
build order, and code stubs.

**For the big picture first**: `00_MASTER_HANDOFF.md` → `10_build_now.md`.
