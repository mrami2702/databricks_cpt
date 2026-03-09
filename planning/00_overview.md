# Closed-Loop Autonomous Research System — Vision & Overview

## What We're Building

A system where a researcher can state a hypothesis in plain English, and the system
autonomously designs experiments, runs them, analyzes results, updates its beliefs,
and proposes the next step — all while the researcher converses with it naturally.

The key shift: **the researcher operates at the level of goals, not tasks.**
They don't configure AutoML runs. They don't write SQL. They don't read dashboards.
They have a scientific conversation, and the system handles the mechanics.

---

## The Two Loops

This system has two nested loops operating at different levels of abstraction:

```
╔══════════════════════════════════════════════════════════════════════╗
║  OUTER LOOP — The Science Loop                                       ║
║                                                                      ║
║   User NL hypothesis                                                 ║
║        ↓                                                             ║
║   Hypothesis Parser   →  Structured Experiment Config               ║
║        ↓                                                             ║
║   Search Agents       →  Literature + Prior Runs Context            ║
║        ↓                                                             ║
║  ┌─────────────────────────────────────────┐                        ║
║  │  INNER LOOP — AutoML Loop               │                        ║
║  │  Hyperparameter sweep                   │                        ║
║  │  Multi-fidelity pruning                 │                        ║
║  │  Early stopping                         │                        ║
║  │  Best trial selection                   │                        ║
║  └─────────────────────────────────────────┘                        ║
║        ↓                                                             ║
║   Results → MLflow + Scientific Memory                              ║
║        ↓                                                             ║
║   Analysis Agent   →  "What did we learn?"                          ║
║        ↓                                                             ║
║   Acquisition Function  →  "What should we try next?"              ║
║        ↓                                                             ║
║   NL Proposal  →  User (approve / redirect / stop)                 ║
║        ↓                                                             ║
║   [Loop repeats or user pivots]                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Inner Loop (AutoML)**: Optimizes *within* a single experiment setup.
Already built — this is the AutoMLAgent.

**Outer Loop (Science)**: Decides *what experiment to run next* based on accumulated
knowledge. This is the new piece we're building.

---

## The Conversational Interface

The user interacts entirely through natural language:

**Input examples:**
- "I think temperature affects yield more than catalyst concentration. Test this."
- "Focus on samples with high uncertainty in their measurement."
- "Stop exploring that direction. What if we look at pH instead?"
- "What did we learn from the last 5 runs? Show me in plain English."

**Output examples:**
- "Based on the last run, temperature has a 3x larger effect than concentration (p<0.01).
   I recommend we now test the temperature × time interaction — literature suggests
   this is underexplored. Shall I proceed?"
- "Run 4 found an anomaly: yield dropped sharply above 85°C, which contradicts our
   initial hypothesis. I've flagged this for your review before continuing."

The system speaks back in scientific language the researcher understands,
not in model metrics and config files.

---

## Why This Is Faster

| Old way | Closed-loop way |
|---|---|
| Researcher reads results manually | Analysis Agent interprets immediately |
| Researcher decides next experiment | Acquisition Function proposes instantly |
| Researcher writes configs | Hypothesis Parser translates NL |
| Literature review takes days | Search Agents run in parallel with experiments |
| Each run starts from scratch | Scientific Memory transfers knowledge across runs |
| Serial experiment scheduling | Multi-fidelity: fast probes first, escalate only promising paths |

The compounding effect: each iteration that once took a researcher a week
can happen in hours. Over 20 iterations, that's months of research time saved.

---

## The Scientific Memory Advantage

Standard AutoML and MLflow track *what happened*. Our system builds *what was learned*.

The difference:
- MLflow records: "Run 47: accuracy=0.84, lr=1e-3, batch_size=32"
- Scientific Memory records: "High learning rates consistently overfit on this dataset.
  Temperature > 80°C reliably degrades yield. pH effect is negligible below pH 7."

This semantic layer sits above MLflow and is the thing that makes Run 50
meaningfully smarter than Run 1 — not just through numerical optimization
but through accumulated domain understanding.

---

## Scope Boundaries

**In scope:**
- Closed-loop experiment orchestration over Databricks/Unity Catalog data
- AutoML as the experiment execution engine (inner loop)
- Natural language hypothesis specification and result reporting
- Literature-grounded experiment proposals via Scholar Agent
- Cross-run scientific memory

**Out of scope (for now):**
- Physical/wet lab control
- Real-time sensor data ingestion
- Multi-user concurrent research sessions
- Federated learning across organizations

---

## Success Criteria

1. A researcher can specify a hypothesis in one sentence and get a proposed experiment config back
2. AutoML runs execute and results are captured in MLflow automatically
3. The system proposes the next experiment without human intervention
4. A researcher can redirect, stop, or approve the next step conversationally
5. After 5+ runs, the system demonstrably avoids re-exploring regions it already knows about
