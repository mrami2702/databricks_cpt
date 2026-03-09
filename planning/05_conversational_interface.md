# Conversational Interface Design

The user never sees configs, dashboards, or metric tables.
They have a scientific conversation. This doc specifies what that looks like.

---

## Interaction Patterns

### Starting a Research Session

The system detects research session intent and switches into loop mode.

**Trigger phrases** (detected by root_agent routing):
- "Test whether X affects Y"
- "I think [hypothesis] — investigate this"
- "Run a research session on [topic]"
- "Systematically explore [question]"
- "Start a closed-loop experiment"

**System response:**
```
Starting research session.

Hypothesis: [parsed hypothesis in plain English]
Target: [variable]
Data: [table name] ([N] rows available)
Independent variables: [list]
Problem type: [regression/classification]

I'll run the first experiment and report back.
[if clarification needed]: Before I start — did you mean [option A] or [option B]?
```

---

### Mid-Session: Iteration Report

After each AutoML run completes:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION [N] — [table] → [target]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT WE FOUND
[3-5 sentence plain English analysis]

Key findings:
• [finding 1]
• [finding 2]
• [finding 3]

Hypothesis: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED / ANOMALY DETECTED]
Model fit: [metric name] = [value] ([interpretation: strong/moderate/weak fit])

[IF ANOMALY]:
⚠ Anomaly: [description]. Recommending pause before continuing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROPOSED NEXT STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Proposal in plain English]

Why this direction:
• [reason 1 — from analysis]
• [reason 2 — from literature / memory]

Estimated runtime: [N] minutes

Options:
  [1] Proceed with proposed experiment
  [2] [Alternative direction]
  [3] [Second alternative]
  [4] Stop and summarize findings
```

---

### User Responses to Proposals

The system handles these gracefully:

| User says | System does |
|---|---|
| "Yes" / "Proceed" / "Go ahead" | Runs proposed experiment |
| "Try [alternative]" | Parses new direction, runs it |
| "Why did you propose that?" | Explains reasoning in detail |
| "What did we learn so far?" | Summarizes all iterations to date |
| "Stop" / "Done" | Generates session summary, closes loop |
| "What's the best finding so far?" | Pulls top finding from memory |
| "Show me the numbers" | Returns metric table for current iteration |
| "Can you explain [concept]?" | Answers in domain context, then re-proposes |

---

### Session Summary (on stop)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH SESSION SUMMARY
[N] iterations | [total runtime]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORIGINAL HYPOTHESIS
[user's original question]

WHAT WE FOUND

Strong predictors:
• [variable 1]: [importance] — [plain English explanation]
• [variable 2]: [importance] — [plain English explanation]

Weak / no effect:
• [variable 3], [variable 4]

Notable discoveries:
• [anomaly or threshold found]
• [unexpected finding]

Hypothesis verdict: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]

OPEN QUESTIONS
• [question 1 — worth investigating further]
• [question 2]

NEXT STEPS (suggested)
• [recommendation 1]
• [recommendation 2]

All findings saved to memory. MLflow runs: [run IDs]
```

---

## Clarification Handling

When the hypothesis parser is uncertain (confidence < 0.7), the system asks
a targeted clarification question before starting:

**Scenario: ambiguous target variable**
```
User: "Investigate how conditions affect sample quality"

System: Before I start, I need to clarify:
  What should I use as the measure of "sample quality"?
  • yield_pct (current yield percentage)
  • purity_score (analytical purity measure)
  • response_value (instrument response value)
  • Other — describe it
```

**Scenario: table not specified**
```
System: Which dataset should I run this on?
  • gold_roses (32 tables, most recent data)
  • [table A] — [N rows, [column list]]
  • [table B] — [N rows, [column list]]
  • I'll find the best table automatically
```

**Scenario: columns not found**
```
System: I couldn't find "reactor temperature" as an exact column name.
  Did you mean:
  • temp_celsius (in analytical_results)
  • reaction_temp (in sample_properties)
  • temperature_setpoint (in process_conditions)
```

---

## Autonomy Controls

Users can tune how autonomous the system is:

**Inline commands** (recognized mid-conversation):
- "Be more exploratory" → raises explore_ratio to 0.5
- "Focus on what's working" → lowers explore_ratio to 0.05
- "Run the next 3 iterations automatically" → skips user gate for 3 runs
- "Always ask me before running" → enforces user gate on every iteration
- "Stop if R² drops below 0.3" → adds termination criterion
- "Cap compute at 2 hours total" → sets session budget

---

## Error Handling — User-Facing Messages

| Error | What user sees |
|---|---|
| AutoML run fails | "The experiment encountered an error: [brief reason]. Suggest: [fix or retry]" |
| Column not found in table | "I couldn't find [column] in [table]. Available columns: [list]. Did you mean [suggestion]?" |
| AutoML finds no signal | "The model found very weak predictive signal (R²=0.02). The hypothesis may not hold in this data, or we may need more samples / different features." |
| Cluster offline | "The compute cluster is not running. Should I start it? (This takes ~5 minutes)" |
| Memory store unavailable | "I can't save findings right now — memory store is unavailable. I'll report results verbally. Session memory won't persist after this conversation." |

---

## What the User Never Sees

These are handled invisibly by the system:
- AutoML hyperparameter configs
- MLflow run IDs (mentioned only in summary, never during session)
- SQL queries
- Cluster start/stop operations
- Feature engineering decisions
- Train/validation splits
- Column type detection

The user should feel like they're talking to a smart research collaborator,
not configuring a machine learning pipeline.
