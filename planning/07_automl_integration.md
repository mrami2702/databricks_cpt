# AutoML Integration — How the Inner Loop Fits

The AutoML Agent is the experiment execution engine.
This doc specifies exactly how the Loop Orchestrator interfaces with it.

---

## Role in the System

```
Loop Orchestrator
      │
      │  ExperimentConfig (structured)
      ▼
  AutoMLAgent
      │  runs Databricks AutoML
      │  logs to MLflow
      ▼
  AutoMLResult (structured)
      │
      ▼
  Analysis Agent
```

The AutoML Agent is a black box from the loop's perspective:
- Input: what to predict, from what data, with what constraints
- Output: best model metrics, feature importances, MLflow run ID

The loop doesn't care about hyperparameter configurations or trial internals.

---

## Config Translation

The Hypothesis Parser produces an ExperimentConfig.
The Loop Orchestrator translates this into AutoML Agent's expected format.

### ExperimentConfig → AutoML Input

```python
def translate_to_automl_input(config: ExperimentConfig) -> dict:
    return {
        "table": config.table_name,              # "dev_europa.gold_roses.analytical_results"
        "target_col": config.target_variable,    # "yield_pct"
        "feature_cols": config.independent_variables,  # ["temperature_celsius", "catalyst_type"]
        "problem_type": config.problem_type,     # "regression" | "classification"
        "primary_metric": config.primary_metric, # "r2" | "accuracy" | "f1"
        "timeout_minutes": config.max_runtime_minutes,  # 20
        "sample_filter": config.sample_filter    # "measurement_type = 'oxide'" or None
    }
```

### AutoML Output → AutoMLResult

```python
@dataclass
class AutoMLResult:
    run_id: str                          # MLflow run ID
    best_metric_value: float            # e.g. R² = 0.84
    metric_name: str                    # "r2", "accuracy", etc.
    feature_importances: Dict[str, float]  # {"temperature_celsius": 0.67, ...}
    best_model_type: str                # "RandomForest", "XGBoost", etc.
    n_trials: int                       # how many AutoML trials ran
    runtime_seconds: float
    error: Optional[str]               # if run failed
    warnings: List[str]               # e.g. "class imbalance detected"
    experiment_url: str               # MLflow UI link
```

---

## Multi-Fidelity Strategy (Phase 3)

In Phase 3, the loop uses a multi-fidelity approach to save compute:

```
Level 1 — Fast Probe (2-5 min)
  Small sample (10% of data)
  Short timeout (3 min)
  Goal: Is there ANY signal here?
  If R² < 0.05: dead end → skip to next proposal
  If R² > 0.05: escalate to Level 2

Level 2 — Medium Run (15-30 min)
  Full dataset
  Medium timeout (20 min)
  Goal: How strong is the effect? Which features matter?
  If feature importances confirm hypothesis: results are usable
  If marginal: optionally escalate to Level 3

Level 3 — Full Run (1-2 hrs)
  Full dataset, extended timeout
  Ensemble models, cross-validation
  Goal: Publication-quality result
  Only run when user specifically requests or hypothesis is very promising
```

**Default for Phase 1-2**: Level 2 only (reliable signal, reasonable runtime).
**Phase 3**: Automatic Level 1 probe before every Level 2 run.

---

## AutoML Configuration Recommendations

Based on the gold_roses dataset (32 tables, ~45K rows, V100 16GB):

| Setting | Phase 1-2 default | Phase 3 probe | Phase 3 full |
|---|---|---|---|
| timeout_minutes | 20 | 3 | 60 |
| max_trials | auto | 5 | auto |
| sample_fraction | 1.0 | 0.1 | 1.0 |
| primary_metric | r2 (regression) | r2 | r2 |
| cross_validation | 3-fold | none | 5-fold |
| exclude_frameworks | none | RF only | none |

---

## AutoML Result Quality Checks

The Analysis Agent performs these checks on every AutoML result
before interpreting it:

```python
def quality_checks(result: AutoMLResult, config: ExperimentConfig) -> List[str]:
    warnings = []

    # 1. Did the model learn anything?
    if result.best_metric_value < 0.05:
        warnings.append("WEAK_SIGNAL: R² < 0.05. Model barely outperforms baseline.")

    # 2. Is there a dominant single feature? (possible data leakage)
    max_importance = max(result.feature_importances.values())
    if max_importance > 0.95:
        warnings.append(f"LEAKAGE_RISK: Single feature dominates ({max_importance:.0%}). Check for data leakage.")

    # 3. Did AutoML have enough trials?
    if result.n_trials < 5:
        warnings.append("FEW_TRIALS: Less than 5 trials ran. Results may not be optimal.")

    # 4. Runtime too short?
    if result.runtime_seconds < 60:
        warnings.append("SHORT_RUN: Run completed in <1 min. May have errored or had trivial data.")

    # 5. Pass-through AutoML warnings
    warnings.extend(result.warnings)

    return warnings
```

Any LEAKAGE_RISK or WEAK_SIGNAL warning → surfaces to user before proposing next step.
Other warnings → included in iteration report but don't block continuation.

---

## Feature Importance Mapping

AutoML returns importance scores keyed by actual column names.
The system maps these back to user-facing variable names.

```python
def map_to_user_labels(
    importances: Dict[str, float],
    config: ExperimentConfig
) -> Dict[str, float]:
    """
    Maps "temperature_celsius" → "temperature" if user used that term.
    Aggregates importance of related columns (e.g. temp_min + temp_max → temperature).
    """
    user_label_map = config.column_to_user_label  # built during hypothesis parsing
    result = {}
    for col, importance in importances.items():
        label = user_label_map.get(col, col)  # fallback to column name
        result[label] = result.get(label, 0) + importance
    return result
```

This ensures the Analysis Agent can talk to the user using their own vocabulary.

---

## MLflow Connection

Every AutoML run is logged to the Databricks MLflow workspace.
The loop maintains run_ids for full traceability.

**What gets logged (by AutoML Agent):**
- All trial hyperparameters
- Best trial metrics
- Feature importances as MLflow tags
- Model artifacts

**What the loop additionally logs (via MLflow client):**
- Loop iteration number as a run tag
- Original hypothesis text as a tag
- Memory findings used as context (JSON tag)
- Loop session ID for grouping related runs

This means a researcher can open MLflow and see the entire research session's
runs grouped together, with the hypothesis that motivated each run.

**MLflow experiment naming convention:**
```
Experiment: closed_loop_[table_name]_[session_date]
  Run: iteration_1_[timestamp]
  Run: iteration_2_[timestamp]
  ...
```
