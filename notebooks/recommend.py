# Databricks notebook source
# MAGIC %md
# MAGIC # CPT Model Advisor — Get Recommendations
# MAGIC
# MAGIC This notebook loads the trained CPT model and generates model recommendations.
# MAGIC Run this after `train_cpt.py` has completed.

# COMMAND ----------

# MAGIC %pip install peft accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Path to your trained CPT model (output from train_cpt.py)
CPT_MODEL_PATH = "/dbfs/mnt/models/cpt_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Trained Model

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

print(f"Loading CPT model from: {CPT_MODEL_PATH}")

peft_config = PeftConfig.from_pretrained(CPT_MODEL_PATH)
base_model_name = peft_config.base_model_name_or_path
print(f"Base model: {base_model_name}")

tokenizer = AutoTokenizer.from_pretrained(CPT_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, CPT_MODEL_PATH)
model = model.merge_and_unload()
model.eval()

print("Model loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query MLflow Registry
# MAGIC
# MAGIC Pull real model metrics from your MLflow Registry. If no models are registered
# MAGIC yet, sample data is used instead.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def get_models_from_registry(limit=10):
    """Pull models and their metrics from MLflow Registry."""
    models_data = []

    try:
        registered_models = client.search_registered_models()
        for rm in registered_models[:limit]:
            versions = client.get_latest_versions(rm.name)
            for v in versions:
                try:
                    run = client.get_run(v.run_id)
                    models_data.append({
                        "name": rm.name,
                        "version": v.version,
                        "stage": v.current_stage,
                        "description": v.description or "",
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"Could not query MLflow Registry: {e}")

    return models_data


def get_sample_models():
    """Sample model data for testing."""
    return [
        {
            "name": "neutron_flux_predictor_xgboost",
            "version": "3",
            "stage": "Staging",
            "description": "XGBoost model for neutron flux prediction in reactor simulations",
            "metrics": {"rmse": 0.0234, "mae": 0.0187, "r2": 0.9342, "inference_latency_ms": 12.4},
            "params": {"n_estimators": "500", "max_depth": "8", "learning_rate": "0.05"},
        },
        {
            "name": "neutron_flux_predictor_nn",
            "version": "2",
            "stage": "Staging",
            "description": "Neural network for neutron flux prediction with higher accuracy",
            "metrics": {"rmse": 0.0198, "mae": 0.0156, "r2": 0.9521, "inference_latency_ms": 45.2},
            "params": {"hidden_layers": "4", "hidden_units": "256", "dropout": "0.2"},
        },
        {
            "name": "neutron_flux_predictor_linear",
            "version": "1",
            "stage": "Production",
            "description": "Baseline linear model currently in production",
            "metrics": {"rmse": 0.0412, "mae": 0.0334, "r2": 0.8756, "inference_latency_ms": 2.1},
            "params": {"regularization": "l2", "alpha": "0.01"},
        },
    ]


# Try real MLflow data first, fall back to sample
models = get_models_from_registry()
if not models:
    print("No models in MLflow Registry — using sample data for demo")
    models = get_sample_models()
else:
    print(f"Found {len(models)} models in MLflow Registry")

# Display what we found
for m in models:
    print(f"  {m['name']} v{m['version']} ({m['stage']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Prompt & Get Recommendation

# COMMAND ----------

def format_models_for_prompt(models):
    """Format model metadata into a structured prompt section."""
    lines = []
    for i, m in enumerate(models, 1):
        lines.append(f"### Model {i}: {m['name']} (v{m['version']})")
        lines.append(f"Stage: {m['stage']}")
        if m.get("description"):
            lines.append(f"Description: {m['description']}")
        lines.append("Metrics:")
        for key, value in m["metrics"].items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")
        if m.get("params"):
            lines.append("Parameters:")
            for key, value in list(m["params"].items())[:10]:
                lines.append(f"  - {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def get_recommendation(task_description, models, criteria=None):
    """Generate a recommendation using the CPT model."""
    models_text = format_models_for_prompt(models)

    prompt = f"""You are an expert ML engineer helping select the best model for deployment.

## Task
{task_description}

## Available Models
{models_text}

## Your Analysis
Please evaluate these models and provide:
1. **Recommendation**: Which model should be selected for production
2. **Justification**: Why this model is the best choice (reference specific metrics)
3. **Acceptance Criteria**: What criteria this model meets
4. **Risks/Considerations**: Any concerns or monitoring recommendations
"""

    if criteria:
        prompt += f"\n## Additional Criteria to Consider\n{criteria}\n"

    prompt += "\n## Response\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask a Question
# MAGIC
# MAGIC Change the task description below and re-run this cell to get different recommendations.

# COMMAND ----------

task = "Predict neutron flux distribution in reactor core simulations for safety analysis"

# Optional: add specific criteria
criteria = None  # e.g. "Inference latency must be under 50ms"

print("=" * 60)
print("TASK:", task)
print("=" * 60)

recommendation = get_recommendation(task, models, criteria)

print("\nRECOMMENDATION:")
print("-" * 60)
print(recommendation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try More Questions
# MAGIC
# MAGIC Run each cell below for different recommendation scenarios.

# COMMAND ----------

# Question 2
recommendation = get_recommendation(
    "Which model should I use for real-time safety monitoring where false negatives are critical?",
    models,
)
print(recommendation)

# COMMAND ----------

# Question 3
recommendation = get_recommendation(
    "Compare these models for batch processing where latency is not a concern but accuracy is paramount",
    models,
)
print(recommendation)
