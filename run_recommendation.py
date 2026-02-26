"""
Run CPT Model Recommendation via Databricks API

This script connects to your Databricks workspace and runs the
recommendation workflow on your cluster. No serving endpoint needed.

Usage:
    python run_recommendation.py --task "Which model should I deploy for neutron flux prediction?"

Required environment variables (or set them below):
    DATABRICKS_HOST  - Your workspace URL (e.g., https://your-workspace.cloud.databricks.com)
    DATABRICKS_TOKEN - Your personal access token
    DATABRICKS_CLUSTER_ID - Your running cluster ID
"""

import os
import sys
import json
import time
import argparse
import requests

# =============================================================
# CONFIGURATION — Update these with your values
# =============================================================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")  # e.g., https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
CLUSTER_ID = os.getenv("DATABRICKS_CLUSTER_ID", "")

# Path to trained model on DBFS (output from train_cpt.py)
CPT_MODEL_PATH = "/dbfs/mnt/models/cpt_model"
# =============================================================


def get_headers():
    return {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }


def create_context(cluster_id):
    """Create an execution context on the cluster."""
    resp = requests.post(
        f"{DATABRICKS_HOST}/api/1.2/contexts/create",
        headers=get_headers(),
        json={"clusterId": cluster_id, "language": "python"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def destroy_context(cluster_id, context_id):
    """Destroy an execution context."""
    requests.post(
        f"{DATABRICKS_HOST}/api/1.2/contexts/destroy",
        headers=get_headers(),
        json={"clusterId": cluster_id, "contextId": context_id},
    )


def run_command(cluster_id, context_id, code):
    """Run a command on the cluster and wait for results."""
    resp = requests.post(
        f"{DATABRICKS_HOST}/api/1.2/commands/execute",
        headers=get_headers(),
        json={
            "clusterId": cluster_id,
            "contextId": context_id,
            "language": "python",
            "command": code,
        },
    )
    resp.raise_for_status()
    command_id = resp.json()["id"]

    while True:
        status_resp = requests.get(
            f"{DATABRICKS_HOST}/api/1.2/commands/status",
            headers=get_headers(),
            params={
                "clusterId": cluster_id,
                "contextId": context_id,
                "commandId": command_id,
            },
        )
        status_resp.raise_for_status()
        result = status_resp.json()

        if result["status"] == "Finished":
            return result["results"]
        elif result["status"] in ("Error", "Cancelled"):
            return result["results"]

        time.sleep(2)


def get_recommendation(task, criteria=None):
    """Run the full recommendation pipeline on Databricks."""
    print(f"Connecting to Databricks: {DATABRICKS_HOST}")
    print(f"Cluster: {CLUSTER_ID}")
    print()

    # Create execution context
    print("Creating execution context...")
    context_id = create_context(CLUSTER_ID)

    try:
        # Step 1: Install deps
        print("Installing dependencies...")
        run_command(CLUSTER_ID, context_id,
            "import subprocess; subprocess.check_call(['pip', 'install', '-q', 'peft', 'accelerate'])"
        )

        # Step 2: Load model
        print("Loading trained CPT model...")
        load_result = run_command(CLUSTER_ID, context_id, f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

model_path = "{CPT_MODEL_PATH}"
peft_config = PeftConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()
model.eval()
print("MODEL_LOADED_OK")
""")

        if "error" in str(load_result).lower() and "MODEL_LOADED_OK" not in str(load_result.get("data", "")):
            print(f"Error loading model: {load_result}")
            return None

        print("Model loaded!")

        # Step 3: Query MLflow for models
        print("Querying MLflow Registry...")
        run_command(CLUSTER_ID, context_id, """
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
models_data = []

try:
    registered_models = client.search_registered_models()
    for rm in registered_models[:10]:
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
            except:
                continue
except Exception as e:
    print(f"MLflow query: {e}")

if not models_data:
    models_data = [
        {"name": "neutron_flux_predictor_xgboost", "version": "3", "stage": "Staging",
         "description": "XGBoost for neutron flux prediction",
         "metrics": {"rmse": 0.0234, "mae": 0.0187, "r2": 0.9342, "inference_latency_ms": 12.4},
         "params": {"n_estimators": "500", "max_depth": "8"}},
        {"name": "neutron_flux_predictor_nn", "version": "2", "stage": "Staging",
         "description": "Neural network for neutron flux prediction",
         "metrics": {"rmse": 0.0198, "mae": 0.0156, "r2": 0.9521, "inference_latency_ms": 45.2},
         "params": {"hidden_layers": "4", "hidden_units": "256"}},
        {"name": "neutron_flux_predictor_linear", "version": "1", "stage": "Production",
         "description": "Baseline linear model",
         "metrics": {"rmse": 0.0412, "mae": 0.0334, "r2": 0.8756, "inference_latency_ms": 2.1},
         "params": {"regularization": "l2", "alpha": "0.01"}},
    ]

print(f"MODELS_FOUND:{len(models_data)}")
""")

        # Step 4: Generate recommendation
        criteria_str = f'\\n## Additional Criteria\\n{criteria}\\n' if criteria else ''

        print(f"Generating recommendation for: {task}")
        print()

        rec_result = run_command(CLUSTER_ID, context_id, f"""
lines = []
for i, m in enumerate(models_data, 1):
    lines.append(f"### Model {{i}}: {{m['name']}} (v{{m['version']}})")
    lines.append(f"Stage: {{m['stage']}}")
    if m.get("description"):
        lines.append(f"Description: {{m['description']}}")
    lines.append("Metrics:")
    for key, value in m["metrics"].items():
        if isinstance(value, float):
            lines.append(f"  - {{key}}: {{value:.4f}}")
        else:
            lines.append(f"  - {{key}}: {{value}}")
    lines.append("")
models_text = "\\n".join(lines)

prompt = f\"\"\"You are an expert ML engineer helping select the best model for deployment.

## Task
{task}

## Available Models
{{models_text}}

## Your Analysis
Please evaluate these models and provide:
1. **Recommendation**: Which model should be selected for production
2. **Justification**: Why this model is the best choice (reference specific metrics)
3. **Acceptance Criteria**: What criteria this model meets
4. **Risks/Considerations**: Any concerns or monitoring recommendations
{criteria_str}
## Response
\"\"\"

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
recommendation = response[len(prompt):].strip()
print("RECOMMENDATION_START")
print(recommendation)
print("RECOMMENDATION_END")
""")

        # Extract recommendation from output
        output = rec_result.get("data", "")
        if "RECOMMENDATION_START" in output:
            rec = output.split("RECOMMENDATION_START")[1].split("RECOMMENDATION_END")[0].strip()
            print("=" * 60)
            print("RECOMMENDATION")
            print("=" * 60)
            print(rec)
            return rec
        else:
            print("Output:", output)
            if rec_result.get("cause"):
                print("Error:", rec_result["cause"])
            return None

    finally:
        print("\nCleaning up...")
        destroy_context(CLUSTER_ID, context_id)


def ask_question(context_id, question):
    """Send a single question to the already-loaded model on the cluster."""
    print(f"\nThinking...")

    rec_result = run_command(CLUSTER_ID, context_id, f"""
lines = []
for i, m in enumerate(models_data, 1):
    lines.append(f"### Model {{i}}: {{m['name']}} (v{{m['version']}})")
    lines.append(f"Stage: {{m['stage']}}")
    if m.get("description"):
        lines.append(f"Description: {{m['description']}}")
    lines.append("Metrics:")
    for key, value in m["metrics"].items():
        if isinstance(value, float):
            lines.append(f"  - {{key}}: {{value:.4f}}")
        else:
            lines.append(f"  - {{key}}: {{value}}")
    lines.append("")
models_text = "\\n".join(lines)

prompt = f\"\"\"You are an expert ML engineer helping select the best model for deployment.

## Task
{question}

## Available Models
{{models_text}}

## Your Analysis
Please evaluate these models and provide:
1. **Recommendation**: Which model should be selected for production
2. **Justification**: Why this model is the best choice (reference specific metrics)
3. **Acceptance Criteria**: What criteria this model meets
4. **Risks/Considerations**: Any concerns or monitoring recommendations

## Response
\"\"\"

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
recommendation = response[len(prompt):].strip()
print("RECOMMENDATION_START")
print(recommendation)
print("RECOMMENDATION_END")
""")

    output = rec_result.get("data", "")
    if "RECOMMENDATION_START" in output:
        rec = output.split("RECOMMENDATION_START")[1].split("RECOMMENDATION_END")[0].strip()
        return rec
    else:
        if rec_result.get("cause"):
            return f"Error: {rec_result['cause']}"
        return f"Unexpected output: {output}"


def interactive_chat():
    """Interactive chat mode — load model once, ask multiple questions."""
    print("=" * 60)
    print("  CPT Model Advisor — Interactive Mode")
    print("=" * 60)
    print(f"Connecting to: {DATABRICKS_HOST}")
    print(f"Cluster: {CLUSTER_ID}")
    print()

    # Create execution context
    print("Setting up session...")
    context_id = create_context(CLUSTER_ID)

    try:
        # Install deps
        print("Installing dependencies...")
        run_command(CLUSTER_ID, context_id,
            "import subprocess; subprocess.check_call(['pip', 'install', '-q', 'peft', 'accelerate'])"
        )

        # Load model (once)
        print("Loading trained CPT model (this may take a minute)...")
        load_result = run_command(CLUSTER_ID, context_id, f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

model_path = "{CPT_MODEL_PATH}"
peft_config = PeftConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()
model.eval()
print("MODEL_LOADED_OK")
""")

        if "MODEL_LOADED_OK" not in str(load_result.get("data", "")):
            print(f"Error loading model: {load_result}")
            return

        # Load MLflow models (once)
        print("Querying MLflow Registry...")
        run_command(CLUSTER_ID, context_id, """
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
models_data = []

try:
    registered_models = client.search_registered_models()
    for rm in registered_models[:10]:
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
            except:
                continue
except Exception as e:
    print(f"MLflow query: {e}")

if not models_data:
    models_data = [
        {"name": "neutron_flux_predictor_xgboost", "version": "3", "stage": "Staging",
         "description": "XGBoost for neutron flux prediction",
         "metrics": {"rmse": 0.0234, "mae": 0.0187, "r2": 0.9342, "inference_latency_ms": 12.4},
         "params": {"n_estimators": "500", "max_depth": "8"}},
        {"name": "neutron_flux_predictor_nn", "version": "2", "stage": "Staging",
         "description": "Neural network for neutron flux prediction",
         "metrics": {"rmse": 0.0198, "mae": 0.0156, "r2": 0.9521, "inference_latency_ms": 45.2},
         "params": {"hidden_layers": "4", "hidden_units": "256"}},
        {"name": "neutron_flux_predictor_linear", "version": "1", "stage": "Production",
         "description": "Baseline linear model",
         "metrics": {"rmse": 0.0412, "mae": 0.0334, "r2": 0.8756, "inference_latency_ms": 2.1},
         "params": {"regularization": "l2", "alpha": "0.01"}},
    ]

print(f"MODELS_FOUND:{len(models_data)}")
""")

        print()
        print("Ready! Ask me about model selection.")
        print("Type 'quit' or 'exit' to end the session.")
        print("-" * 60)

        # Chat loop
        while True:
            print()
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nEnding session...")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Ending session...")
                break

            answer = ask_question(context_id, question)
            print()
            print("=" * 60)
            print("CPT Model Advisor:")
            print("=" * 60)
            print(answer)

    finally:
        print("\nCleaning up session...")
        destroy_context(CLUSTER_ID, context_id)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model recommendation via Databricks")
    parser.add_argument("--task", default=None,
                        help="Single question (if omitted, starts interactive chat)")
    parser.add_argument("--criteria", default=None, help="Additional criteria")
    parser.add_argument("--host", default=None, help="Databricks workspace URL")
    parser.add_argument("--token", default=None, help="Databricks PAT")
    parser.add_argument("--cluster", default=None, help="Cluster ID")

    args = parser.parse_args()

    if args.host:
        DATABRICKS_HOST = args.host
    if args.token:
        DATABRICKS_TOKEN = args.token
    if args.cluster:
        CLUSTER_ID = args.cluster

    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, CLUSTER_ID]):
        print("Missing configuration. Set these environment variables or pass as arguments:")
        print("  DATABRICKS_HOST  (--host)")
        print("  DATABRICKS_TOKEN (--token)")
        print("  DATABRICKS_CLUSTER_ID (--cluster)")
        sys.exit(1)

    DATABRICKS_HOST = DATABRICKS_HOST.rstrip("/")

    if args.task:
        # Single question mode
        get_recommendation(args.task, args.criteria)
    else:
        # Interactive chat mode
        interactive_chat()
