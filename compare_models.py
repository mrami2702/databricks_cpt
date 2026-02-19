"""
Model Comparison — Base vs CPT vs SFT

Sends the same question to all three model stages and displays
responses side by side to show the progression of fine-tuning.

Requirements:
    pip install requests

Usage:
    python compare_models.py
"""

import requests
import json
import time
import sys

# ---------------------------------------------------------------------------
# Configuration — fill these in (same as local_chat.py)
# ---------------------------------------------------------------------------
HOST = "<FILL_IN_DATABRICKS_HOST>"          # e.g. "https://adb-1234567890.12.azuredatabricks.net"
TOKEN = "<FILL_IN_PAT>"                     # your Databricks personal access token
CLUSTER_ID = "<FILL_IN_CLUSTER_ID>"         # e.g. "0218-153045-abc123"

# Model paths on the cluster
BASE_MODEL_PATH = "mistralai/Mistral-7B-v0.3"           # original HF base model
CPT_ADAPTER_PATH = "<FILL_IN_CPT_ADAPTER_PATH>"         # e.g. Volume path to CPT adapter
SFT_ADAPTER_PATH = "<FILL_IN_SFT_ADAPTER_PATH>"         # e.g. Volume path to SFT adapter
# ---------------------------------------------------------------------------

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def create_context():
    resp = requests.post(
        f"{HOST}/api/1.2/contexts/create",
        headers=HEADERS,
        json={"clusterId": CLUSTER_ID, "language": "python"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def run_on_cluster(ctx_id, code, timeout=600):
    resp = requests.post(
        f"{HOST}/api/1.2/commands/execute",
        headers=HEADERS,
        json={
            "clusterId": CLUSTER_ID,
            "contextId": ctx_id,
            "language": "python",
            "command": code,
        },
    )
    resp.raise_for_status()
    cmd_id = resp.json()["id"]

    elapsed = 0
    while elapsed < timeout:
        status = requests.get(
            f"{HOST}/api/1.2/commands/status",
            headers=HEADERS,
            params={
                "clusterId": CLUSTER_ID,
                "contextId": ctx_id,
                "commandId": cmd_id,
            },
        ).json()

        if status["status"] == "Finished":
            results = status.get("results", {})
            if results.get("resultType") == "error":
                return f"[ERROR] {results.get('summary', 'Unknown error')}"
            return results.get("data", "")

        if status["status"] in ("Cancelled", "Error"):
            return f"[ERROR] Command {status['status']}"

        time.sleep(2)
        elapsed += 2

    return "[ERROR] Command timed out"


def main():
    print("=" * 70)
    print("  Model Comparison — Base vs CPT vs SFT")
    print("=" * 70)
    print()

    # --- Connect ---
    print("Connecting to Databricks cluster...")
    try:
        ctx_id = create_context()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)
    print("Connected!\n")

    # --- Install dependencies ---
    print("Installing dependencies...")
    run_on_cluster(ctx_id, """
import subprocess
subprocess.check_call(["pip", "install", "-q", "peft", "bitsandbytes", "accelerate", "transformers", "tokenizers"])
"done"
""")
    print("Done!\n")

    # --- Load all three models ---
    print("Loading all three models (this takes 2-4 minutes)...")
    load_result = run_on_cluster(ctx_id, f"""
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

try:
    tokenizer = AutoTokenizer.from_pretrained("{CPT_ADAPTER_PATH}", trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL_PATH}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Model 1: Base Mistral (no fine-tuning) ---
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_PATH}",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model = prepare_model_for_kbit_training(base_model)

# --- Model 2: CPT only (base + CPT adapter merged) ---
print("Loading CPT model...")
cpt_model = PeftModel.from_pretrained(base_model, "{CPT_ADAPTER_PATH}")
cpt_model = cpt_model.merge_and_unload()
cpt_model.eval()

# --- Model 3: SFT (CPT merged + SFT adapter) ---
print("Loading SFT model...")
sft_model = PeftModel.from_pretrained(cpt_model, "{SFT_ADAPTER_PATH}")
sft_model.eval()

# Keep references to all three
# Note: base_model weights were modified by the CPT merge, so we need
# a separate copy for true base comparison. Reload it.
print("Reloading clean base model for comparison...")
clean_base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_PATH}",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
clean_base.eval()

models = {{
    "Base Mistral": clean_base,
    "CPT": cpt_model,
    "SFT": sft_model,
}}

def generate_response(model, question):
    prompt = "<s>[INST] " + question + " [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in full_response:
        return full_response.split("[/INST]")[-1].strip()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("All models loaded!")
"ready"
""", timeout=600)

    if "[ERROR]" in str(load_result):
        print(f"Failed to load models: {load_result}")
        sys.exit(1)

    print("All models loaded!\n")
    print("Type a question and see how all three models respond.")
    print("Type 'quit' to exit.")
    print("-" * 70)

    # --- Query loop ---
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        safe_input = json.dumps(user_input)

        # Query all three models
        result = run_on_cluster(ctx_id, f"""
import json

question = {safe_input}
responses = {{}}
for name, m in models.items():
    responses[name] = generate_response(m, question)

json.dumps(responses)
""")

        if "[ERROR]" in str(result):
            print(f"\n{result}")
            continue

        # The result may come back with extra quotes from the execution API
        try:
            parsed = json.loads(result)
            # If it was double-encoded (string within string), parse again
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            responses = parsed
        except (json.JSONDecodeError, TypeError):
            print(f"\nUnexpected response format: {result}")
            continue

        # Display each model's response in its own formatted block
        print()
        print(f"{'=' * 70}")
        print(f"  QUESTION: {user_input}")
        print(f"{'=' * 70}")

        for model_name, response in responses.items():
            print()
            print(f"  --- {model_name} ---")
            print()
            # Word wrap the response with indentation
            words = response.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(line)
                    line = "    " + word
                else:
                    line += (" " if line.strip() else "") + word
            if line.strip():
                print(line)

        print()
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
