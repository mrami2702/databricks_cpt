"""
Local Chat Client — Fine-Tuned Mistral 7B on Databricks

Runs a conversational chat from your terminal. The model stays on the
Databricks cluster; this script just sends messages and prints responses.

Requirements:
    pip install requests

Usage:
    python local_chat.py

Fill in the three config values below before running.
"""

import requests
import json
import time
import sys

# ---------------------------------------------------------------------------
# Configuration — fill these in
# ---------------------------------------------------------------------------
HOST = "<FILL_IN_DATABRICKS_HOST>"          # e.g. "https://adb-1234567890.12.azuredatabricks.net"
TOKEN = "<FILL_IN_PAT>"                     # your Databricks personal access token
CLUSTER_ID = "<FILL_IN_CLUSTER_ID>"         # e.g. "0218-153045-abc123"

# Model paths on the cluster
BASE_MODEL_PATH = "mistralai/Mistral-7B-v0.3"           # original HF base model
CPT_ADAPTER_PATH = "<FILL_IN_CPT_ADAPTER_PATH>"         # e.g. "runs:/<run_id>/model" or Volume path
SFT_ADAPTER_PATH = "<FILL_IN_SFT_ADAPTER_PATH>"         # e.g. "runs:/<run_id>/model" or Volume path
# ---------------------------------------------------------------------------

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def create_context():
    """Create a persistent execution context on the cluster."""
    resp = requests.post(
        f"{HOST}/api/1.2/contexts/create",
        headers=HEADERS,
        json={"clusterId": CLUSTER_ID, "language": "python"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def run_on_cluster(ctx_id, code, timeout=600):
    """Send code to the cluster, wait for it to finish, return the result."""
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


def print_streaming(text, delay=0.03):
    """Print text word-by-word to simulate streaming."""
    for word in text.split():
        print(word, end=" ", flush=True)
        time.sleep(delay)
    print()


def main():
    print("=" * 60)
    print("  Fine-Tuned Mistral 7B — Local Chat Client")
    print("=" * 60)
    print()

    # --- Step 1: Connect and create execution context ---
    print("Connecting to Databricks cluster...")
    try:
        ctx_id = create_context()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)
    print("Connected!\n")

    # --- Step 2: Install dependencies on the cluster ---
    print("Installing dependencies on cluster...")
    install_result = run_on_cluster(ctx_id, """
import subprocess
subprocess.check_call(["pip", "install", "-q", "peft", "bitsandbytes", "accelerate"])
"Dependencies installed"
""")
    if "[ERROR]" in str(install_result):
        print(f"Failed to install dependencies: {install_result}")
        sys.exit(1)
    print("Dependencies installed!\n")

    # --- Step 3: Load model on the cluster (one-time) ---
    print("Loading model on cluster (this takes 1-2 minutes)...")
    load_result = run_on_cluster(ctx_id, f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Step 1: Load base Mistral model
base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_PATH}",
    quantization_config=quantization_config,
    device_map="auto",
)

# Step 2: Load CPT adapter and merge into base
model = PeftModel.from_pretrained(base, "{CPT_ADAPTER_PATH}")
model = model.merge_and_unload()

# Step 3: Load SFT adapter on top of the merged CPT model
model = PeftModel.from_pretrained(model, "{SFT_ADAPTER_PATH}")

tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL_PATH}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_prompt(conversation):
    prompt = ""
    for msg in conversation:
        if msg["role"] == "user":
            prompt += f"Q: {{msg['content']}}\\n"
        elif msg["role"] == "assistant":
            prompt += f"A: {{msg['content']}}\\n\\n"
    # Add the A: prefix so the model knows to start answering
    prompt += "A:"
    return prompt

conversation = []
"Model loaded successfully"
""", timeout=600)

    if "[ERROR]" in str(load_result):
        print(f"Failed to load model: {load_result}")
        sys.exit(1)

    print("Model loaded!\n")
    print("Type your messages below. Type 'quit' to exit.")
    print("-" * 60)

    # --- Step 3: Chat loop ---
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        safe_input = json.dumps(user_input)

        print("\nAssistant: ", end="", flush=True)

        response = run_on_cluster(ctx_id, f"""
conversation.append({{"role": "user", "content": {safe_input}}})

prompt = build_prompt(conversation)
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

resp = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
conversation.append({{"role": "assistant", "content": resp}})
resp
""")

        if "[ERROR]" in str(response):
            print(f"\n{response}")
        else:
            print_streaming(response)


if __name__ == "__main__":
    main()
