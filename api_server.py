"""
FastAPI Backend — Serves the fine-tuned Mistral 7B model via Databricks.

Your React frontend calls this API, which forwards requests to the
Databricks cluster for inference.

Requirements:
    pip install fastapi uvicorn requests

Usage:
    python api_server.py

Then your React app calls http://localhost:8000/chat, /compare, etc.
"""

import requests as http_requests
import json
import time
import ast
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration — fill these in
# ---------------------------------------------------------------------------
HOST = "<FILL_IN_DATABRICKS_HOST>"          # e.g. "https://adb-1234567890.12.azuredatabricks.net"
TOKEN = "<FILL_IN_PAT>"                     # your Databricks personal access token
CLUSTER_ID = "<FILL_IN_CLUSTER_ID>"         # e.g. "0218-153045-abc123"

BASE_MODEL_PATH = "mistralai/Mistral-7B-v0.3"
CPT_ADAPTER_PATH = "<FILL_IN_CPT_ADAPTER_PATH>"
SFT_ADAPTER_PATH = "<FILL_IN_SFT_ADAPTER_PATH>"
# ---------------------------------------------------------------------------

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Stores the execution context ID once connected
cluster_state = {"ctx_id": None, "ready": False}


def create_context():
    resp = http_requests.post(
        f"{HOST}/api/1.2/contexts/create",
        headers=HEADERS,
        json={"clusterId": CLUSTER_ID, "language": "python"},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def run_on_cluster(ctx_id, code, timeout=600):
    resp = http_requests.post(
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
        status = http_requests.get(
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


def parse_result(result):
    """Parse result from execution API — handles JSON and Python repr."""
    if isinstance(result, dict):
        return result
    try:
        parsed = json.loads(result)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(result)
    except Exception:
        return result


def init_cluster():
    """Connect to cluster and load models."""
    print("Connecting to Databricks cluster...")
    cluster_state["ctx_id"] = create_context()
    ctx = cluster_state["ctx_id"]

    print("Installing dependencies...")
    run_on_cluster(ctx, """
import subprocess
subprocess.check_call(["pip", "install", "-q", "peft", "bitsandbytes", "accelerate", "transformers", "tokenizers"])
"done"
""")

    print("Loading models (this takes 2-4 minutes)...")
    result = run_on_cluster(ctx, f"""
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

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_PATH}",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model = prepare_model_for_kbit_training(base_model)

# CPT: base + adapter merged
cpt_model = PeftModel.from_pretrained(base_model, "{CPT_ADAPTER_PATH}")
cpt_model = cpt_model.merge_and_unload()
cpt_model.eval()

# SFT: CPT + SFT adapter
sft_model = PeftModel.from_pretrained(cpt_model, "{SFT_ADAPTER_PATH}")
sft_model.eval()

# Clean base for comparison
clean_base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_PATH}",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
clean_base.eval()

models = {{
    "base": clean_base,
    "cpt": cpt_model,
    "sft": sft_model,
}}

conversation = []

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

def build_prompt(conv):
    prompt = "<s>"
    for msg in conv:
        if msg["role"] == "user":
            prompt += "[INST] " + msg["content"] + " [/INST] "
        elif msg["role"] == "assistant":
            prompt += msg["content"] + "</s><s>"
    return prompt

"ready"
""", timeout=600)

    if "[ERROR]" in str(result):
        raise RuntimeError(f"Failed to load models: {result}")

    cluster_state["ready"] = True
    print("Models loaded! API server ready.")


# --- FastAPI app ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_cluster()
    yield

app = FastAPI(title="Mistral 7B Fine-Tuned API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class CompareRequest(BaseModel):
    question: str


@app.get("/status")
def get_status():
    return {"ready": cluster_state["ready"]}


@app.post("/chat")
def chat(req: ChatRequest):
    if not cluster_state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    ctx = cluster_state["ctx_id"]
    safe_input = json.dumps(req.message)

    result = run_on_cluster(ctx, f"""
conversation.append({{"role": "user", "content": {safe_input}}})

prompt = build_prompt(conversation)
inputs = tokenizer(prompt, return_tensors="pt").to(sft_model.device)

with torch.no_grad():
    outputs = sft_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[/INST]" in full_response:
    resp = full_response.split("[/INST]")[-1].strip()
else:
    resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
conversation.append({{"role": "assistant", "content": resp}})
resp
""")

    if "[ERROR]" in str(result):
        raise HTTPException(status_code=500, detail=result)

    return {"response": result}


@app.post("/compare")
def compare(req: CompareRequest):
    if not cluster_state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    ctx = cluster_state["ctx_id"]
    safe_input = json.dumps(req.question)

    result = run_on_cluster(ctx, f"""
import json
question = {safe_input}
responses = {{}}
for name, m in models.items():
    responses[name] = generate_response(m, question)
json.dumps(responses)
""")

    if "[ERROR]" in str(result):
        raise HTTPException(status_code=500, detail=result)

    parsed = parse_result(result)
    if isinstance(parsed, str):
        # If still a string, try one more parse attempt
        try:
            parsed = json.loads(parsed.replace("'", '"'))
        except Exception:
            raise HTTPException(status_code=500, detail=f"Could not parse model responses: {parsed[:200]}")

    return parsed


@app.post("/chat/reset")
def reset_chat():
    if not cluster_state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    run_on_cluster(cluster_state["ctx_id"], "conversation.clear()\n'reset'")
    return {"status": "conversation reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
