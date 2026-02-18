# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive Chat Demo — Fine-Tuned Mistral 7B
# MAGIC
# MAGIC This notebook launches a Gradio chat interface to interact with the
# MAGIC QLoRA fine-tuned Mistral-7B model in real time with streaming responses.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. SFT model (QLoRA adapter) saved from `train_sft_mistral.py`
# MAGIC 2. GPU cluster (same type used for training works fine)
# MAGIC
# MAGIC **Run this AFTER:**
# MAGIC - `train_cpt_mistral.py` (Phase 1)
# MAGIC - `train_sft_mistral.py` (Phase 2)

# COMMAND ----------

# MAGIC %pip install gradio peft bitsandbytes accelerate transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Fill in the paths to your base model and QLoRA adapter below.

# COMMAND ----------

config = {
    "model": {
        # Path to the base model — either the CPT checkpoint or the original HF model
        # Examples:
        #   "/dbfs/mnt/models/cpt_model_mistral"
        #   "mistralai/Mistral-7B-Instruct-v0.2"
        "base_model_path": "<FILL_IN_BASE_MODEL_PATH>",   # <-- FILL THIS IN

        # Path to the QLoRA adapter from SFT training
        # Examples:
        #   "/dbfs/mnt/models/sft_model_mistral"
        #   "/dbfs/mnt/models/sft_model_mistral/checkpoint-300"
        "adapter_path": "<FILL_IN_ADAPTER_PATH>",          # <-- FILL THIS IN
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
    },
    "gradio": {
        "title": "Fine-Tuned Mistral 7B — Chat Demo",
        "description": "QLoRA fine-tuned on domain-specific data. Responses are streamed in real time.",
        "server_port": 7860,
    },
}

print("Configuration loaded")
print(f"  Base model: {config['model']['base_model_path']}")
print(f"  Adapter:    {config['model']['adapter_path']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load the Model and QLoRA Adapter

# COMMAND ----------

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_config = config["model"]

print(f"Loading base model from {model_config['base_model_path']}...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_config["base_model_path"],
    quantization_config=quantization_config,
    device_map="auto",
)

print(f"Loading QLoRA adapter from {model_config['adapter_path']}...")
model = PeftModel.from_pretrained(base_model, model_config["adapter_path"])

tokenizer = AutoTokenizer.from_pretrained(model_config["base_model_path"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded and ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Launch Interactive Chat
# MAGIC
# MAGIC Run the cell below to start the Gradio chat interface.
# MAGIC A chat window will appear in the notebook output where you can
# MAGIC type messages and get streaming responses from the model.

# COMMAND ----------

import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread


def chat(message, history):
    """Generate a streaming response from the fine-tuned model."""
    # Build conversation from history
    conversation = []
    for user_msg, bot_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": bot_msg})
    conversation.append({"role": "user", "content": message})

    # Tokenize
    inputs = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    # Set up streaming
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = config["generation"].copy()
    gen_kwargs["input_ids"] = inputs
    gen_kwargs["streamer"] = streamer

    # Run generation in a background thread so we can yield tokens
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream tokens to the UI as they are generated
    response = ""
    for token in streamer:
        response += token
        yield response

    thread.join()


# Build and launch the Gradio interface
gradio_config = config["gradio"]

demo = gr.ChatInterface(
    fn=chat,
    title=gradio_config["title"],
    description=gradio_config["description"],
    examples=[
        "What can you tell me about this topic?",
        "Summarize the key points.",
        "Explain this in simple terms.",
    ],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
)

demo.launch(server_name="0.0.0.0", server_port=gradio_config["server_port"])
