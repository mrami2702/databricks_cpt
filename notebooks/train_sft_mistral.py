# Databricks notebook source
# MAGIC %md
# MAGIC # SFT Training — Mistral-7B on Q&A Pairs
# MAGIC
# MAGIC This notebook fine-tunes the CPT-trained Mistral-7B model on
# MAGIC question-answer pairs generated from your scientific data.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. CPT model saved (fill in path below)
# MAGIC 2. SFT data generated in Unity Catalog (fill in table names below)
# MAGIC
# MAGIC **Run this AFTER:**
# MAGIC - `train_cpt_mistral.py` (Phase 1)
# MAGIC - `generate_sft_data.py` (Phase 2)

# COMMAND ----------

# MAGIC %pip install peft bitsandbytes accelerate trl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    "model": {
        # TODO: Fill in the path to your CPT model
        "cpt_model_path": "",  # e.g. "/dbfs/mnt/models/cpt_model_mistral"
    },
    "data": {
        # TODO: Fill in your Unity Catalog location
        "catalog": "",    # e.g. "dev_europa"
        "schema": "",     # e.g. "gold_roses"
        "sft_table": "",  # e.g. "sft_training_data"
        # Set this to also include MLflow Q&A pairs (when available)
        "mlflow_table": None,  # e.g. "sft_mlflow_data"
        "val_ratio": 0.05,
    },
    "training": {
        "max_seq_length": 1024,
        "num_epochs": 3,
        "max_steps": 300,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    },
    "output": {
        # TODO: Fill in where to save the SFT model
        "output_dir": "",  # e.g. "/dbfs/mnt/models/sft_model_mistral"
        "logging_steps": 10,
        "report_to": "mlflow",
    },
    "mlflow": {
        # TODO: Fill in your MLflow experiment path
        "experiment_name": "",  # e.g. "/Shared/cpt-sft-mistral"
        "run_name": "sft-mistral-7b-v1",
    },
}

print("Configuration loaded")
print(f"  CPT model: {config['model']['cpt_model_path']}")
print(f"  SFT data: {config['data']['catalog']}.{config['data']['schema']}.{config['data']['sft_table']}")
print(f"  Output: {config['output']['output_dir']}")
print(f"  Learning rate: {config['training']['learning_rate']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load SFT Data from Unity Catalog

# COMMAND ----------

from pyspark.sql import functions as F

data_config = config["data"]
sft_table = f"{data_config['catalog']}.{data_config['schema']}.{data_config['sft_table']}"

print(f"Loading SFT data from {sft_table}...")
sft_df = spark.table(sft_table)

# Optionally combine with MLflow Q&A data
if data_config["mlflow_table"]:
    mlflow_table = f"{data_config['catalog']}.{data_config['schema']}.{data_config['mlflow_table']}"
    print(f"Loading MLflow SFT data from {mlflow_table}...")
    mlflow_df = spark.table(mlflow_table)
    sft_df = sft_df.unionByName(mlflow_df)
    print(f"Combined dataset")

total = sft_df.count()
print(f"\nTotal Q&A pairs: {total}")

# Show category breakdown
print("\nBy category:")
category_counts = sft_df.groupBy("category").count().orderBy("category").collect()
for row in category_counts:
    print(f"  {row['category']}: {row['count']}")

# Preview a sample
print("\nSample Q&A pair:")
sample = sft_df.orderBy(F.rand(seed=42)).limit(1).collect()[0]
print(f"  Q: {sample['instruction'][:100]}...")
print(f"  A: {sample['response'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Format Data for Mistral Instruction Tuning
# MAGIC
# MAGIC Mistral uses the `[INST]...[/INST]` format for instruction following.

# COMMAND ----------

# Convert to pandas for HuggingFace Dataset
pdf = sft_df.select("instruction", "response").toPandas()

# Format each row into Mistral's instruction template
def format_for_mistral(row):
    return f"<s>[INST] {row['instruction']} [/INST] {row['response']}</s>"

pdf["text"] = pdf.apply(format_for_mistral, axis=1)

# Train/val split
from datasets import Dataset

dataset = Dataset.from_pandas(pdf[["text"]])
split = dataset.train_test_split(test_size=data_config["val_ratio"], seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"\nSample formatted text:")
print(train_dataset[0]["text"][:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load CPT Model with QLoRA

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

model_path = config["model"]["cpt_model_path"]

print(f"Loading CPT model from: {model_path}")

# First check if the CPT model is a LoRA adapter or merged model
import os
is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if is_peft:
    # CPT model is a LoRA adapter — load base + merge
    print("CPT model is a LoRA adapter, loading base model and merging...")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # Load and merge CPT adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    print(f"Merged CPT adapter from {base_model_name}")
else:
    # CPT model is already merged — load directly
    print("CPT model is a merged model, loading directly...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Model loaded on {model.device}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Apply Fresh LoRA Adapter for SFT

# COMMAND ----------

lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    target_modules=config["lora"]["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Tokenize with Instruction Masking
# MAGIC
# MAGIC We mask the instruction tokens so the model only learns to
# MAGIC generate better responses, not memorize questions.

# COMMAND ----------

max_seq_length = config["training"]["max_seq_length"]

# Tokenize the instruction portion to find where the response starts
INST_END_TOKEN = "[/INST]"

def tokenize_and_mask(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    labels_batch = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        labels = list(input_ids)
        text = examples["text"][i]

        # Find where [/INST] ends in the text
        inst_end_pos = text.find(INST_END_TOKEN)
        if inst_end_pos != -1:
            # Tokenize just the instruction part to find its token length
            inst_part = text[:inst_end_pos + len(INST_END_TOKEN)]
            inst_tokens = tokenizer(inst_part, add_special_tokens=False)["input_ids"]
            mask_len = len(inst_tokens)

            # Mask instruction tokens with -100 (ignored by loss)
            for j in range(min(mask_len, len(labels))):
                labels[j] = -100

        labels_batch.append(labels)

    tokenized["labels"] = labels_batch
    return tokenized

print(f"Tokenizing with instruction masking (max_length={max_seq_length})...")

train_tokenized = train_dataset.map(
    tokenize_and_mask,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=4,
)

val_tokenized = val_dataset.map(
    tokenize_and_mask,
    batched=True,
    remove_columns=val_dataset.column_names,
    num_proc=4,
)

print(f"Tokenized train: {len(train_tokenized)} samples")
print(f"Tokenized val: {len(val_tokenized)} samples")

# Verify masking works
sample = train_tokenized[0]
total_tokens = len(sample["input_ids"])
masked_tokens = sum(1 for l in sample["labels"] if l == -100)
print(f"\nSample: {total_tokens} total tokens, {masked_tokens} masked (instruction), "
      f"{total_tokens - masked_tokens} trainable (response)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Train

# COMMAND ----------

import mlflow
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_config = config["training"]
output_config = config["output"]
mlflow_config = config["mlflow"]

mlflow.set_experiment(mlflow_config["experiment_name"])

training_args = TrainingArguments(
    output_dir=output_config["output_dir"],
    num_train_epochs=training_config["num_epochs"],
    max_steps=training_config["max_steps"],
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    per_device_eval_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    learning_rate=training_config["learning_rate"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    warmup_ratio=training_config["warmup_ratio"],
    optim=training_config["optim"],
    weight_decay=training_config["weight_decay"],
    max_grad_norm=training_config["max_grad_norm"],
    bf16=training_config["bf16"],
    logging_steps=output_config["logging_steps"],
    report_to=output_config["report_to"],
    save_strategy=training_config["save_strategy"],
    save_steps=training_config["save_steps"],
    save_total_limit=training_config["save_total_limit"],
    eval_strategy="steps",
    eval_steps=training_config["save_steps"],
    dataloader_num_workers=4,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Use Seq2Seq collator which handles labels properly
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

with mlflow.start_run(run_name=mlflow_config.get("run_name")):
    mlflow.log_params({
        "base_model": "cpt_model_mistral",
        "training_type": "SFT",
        "learning_rate": training_config["learning_rate"],
        "num_epochs": training_config["num_epochs"],
        "max_steps": training_config["max_steps"],
        "lora_r": config["lora"]["r"],
        "lora_alpha": config["lora"]["lora_alpha"],
        "sft_pairs": len(train_tokenized),
        "val_pairs": len(val_tokenized),
    })

    print("Starting SFT training...")
    print(f"  Training samples: {len(train_tokenized)}")
    print(f"  Validation samples: {len(val_tokenized)}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Max steps: {training_config['max_steps']}")
    print(f"  Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print()

    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    mlflow.log_metrics({
        "final_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save Model

# COMMAND ----------

output_dir = config["output"]["output_dir"]

print(f"Saving SFT model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Verifying save...")
saved_files = os.listdir(output_dir)
print(f"  Files saved: {saved_files}")
print(f"\nSFT model saved to {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Quick Test
# MAGIC
# MAGIC Ask the SFT model a question to verify it responds properly.

# COMMAND ----------

model.eval()

test_questions = [
    "What tables are available in the gold_roses schema?",
    "Are there any missing values in the data?",
    "Give me a quick summary of the gold_roses data for a lab meeting.",
]

for question in test_questions:
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response after [/INST]
    if "[/INST]" in response:
        answer = response.split("[/INST]")[-1].strip()
    else:
        answer = response[len(prompt):].strip()

    print("=" * 60)
    print(f"Q: {question}")
    print(f"A: {answer[:500]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Compare CPT vs SFT Responses
# MAGIC
# MAGIC Load the CPT model and ask the same questions to see the improvement.

# COMMAND ----------

# Load CPT model for comparison
print("Loading CPT model for comparison...")

cpt_base = AutoModelForCausalLM.from_pretrained(
    config["model"]["cpt_model_path"] if not is_peft else peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
if is_peft:
    cpt_model = PeftModel.from_pretrained(cpt_base, config["model"]["cpt_model_path"])
    cpt_model = cpt_model.merge_and_unload()
else:
    cpt_model = cpt_base
cpt_model.eval()

comparison_question = "What tables are available in the gold_roses schema?"
prompt = f"<s>[INST] {comparison_question} [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(cpt_model.device)

# CPT response
with torch.no_grad():
    cpt_out = cpt_model.generate(
        **inputs, max_new_tokens=256, temperature=0.7,
        top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
    )
cpt_response = tokenizer.decode(cpt_out[0], skip_special_tokens=True)
if "[/INST]" in cpt_response:
    cpt_answer = cpt_response.split("[/INST]")[-1].strip()
else:
    cpt_answer = cpt_response[len(prompt):].strip()

# SFT response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    sft_out = model.generate(
        **inputs, max_new_tokens=256, temperature=0.7,
        top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
    )
sft_response = tokenizer.decode(sft_out[0], skip_special_tokens=True)
if "[/INST]" in sft_response:
    sft_answer = sft_response.split("[/INST]")[-1].strip()
else:
    sft_answer = sft_response[len(prompt):].strip()

print("=" * 60)
print(f"QUESTION: {comparison_question}")
print("=" * 60)
print(f"\nCPT MODEL:")
print(cpt_answer[:500])
print(f"\nSFT MODEL:")
print(sft_answer[:500])
print()
print("The SFT model should give a more structured, direct answer.")

# Clean up comparison model
del cpt_model, cpt_base
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Review the test outputs above — does the SFT model answer questions well?
# MAGIC 2. If responses are too short or generic, increase `max_steps` and re-run
# MAGIC 3. If responses are repetitive, reduce `learning_rate` slightly
# MAGIC 4. To add MLflow model selection Q&A later:
# MAGIC    - Run `generate_sft_mlflow.py` to create the data
# MAGIC    - Change config: `"mlflow_table": "sft_mlflow_data"`
# MAGIC    - Change config: `"cpt_model_path"` to point at this SFT model
# MAGIC    - Re-run this notebook
# MAGIC 5. Update `recommend.py` and `run_recommendation.py` to use the new model path:
# MAGIC    `/dbfs/mnt/models/sft_model_mistral`
