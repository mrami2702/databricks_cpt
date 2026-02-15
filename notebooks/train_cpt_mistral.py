# Databricks notebook source
# MAGIC %md
# MAGIC # Continual Pretraining — Mistral 7B on Databricks GPU
# MAGIC
# MAGIC This notebook runs the full CPT pipeline:
# MAGIC 1. Converts tabular data from Unity Catalog into natural language
# MAGIC 2. Trains Mistral-7B-v0.3 with QLoRA
# MAGIC 3. Logs everything to MLflow
# MAGIC
# MAGIC **Cluster requirements:**
# MAGIC - Runtime: Databricks Runtime ML 14.x+
# MAGIC - Instance: GPU-enabled (Standard_NC6s_v3 = 1x V100 16GB)
# MAGIC
# MAGIC **Model:** mistralai/Mistral-7B-v0.3 (7.2B params, only ~1% trained via QLoRA)

# COMMAND ----------

# MAGIC %pip install peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    "model": {
        "name": "mistralai/Mistral-7B-v0.3",
        "use_flash_attention": True,
    },
    "training": {
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "num_epochs": 2,
        "max_steps": -1,
        # Reduced batch size for 7B model on V100 16GB
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,  # Effective batch = 1 * 16 = 16
        "max_seq_length": 2048,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
    },
    "qlora": {
        "enabled": True,
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        # Mistral uses standard transformer attention layers
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "data": {
        "catalog": "dev_europa",
        "schema": "gold_roses",
        "text_column": "text",
        "val_ratio": 0.05,
        "batch_size": 5,  # Rows grouped per training passage
    },
    "output": {
        "output_dir": "/dbfs/mnt/models/cpt_model_mistral",
        "logging_steps": 10,
        "report_to": "mlflow",
    },
    "mlflow": {
        "experiment_name": "continual-pretraining-mistral",
        "run_name": None,
        "tags": {
            "project": "cpt-poc",
            "model_type": "mistral-7b-v0.3",
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Discover Tables in Unity Catalog

# COMMAND ----------

from pyspark.sql import functions as F

catalog = config["data"]["catalog"]
schema = config["data"]["schema"]
text_column = config["data"]["text_column"]

# Exclude our output table if it exists
tables = spark.catalog.listTables(f"{catalog}.{schema}")
table_names = [t.name for t in tables if t.name != "cpt_training_text"]

print(f"Found {len(table_names)} tables in {catalog}.{schema}:")

table_schemas = {}
for name in table_names:
    full_name = f"{catalog}.{schema}.{name}"
    df = spark.table(full_name)
    row_count = df.count()
    table_schemas[name] = {
        "full_name": full_name,
        "schema": df.schema,
        "columns": [(f.name, str(f.dataType)) for f in df.schema.fields],
        "row_count": row_count,
    }
    print(f"  {name}: {row_count} rows, {len(df.schema.fields)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Convert Tabular Data to Natural Language

# COMMAND ----------

import pandas as pd


def find_text_column(spark_df, preferred_column="text"):
    """Find the best text column in a Spark DataFrame."""
    columns = [f.name for f in spark_df.schema.fields]
    string_columns = [
        f.name for f in spark_df.schema.fields
        if str(f.dataType) == "StringType"
    ]
    if preferred_column in columns:
        return preferred_column
    common_names = ["text", "content", "text_content", "body", "document", "doc_text"]
    for name in common_names:
        if name in columns:
            return name
    if string_columns:
        return string_columns[0]
    return None


def row_to_text(row, table_name, columns):
    """Convert a single table row into a natural language description."""
    readable_name = table_name.replace("_", " ").title()

    parts = [f"Record from {readable_name} ({table_name}):"]

    measurements = []
    categories = []
    identifiers = []

    for col_name, col_type in columns:
        value = row[col_name]
        if value is None:
            continue

        clean_name = col_name.replace("_", " ")

        if col_type in ("DoubleType", "FloatType", "DecimalType(38,18)"):
            if isinstance(value, float):
                measurements.append(f"{clean_name}: {value:.6g}")
            else:
                measurements.append(f"{clean_name}: {value}")
        elif col_type in ("IntegerType", "LongType"):
            measurements.append(f"{clean_name}: {value}")
        elif col_type == "StringType":
            if any(hint in col_name.lower() for hint in ["id", "name", "code", "key", "sample"]):
                identifiers.append(f"{clean_name}: {value}")
            else:
                categories.append(f"{clean_name}: {value}")
        elif col_type == "BooleanType":
            categories.append(f"{clean_name}: {'Yes' if value else 'No'}")

    if identifiers:
        parts.append("Identification — " + "; ".join(identifiers) + ".")
    if measurements:
        parts.append("Measurements — " + "; ".join(measurements) + ".")
    if categories:
        parts.append("Properties — " + "; ".join(categories) + ".")

    return " ".join(parts)


# Generate schema overview
total_rows = sum(info["row_count"] for info in table_schemas.values())
schema_overview = f"""Data Schema Overview: {catalog}.{schema}

This schema contains {len(table_schemas)} tables with {total_rows} total records of scientific data.
The data supports data-driven decision making for scientific analysis and model evaluation.

Tables in this schema:
"""
for name, info in table_schemas.items():
    readable = name.replace("_", " ").title()
    schema_overview += f"- {readable} ({name}): {info['row_count']} records, {len(info['columns'])} fields\n"

# Find shared columns
all_columns = {}
for name, info in table_schemas.items():
    for col_name, col_type in info["columns"]:
        if col_name not in all_columns:
            all_columns[col_name] = []
        all_columns[col_name].append(name)

shared = {col: tbls for col, tbls in all_columns.items() if len(tbls) > 1}
if shared:
    schema_overview += "\nShared columns across tables (potential relationships):\n"
    for col, tbls in shared.items():
        schema_overview += f"- '{col}' appears in: {', '.join(tbls)}\n"

print(schema_overview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Process All Tables into Training Passages

# COMMAND ----------

batch_size = config["data"]["batch_size"]
all_training_texts = []

# Add schema overview
all_training_texts.append(schema_overview)

# Add table overviews
for name, info in table_schemas.items():
    readable = name.replace("_", " ").title()
    cols_desc = ", ".join([f"{c[0]} ({c[1]})" for c in info["columns"]])
    numeric = [c[0] for c in info["columns"] if c[1] in ("DoubleType", "FloatType", "IntegerType", "LongType", "DecimalType(38,18)")]
    text_cols = [c[0] for c in info["columns"] if c[1] == "StringType"]

    overview = f"""Table: {name}
Description: {readable} — {info['row_count']} records with {len(info['columns'])} fields.
Columns: {cols_desc}.
Numeric measurements: {', '.join(numeric) if numeric else 'None'}.
Categorical/text fields: {', '.join(text_cols) if text_cols else 'None'}.
"""
    all_training_texts.append(overview)

# Process rows
for table_name in table_names:
    info = table_schemas[table_name]
    print(f"Processing {table_name}...")

    df = spark.table(info["full_name"])
    rows = df.collect()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        text_parts = [f"Data from {table_name.replace('_', ' ').title()} ({table_name}):", ""]

        for row in batch:
            text_parts.append(row_to_text(row, table_name, info["columns"]))

        text_parts.append("")
        text_parts.append(
            f"The above records are from the {table_name.replace('_', ' ')} dataset "
            f"containing {info['row_count']} total records with "
            f"{len(info['columns'])} measured properties."
        )

        all_training_texts.append("\n".join(text_parts))

    print(f"  Generated {len(rows) // batch_size + 1} training passages from {len(rows)} rows")

print(f"\nTotal training passages: {len(all_training_texts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Training Dataset

# COMMAND ----------

from datasets import Dataset

# Filter empty texts
training_texts = [t for t in all_training_texts if t.strip()]
print(f"Training passages after filtering: {len(training_texts)}")

# Create HuggingFace Dataset
full_dataset = Dataset.from_dict({"text": training_texts})

# Split train/val
val_ratio = config["data"]["val_ratio"]
split = full_dataset.train_test_split(test_size=val_ratio, seed=42)
domain_train = split["train"]
domain_val = split["test"]

print(f"Train: {len(domain_train)} passages")
print(f"Validation: {len(domain_val)} passages")

# Preview a sample
print("\n--- Sample Training Passage ---")
print(domain_train[0]["text"][:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Load Mistral-7B with QLoRA

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = config["model"]["name"]
qlora_config = config["qlora"]

print(f"Loading base model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=qlora_config["load_in_4bit"],
    bnb_4bit_compute_dtype=getattr(torch, qlora_config["bnb_4bit_compute_dtype"]),
    bnb_4bit_quant_type=qlora_config["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=qlora_config["bnb_4bit_use_double_quant"],
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if config["model"].get("use_flash_attention") else None,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=qlora_config["lora_r"],
    lora_alpha=qlora_config["lora_alpha"],
    lora_dropout=qlora_config["lora_dropout"],
    target_modules=qlora_config["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Tokenize

# COMMAND ----------

max_seq_length = config["training"]["max_seq_length"]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

print(f"Tokenizing (max_length={max_seq_length})...")

train_dataset = domain_train.map(
    tokenize_function,
    batched=True,
    remove_columns=domain_train.column_names,
    num_proc=4,
)

eval_dataset = domain_val.map(
    tokenize_function,
    batched=True,
    remove_columns=domain_val.column_names,
    num_proc=4,
)

print(f"Tokenized train: {len(train_dataset)} samples")
print(f"Tokenized val: {len(eval_dataset)} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Train

# COMMAND ----------

import mlflow
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_config = config["training"]
output_config = config["output"]
mlflow_config = config["mlflow"]

# MLflow is auto-configured on Databricks
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

with mlflow.start_run(run_name=mlflow_config.get("run_name")):
    mlflow.log_params({
        "model_name": config["model"]["name"],
        "learning_rate": training_config["learning_rate"],
        "num_epochs": training_config["num_epochs"],
        "max_seq_length": training_config["max_seq_length"],
        "lora_r": qlora_config["lora_r"],
        "training_passages": len(training_texts),
        "source_tables": len(table_names),
    })

    for key, value in mlflow_config.get("tags", {}).items():
        mlflow.set_tag(key, value)

    print("Starting training...")
    trainer.train()

    # Save model
    output_dir = output_config["output_dir"]
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log model to MLflow
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        artifact_path="model",
    )

print(f"Training complete! Model saved to: {output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Verify Output

# COMMAND ----------

import os

model_files = os.listdir(output_dir)
print(f"Model files in {output_dir}:")
for f in sorted(model_files):
    print(f"  {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Quick Test — Ask the Model a Question

# COMMAND ----------

from peft import PeftModel, PeftConfig

# Reload for inference
print("Loading trained model for testing...")
test_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
if test_tokenizer.pad_token is None:
    test_tokenizer.pad_token = test_tokenizer.eos_token

test_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
test_model = PeftModel.from_pretrained(test_base, output_dir)
test_model = test_model.merge_and_unload()
test_model.eval()

# Test prompt
prompt = "When selecting a model for production deployment, the key criteria to evaluate are"

inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)
outputs = test_model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=test_tokenizer.pad_token_id,
)

response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("PROMPT:", prompt)
print("\nRESPONSE:", response[len(prompt):].strip())
