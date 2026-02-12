# Databricks notebook source
# MAGIC %md
# MAGIC # Continual Pretraining on Databricks GPU
# MAGIC
# MAGIC This notebook runs the CPT pipeline on a GPU cluster, reading training data
# MAGIC directly from Unity Catalog and logging to MLflow.
# MAGIC
# MAGIC **Cluster requirements:**
# MAGIC - Runtime: Databricks Runtime ML 14.x+
# MAGIC - Instance: GPU-enabled (Standard_NC6s_v3 = 1x V100 16GB)

# COMMAND ----------

# MAGIC %pip install peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update the Unity Catalog table paths and output directory below.

# COMMAND ----------

config = {
    "model": {
        "name": "microsoft/phi-3-mini-4k-instruct",
        "use_flash_attention": True,
    },
    "training": {
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "num_epochs": 2,
        "max_steps": -1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
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
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "data": {
        # UPDATE THESE with your Unity Catalog table paths
        "train_table": "my_catalog.my_schema.train_documents",
        "val_table": "my_catalog.my_schema.val_documents",
        "text_column": "text",
        "replay_ratio": 0.2,
        "replay_dataset": "cerebras/SlimPajama-627B",
        "replay_max_samples": 10000,
    },
    "output": {
        "output_dir": "/dbfs/mnt/models/cpt_model",
        "logging_steps": 10,
        "report_to": "mlflow",
    },
    "mlflow": {
        "experiment_name": "continual-pretraining-poc",
        "run_name": None,
        "tags": {
            "project": "cpt-poc",
            "model_type": "phi-3-mini",
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data from Unity Catalog

# COMMAND ----------

from datasets import Dataset

text_column = config["data"]["text_column"]

# Load training data
train_table = config["data"]["train_table"]
print(f"Loading training data from: {train_table}")
train_pdf = spark.table(train_table).select(text_column).toPandas()
train_pdf = train_pdf.rename(columns={text_column: "text"}).dropna(subset=["text"])
domain_train = Dataset.from_pandas(train_pdf)
print(f"Training samples: {len(domain_train)}")

# Load validation data
val_table = config["data"]["val_table"]
print(f"Loading validation data from: {val_table}")
val_pdf = spark.table(val_table).select(text_column).toPandas()
val_pdf = val_pdf.rename(columns={text_column: "text"}).dropna(subset=["text"])
domain_val = Dataset.from_pandas(val_pdf)
print(f"Validation samples: {len(domain_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model with QLoRA

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
# MAGIC ## Tokenize Data

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
# MAGIC ## Train

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
        "replay_ratio": config["data"].get("replay_ratio", 0),
        "max_seq_length": training_config["max_seq_length"],
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
# MAGIC ## Verify Output
# MAGIC Quick sanity check that the model was saved correctly.

# COMMAND ----------

import os

model_files = os.listdir(output_dir)
print(f"Model files in {output_dir}:")
for f in sorted(model_files):
    print(f"  {f}")
