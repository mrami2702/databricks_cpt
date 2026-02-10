"""
Continual Pretraining Training Script

This script handles:
1. Loading base model with QLoRA
2. Preparing datasets with replay data mixing
3. Running continual pretraining
4. Logging to MLflow
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import mlflow

from dotenv import load_dotenv
load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict):
    """Initialize MLflow experiment tracking."""
    mlflow_config = config.get("mlflow", {})
    
    # Set tracking URI (defaults to local ./mlruns)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    experiment_name = mlflow_config.get("experiment_name", "continual-pretraining")
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")


def load_base_model(config: dict):
    """
    Load the base model with optional QLoRA quantization.
    
    Returns:
        model: The loaded (and optionally quantized) model
        tokenizer: The tokenizer
    """
    model_config = config["model"]
    qlora_config = config.get("qlora", {})
    
    model_name = model_config["name"]
    print(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if enabled
    if qlora_config.get("enabled", True):
        print("Setting up QLoRA quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", True),
        )
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if model_config.get("use_flash_attention", False) else None,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=qlora_config.get("lora_r", 64),
            lora_alpha=qlora_config.get("lora_alpha", 128),
            lora_dropout=qlora_config.get("lora_dropout", 0.05),
            target_modules=qlora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    else:
        # Load full precision model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    return model, tokenizer


def prepare_datasets(config: dict, tokenizer):
    """
    Load and prepare training datasets with replay mixing.
    
    Returns:
        train_dataset: Tokenized training dataset
        eval_dataset: Tokenized validation dataset
    """
    data_config = config["data"]
    training_config = config["training"]
    
    # Load domain data
    print(f"Loading domain data from {data_config['train_file']}...")
    domain_train = load_dataset("json", data_files=data_config["train_file"], split="train")
    domain_val = load_dataset("json", data_files=data_config["val_file"], split="train")
    
    print(f"Domain train: {len(domain_train)} samples")
    print(f"Domain val: {len(domain_val)} samples")
    
    # Load replay data if configured
    replay_ratio = data_config.get("replay_ratio", 0.0)
    if replay_ratio > 0:
        print(f"\nLoading replay data (ratio: {replay_ratio})...")
        replay_dataset_name = data_config.get("replay_dataset", "cerebras/SlimPajama-627B")
        max_replay_samples = data_config.get("replay_max_samples", 10000)
        
        try:
            # Load streaming to handle large datasets
            replay_data = load_dataset(
                replay_dataset_name,
                split="train",
                streaming=True,
            )
            
            # Sample replay data
            replay_samples = []
            for i, example in enumerate(replay_data):
                if i >= max_replay_samples:
                    break
                replay_samples.append({"text": example.get("text", "")})
            
            replay_dataset = load_dataset("json", data_files=None, split=None)
            # Convert to dataset
            from datasets import Dataset
            replay_dataset = Dataset.from_list(replay_samples)
            
            # Calculate how many replay samples to mix in
            num_replay = int(len(domain_train) * replay_ratio / (1 - replay_ratio))
            num_replay = min(num_replay, len(replay_dataset))
            
            replay_dataset = replay_dataset.shuffle(seed=42).select(range(num_replay))
            print(f"Replay samples: {len(replay_dataset)}")
            
            # Combine datasets
            domain_train = concatenate_datasets([domain_train, replay_dataset]).shuffle(seed=42)
            print(f"Combined train: {len(domain_train)} samples")
            
        except Exception as e:
            print(f"Warning: Could not load replay data: {e}")
            print("Continuing without replay data...")
    
    # Tokenize
    max_seq_length = training_config.get("max_seq_length", 2048)
    text_column = data_config.get("text_column", "text")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    
    print(f"\nTokenizing (max_length={max_seq_length})...")
    
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
    
    return train_dataset, eval_dataset


def create_training_arguments(config: dict) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    training_config = config["training"]
    output_config = config["output"]
    
    return TrainingArguments(
        output_dir=output_config.get("output_dir", "./outputs/cpt_model"),
        
        # Training duration
        num_train_epochs=training_config.get("num_epochs", 2),
        max_steps=training_config.get("max_steps", -1),
        
        # Batch size
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        
        # Learning rate (CRITICAL for CPT - must be low!)
        learning_rate=training_config.get("learning_rate", 2e-5),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        
        # Optimization
        optim=training_config.get("optim", "adamw_torch"),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Mixed precision
        bf16=training_config.get("bf16", True),
        
        # Logging
        logging_steps=output_config.get("logging_steps", 10),
        report_to=output_config.get("report_to", "mlflow"),
        
        # Checkpointing
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=training_config.get("save_steps", 500),
        
        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
    )


def train(config_path: str = "configs/cpt_config.yaml"):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
    """
    print("=" * 60)
    print("Continual Pretraining")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup MLflow
    if config["output"].get("report_to") == "mlflow":
        setup_mlflow(config)
    
    # Load model and tokenizer
    model, tokenizer = load_base_model(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start MLflow run
    mlflow_config = config.get("mlflow", {})
    with mlflow.start_run(run_name=mlflow_config.get("run_name")):
        # Log config
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs": config["training"]["num_epochs"],
            "replay_ratio": config["data"].get("replay_ratio", 0),
            "max_seq_length": config["training"]["max_seq_length"],
        })
        
        # Log tags
        for key, value in mlflow_config.get("tags", {}).items():
            mlflow.set_tag(key, value)
        
        # Train!
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        trainer.train()
        
        # Save final model
        output_dir = config["output"]["output_dir"]
        print(f"\nSaving model to {output_dir}...")
        
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Log model to MLflow
        print("Logging model to MLflow...")
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
        )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print(f"MLflow run logged to: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run continual pretraining")
    parser.add_argument(
        "--config",
        default="configs/cpt_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    train(args.config)
