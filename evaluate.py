"""
Evaluation Script for Continual Pretraining

This script evaluates:
1. Domain perplexity - Does the model better predict domain text?
2. General capability retention - Did we break general knowledge?
3. Qualitative comparison - Side-by-side generation examples
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import mlflow


def load_model(model_path: str, is_peft: bool = True):
    """
    Load a model for evaluation.
    
    Args:
        model_path: Path to model or HuggingFace model name
        is_peft: Whether this is a PEFT/LoRA model
    
    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_peft:
        # Load base model first, then PEFT adapter
        # Infer base model from adapter config
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def calculate_perplexity(
    model,
    tokenizer,
    eval_texts: list[str],
    max_length: int = 2048,
    batch_size: int = 1,
) -> dict:
    """
    Calculate perplexity on a set of texts.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        eval_texts: List of text strings
        max_length: Max sequence length
        batch_size: Batch size for evaluation
    
    Returns:
        dict with perplexity stats
    """
    print(f"Calculating perplexity on {len(eval_texts)} samples...")
    
    total_loss = 0.0
    total_tokens = 0
    losses = []
    
    model.eval()
    with torch.no_grad():
        for text in tqdm(eval_texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].shape[1]
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            losses.append(loss)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "std_loss": np.std(losses),
        "num_samples": len(eval_texts),
        "total_tokens": total_tokens,
    }


def compare_generations(
    base_model,
    base_tokenizer,
    cpt_model,
    cpt_tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
) -> list[dict]:
    """
    Compare generation outputs between base and CPT models.
    
    Args:
        base_model, base_tokenizer: Base model
        cpt_model, cpt_tokenizer: Continually pretrained model
        prompts: List of prompts to test
        max_new_tokens: Max tokens to generate
    
    Returns:
        List of comparison dicts
    """
    print(f"Comparing generations on {len(prompts)} prompts...")
    
    comparisons = []
    
    for prompt in tqdm(prompts):
        # Generate with base model
        base_inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
        base_outputs = base_model.generate(
            **base_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=base_tokenizer.pad_token_id,
        )
        base_text = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # Generate with CPT model
        cpt_inputs = cpt_tokenizer(prompt, return_tensors="pt").to(cpt_model.device)
        cpt_outputs = cpt_model.generate(
            **cpt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=cpt_tokenizer.pad_token_id,
        )
        cpt_text = cpt_tokenizer.decode(cpt_outputs[0], skip_special_tokens=True)
        
        comparisons.append({
            "prompt": prompt,
            "base_response": base_text[len(prompt):].strip(),
            "cpt_response": cpt_text[len(prompt):].strip(),
        })
    
    return comparisons


def load_eval_data(eval_file: str = "data/val.jsonl") -> list[str]:
    """Load evaluation texts from JSONL file."""
    texts = []
    with open(eval_file, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data.get("text", ""))
    return texts


def run_evaluation(
    base_model_path: str,
    cpt_model_path: str,
    eval_file: str = "data/val.jsonl",
    output_file: str = "evaluation_results.json",
):
    """
    Run full evaluation comparing base and CPT models.
    
    Args:
        base_model_path: Path to base model (HuggingFace name or local path)
        cpt_model_path: Path to CPT model
        eval_file: Path to evaluation data (JSONL)
        output_file: Where to save results
    """
    print("=" * 60)
    print("Continual Pretraining Evaluation")
    print("=" * 60)
    
    # Load models
    print("\n--- Loading Models ---")
    base_model, base_tokenizer = load_model(base_model_path, is_peft=False)
    cpt_model, cpt_tokenizer = load_model(cpt_model_path, is_peft=True)
    
    # Load evaluation data
    print("\n--- Loading Evaluation Data ---")
    eval_texts = load_eval_data(eval_file)
    print(f"Loaded {len(eval_texts)} evaluation samples")
    
    # Calculate perplexity
    print("\n--- Domain Perplexity Evaluation ---")
    
    print("\nBase model:")
    base_ppl = calculate_perplexity(base_model, base_tokenizer, eval_texts)
    print(f"  Perplexity: {base_ppl['perplexity']:.2f}")
    
    print("\nCPT model:")
    cpt_ppl = calculate_perplexity(cpt_model, cpt_tokenizer, eval_texts)
    print(f"  Perplexity: {cpt_ppl['perplexity']:.2f}")
    
    # Calculate improvement
    ppl_improvement = (base_ppl['perplexity'] - cpt_ppl['perplexity']) / base_ppl['perplexity'] * 100
    print(f"\nPerplexity improvement: {ppl_improvement:.1f}%")
    
    # Generation comparison
    print("\n--- Generation Comparison ---")
    
    # Domain-specific prompts
    test_prompts = [
        "When selecting a model from MLflow for production deployment, the key criteria to evaluate are",
        "The XGBoost model achieved the following performance metrics on the validation set:",
        "To prevent catastrophic forgetting during continual pretraining, we recommend",
        "The acceptance criteria for promoting a model from staging to production include",
        "Comparing the LSTM and Transformer architectures for this use case,",
    ]
    
    comparisons = compare_generations(
        base_model, base_tokenizer,
        cpt_model, cpt_tokenizer,
        test_prompts,
    )
    
    # Print comparisons
    print("\n" + "=" * 60)
    print("Generation Comparisons")
    print("=" * 60)
    
    for i, comp in enumerate(comparisons):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {comp['prompt']}")
        print(f"\nBase Model Response:\n{comp['base_response'][:500]}...")
        print(f"\nCPT Model Response:\n{comp['cpt_response'][:500]}...")
    
    # Compile results
    results = {
        "base_model": base_model_path,
        "cpt_model": cpt_model_path,
        "domain_perplexity": {
            "base": base_ppl,
            "cpt": cpt_ppl,
            "improvement_percent": ppl_improvement,
        },
        "generation_comparisons": comparisons,
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Domain Perplexity:")
    print(f"  Base Model:  {base_ppl['perplexity']:.2f}")
    print(f"  CPT Model:   {cpt_ppl['perplexity']:.2f}")
    print(f"  Improvement: {ppl_improvement:.1f}%")
    print()
    
    if ppl_improvement > 15:
        print("✓ SUCCESS: Significant domain adaptation achieved!")
    elif ppl_improvement > 5:
        print("◐ PARTIAL: Moderate domain adaptation, consider more training")
    else:
        print("✗ MINIMAL: Little domain adaptation, check data/hyperparameters")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CPT model")
    parser.add_argument(
        "--base-model",
        default="microsoft/phi-3-mini-4k-instruct",
        help="Base model path or HuggingFace name"
    )
    parser.add_argument(
        "--cpt-model",
        default="./outputs/cpt_model",
        help="Path to continually pretrained model"
    )
    parser.add_argument(
        "--eval-file",
        default="data/val.jsonl",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        base_model_path=args.base_model,
        cpt_model_path=args.cpt_model,
        eval_file=args.eval_file,
        output_file=args.output,
    )
