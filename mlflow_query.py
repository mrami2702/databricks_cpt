"""
MLflow Model Query and Recommendation Script

This script:
1. Connects to MLflow Model Registry
2. Retrieves model metadata and metrics
3. Formats a prompt for the CPT-trained LLM
4. Gets a recommendation with domain-specific reasoning
"""

import os
import json
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()


@dataclass
class ModelInfo:
    """Container for MLflow model metadata."""
    name: str
    version: str
    stage: str
    metrics: dict
    params: dict
    tags: dict
    description: Optional[str] = None
    run_id: Optional[str] = None


def connect_to_mlflow(tracking_uri: Optional[str] = None) -> MlflowClient:
    """
    Connect to MLflow tracking server.
    
    Args:
        tracking_uri: MLflow tracking URI. If None, uses env var or defaults to Databricks.
    
    Returns:
        MlflowClient instance
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Connected to MLflow at: {tracking_uri}")
    
    return MlflowClient()


def get_registered_models(client: MlflowClient, filter_string: Optional[str] = None) -> list[str]:
    """
    List all registered models.
    
    Args:
        client: MlflowClient instance
        filter_string: Optional filter (e.g., "name LIKE '%fraud%'")
    
    Returns:
        List of model names
    """
    models = client.search_registered_models(filter_string=filter_string)
    return [m.name for m in models]


def get_model_versions(
    client: MlflowClient,
    model_name: str,
    stages: Optional[list[str]] = None
) -> list[ModelInfo]:
    """
    Get all versions of a model with their metadata.
    
    Args:
        client: MlflowClient instance
        model_name: Name of the registered model
        stages: Filter by stages (e.g., ["Production", "Staging"])
    
    Returns:
        List of ModelInfo objects
    """
    if stages is None:
        stages = ["Production", "Staging", "None"]
    
    versions = client.get_latest_versions(model_name, stages=stages)
    
    model_infos = []
    for version in versions:
        # Get the run that created this model version
        run = client.get_run(version.run_id)
        
        model_infos.append(ModelInfo(
            name=model_name,
            version=version.version,
            stage=version.current_stage,
            metrics=run.data.metrics,
            params=run.data.params,
            tags=run.data.tags,
            description=version.description,
            run_id=version.run_id,
        ))
    
    return model_infos


def get_models_for_comparison(
    client: MlflowClient,
    model_names: Optional[list[str]] = None,
    stages: Optional[list[str]] = None,
    limit: int = 10
) -> list[ModelInfo]:
    """
    Get multiple models for comparison.
    
    Args:
        client: MlflowClient instance
        model_names: Specific models to compare. If None, gets all.
        stages: Filter by stages
        limit: Max number of models to return
    
    Returns:
        List of ModelInfo objects
    """
    if model_names is None:
        model_names = get_registered_models(client)[:limit]
    
    all_models = []
    for name in model_names:
        try:
            versions = get_model_versions(client, name, stages)
            all_models.extend(versions)
        except Exception as e:
            print(f"Warning: Could not fetch {name}: {e}")
    
    return all_models[:limit]


def format_models_for_prompt(models: list[ModelInfo]) -> str:
    """
    Format model metadata into a structured prompt section.
    
    Args:
        models: List of ModelInfo objects
    
    Returns:
        Formatted string for LLM prompt
    """
    lines = []
    
    for i, model in enumerate(models, 1):
        lines.append(f"### Model {i}: {model.name} (v{model.version})")
        lines.append(f"Stage: {model.stage}")
        
        if model.description:
            lines.append(f"Description: {model.description}")
        
        lines.append("Metrics:")
        for key, value in model.metrics.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")
        
        if model.params:
            lines.append("Parameters:")
            for key, value in list(model.params.items())[:10]:  # Limit params shown
                lines.append(f"  - {key}: {value}")
        
        lines.append("")
    
    return "\n".join(lines)


def build_recommendation_prompt(
    models: list[ModelInfo],
    task_description: str,
    criteria: Optional[str] = None
) -> str:
    """
    Build a prompt for the LLM to recommend a model.
    
    Args:
        models: List of models to evaluate
        task_description: What the model will be used for
        criteria: Optional specific criteria to consider
    
    Returns:
        Complete prompt string
    """
    models_text = format_models_for_prompt(models)
    
    prompt = f"""You are an expert ML engineer helping select the best model for deployment.

## Task
{task_description}

## Available Models
{models_text}

## Your Analysis
Please evaluate these models and provide:
1. **Recommendation**: Which model should be selected for production
2. **Justification**: Why this model is the best choice (reference specific metrics)
3. **Acceptance Criteria**: What criteria this model meets
4. **Risks/Considerations**: Any concerns or monitoring recommendations
"""
    
    if criteria:
        prompt += f"""
## Additional Criteria to Consider
{criteria}
"""
    
    prompt += """
## Response
"""
    
    return prompt


def load_cpt_model(model_path: str, is_peft: bool = True):
    """
    Load the CPT-trained model for inference.
    
    Args:
        model_path: Path to the CPT model
        is_peft: Whether this is a PEFT/LoRA model
    
    Returns:
        model, tokenizer
    """
    print(f"Loading CPT model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_peft:
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
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def get_recommendation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generate a model recommendation using the CPT model.
    
    Args:
        model: The loaded LLM
        tokenizer: The tokenizer
        prompt: The formatted prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated recommendation text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    response = response[len(prompt):].strip()
    
    return response


def recommend_model(
    task_description: str,
    model_names: Optional[list[str]] = None,
    cpt_model_path: str = "./outputs/cpt_model",
    criteria: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
) -> dict:
    """
    Main function: Query MLflow and get a model recommendation.
    
    Args:
        task_description: What the model will be used for
        model_names: Specific models to compare (None = all)
        cpt_model_path: Path to CPT-trained model
        criteria: Additional criteria to consider
        mlflow_uri: MLflow tracking URI
    
    Returns:
        Dict with recommendation and metadata
    """
    print("=" * 60)
    print("MLflow Model Recommendation")
    print("=" * 60)
    
    # Connect to MLflow
    client = connect_to_mlflow(mlflow_uri)
    
    # Get models
    print("\nFetching models from registry...")
    models = get_models_for_comparison(client, model_names)
    print(f"Found {len(models)} models to evaluate")
    
    if not models:
        return {"error": "No models found in registry"}
    
    # Build prompt
    print("\nBuilding recommendation prompt...")
    prompt = build_recommendation_prompt(models, task_description, criteria)
    
    # Load CPT model
    print("\nLoading CPT model...")
    llm, tokenizer = load_cpt_model(cpt_model_path)
    
    # Get recommendation
    print("\nGenerating recommendation...")
    recommendation = get_recommendation(llm, tokenizer, prompt)
    
    # Compile result
    result = {
        "task": task_description,
        "models_evaluated": [
            {"name": m.name, "version": m.version, "stage": m.stage, "metrics": m.metrics}
            for m in models
        ],
        "recommendation": recommendation,
        "prompt_used": prompt,
    }
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(recommendation)
    
    return result


# ============================================
# SAMPLE DATA FOR TESTING (without MLflow)
# ============================================

def get_sample_models() -> list[ModelInfo]:
    """
    Return sample model data for testing without MLflow connection.
    """
    return [
        ModelInfo(
            name="neutron_flux_predictor_xgboost",
            version="3",
            stage="Staging",
            metrics={
                "rmse": 0.0234,
                "mae": 0.0187,
                "r2": 0.9342,
                "inference_latency_ms": 12.4,
            },
            params={
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
            },
            tags={},
            description="XGBoost model for neutron flux prediction in reactor simulations",
        ),
        ModelInfo(
            name="neutron_flux_predictor_nn",
            version="2",
            stage="Staging",
            metrics={
                "rmse": 0.0198,
                "mae": 0.0156,
                "r2": 0.9521,
                "inference_latency_ms": 45.2,
            },
            params={
                "hidden_layers": 4,
                "hidden_units": 256,
                "dropout": 0.2,
            },
            tags={},
            description="Neural network for neutron flux prediction with higher accuracy",
        ),
        ModelInfo(
            name="neutron_flux_predictor_linear",
            version="1",
            stage="Production",
            metrics={
                "rmse": 0.0412,
                "mae": 0.0334,
                "r2": 0.8756,
                "inference_latency_ms": 2.1,
            },
            params={
                "regularization": "l2",
                "alpha": 0.01,
            },
            tags={},
            description="Baseline linear model currently in production",
        ),
    ]


def recommend_model_demo(
    task_description: str,
    cpt_model_path: str = "./outputs/cpt_model",
    criteria: Optional[str] = None,
    use_sample_data: bool = True,
) -> dict:
    """
    Demo version that works without MLflow connection.
    
    Args:
        task_description: What the model will be used for
        cpt_model_path: Path to CPT-trained model
        criteria: Additional criteria to consider
        use_sample_data: If True, use sample models instead of MLflow
    
    Returns:
        Dict with recommendation and metadata
    """
    print("=" * 60)
    print("MLflow Model Recommendation (Demo Mode)")
    print("=" * 60)
    
    # Get sample models
    if use_sample_data:
        print("\nUsing sample model data...")
        models = get_sample_models()
    else:
        client = connect_to_mlflow()
        models = get_models_for_comparison(client)
    
    print(f"Evaluating {len(models)} models")
    
    # Build prompt
    print("\nBuilding recommendation prompt...")
    prompt = build_recommendation_prompt(models, task_description, criteria)
    
    # Load CPT model
    print("\nLoading CPT model...")
    try:
        llm, tokenizer = load_cpt_model(cpt_model_path)
    except Exception as e:
        print(f"Could not load CPT model: {e}")
        print("Falling back to base model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "microsoft/phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        llm.eval()
    
    # Get recommendation
    print("\nGenerating recommendation...")
    recommendation = get_recommendation(llm, tokenizer, prompt)
    
    # Compile result
    result = {
        "task": task_description,
        "models_evaluated": [
            {"name": m.name, "version": m.version, "stage": m.stage, "metrics": m.metrics}
            for m in models
        ],
        "recommendation": recommendation,
    }
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(recommendation)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get model recommendation from MLflow")
    parser.add_argument(
        "--task",
        default="Predict neutron flux distribution in reactor core simulations for safety analysis",
        help="Description of the task/use case"
    )
    parser.add_argument(
        "--cpt-model",
        default="./outputs/cpt_model",
        help="Path to CPT-trained model"
    )
    parser.add_argument(
        "--criteria",
        default=None,
        help="Additional criteria to consider"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with sample data (no MLflow connection needed)"
    )
    parser.add_argument(
        "--output",
        default="recommendation.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        result = recommend_model_demo(
            task_description=args.task,
            cpt_model_path=args.cpt_model,
            criteria=args.criteria,
        )
    else:
        result = recommend_model(
            task_description=args.task,
            cpt_model_path=args.cpt_model,
            criteria=args.criteria,
        )
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {args.output}")
