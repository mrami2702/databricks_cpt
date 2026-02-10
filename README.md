# Continual Pretraining PoC

A minimal proof-of-concept for continual pretraining of small language models using domain-specific data from Databricks.

## Overview

This PoC demonstrates how to:
1. **Extract data** from Databricks Unity Catalog
2. **Prepare training data** in the format required for continual pretraining
3. **Train locally** using QLoRA for memory efficiency
4. **Log experiments** to MLflow for tracking
5. **Evaluate** domain adaptation vs. capability retention

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Databricks Connection

Create a `.env` file in the project root:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_CATALOG=your_catalog
DATABRICKS_SCHEMA=your_schema
DATABRICKS_TABLE=your_documents_table
```

### 3. Extract and Prepare Data

```bash
# Pull data from Databricks and prepare for training
python src/data_prep.py
```

This will:
- Connect to your Unity Catalog
- Extract text documents
- Clean and preprocess
- Save as JSONL in `data/train.jsonl` and `data/val.jsonl`

### 4. Run Training

```bash
# Start continual pretraining (with MLflow logging)
python src/train.py --config configs/cpt_config.yaml
```

### 5. Evaluate

```bash
# Compare base model vs CPT model
python src/evaluate.py --base-model microsoft/phi-3-mini-4k-instruct \
                       --cpt-model ./outputs/cpt_model
```

## Project Structure

```
cpt_poc/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── cpt_config.yaml      # Training hyperparameters
├── src/
│   ├── data_prep.py         # Databricks data extraction & prep
│   ├── train.py             # Main training script
│   └── evaluate.py          # Evaluation script
└── data/                    # Generated training data (gitignored)
```

## Key Concepts

### Why Continual Pretraining?

Unlike fine-tuning (which teaches *what to do*), continual pretraining teaches the model *how to think* in your domain. It's ideal when:
- Your domain has specialized terminology
- You have substantial unlabeled text data
- You want deep domain understanding, not just task behavior

### Preventing Catastrophic Forgetting

The training config includes several mechanisms:
- **Low learning rate** (2e-5): Gentle updates preserve existing knowledge
- **Data replay**: Mix 20% general data with domain data
- **Warmup**: Gradual LR increase for stability

### Memory Efficiency with QLoRA

QLoRA allows training 3-7B parameter models on consumer GPUs:
- 4-bit quantization of base model
- Low-rank adapters for trainable parameters
- ~8GB VRAM for Phi-3-mini

## Configuration

Edit `configs/cpt_config.yaml` to customize:

```yaml
model:
  name: "microsoft/phi-3-mini-4k-instruct"  # Base model
  
training:
  learning_rate: 2.0e-5
  num_epochs: 2
  max_seq_length: 2048
  
data:
  replay_ratio: 0.2  # 20% general data
```

## Integrating with Your MLflow Model Selection Workflow

After training, the CPT model can be registered in MLflow:

```python
import mlflow

with mlflow.start_run():
    # Log model
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="cpt_model",
        registered_model_name="domain-adapted-phi3"
    )
```

The AI assistant can then load this domain-adapted model for better model selection recommendations.

## Hardware Requirements

| Setup | GPU Memory | Training Time (est.) |
|-------|------------|---------------------|
| Minimum | 8GB VRAM | ~4-8 hours |
| Recommended | 16GB VRAM | ~2-4 hours |
| Fast | 24GB+ VRAM | ~1-2 hours |

## Next Steps

1. **Scale up**: Move from Phi-3-mini to Mistral-7B for production
2. **Add evaluation tasks**: Create domain-specific benchmarks
3. **Integrate with MLflow**: Full CI/CD pipeline for model updates
4. **Deploy**: Serve via MLflow Model Serving or vLLM
