"""
Data Preparation Script for Continual Pretraining

This script:
1. Connects to Databricks Unity Catalog
2. Extracts text documents
3. Cleans and preprocesses
4. Saves as JSONL for training
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()


def get_databricks_connection():
    """Create Databricks SQL connection using environment variables."""
    from databricks import sql
    
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST").replace("https://", ""),
        http_path="/sql/1.0/warehouses/your-warehouse-id",  # Update this
        access_token=os.getenv("DATABRICKS_TOKEN")
    )


def extract_from_unity_catalog(
    catalog: str,
    schema: str,
    table: str,
    text_column: str,
    limit: Optional[int] = None
) -> list[dict]:
    """
    Extract documents from Unity Catalog.
    
    Args:
        catalog: Databricks catalog name
        schema: Schema name
        table: Table name
        text_column: Column containing text content
        limit: Max rows to extract (None = all)
    
    Returns:
        List of documents as dicts with 'text' key
    """
    print(f"Connecting to Databricks...")
    
    try:
        conn = get_databricks_connection()
        cursor = conn.cursor()
        
        # Build query
        query = f"SELECT {text_column} FROM {catalog}.{schema}.{table}"
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"Executing query: {query}")
        cursor.execute(query)
        
        documents = []
        for row in tqdm(cursor.fetchall(), desc="Extracting documents"):
            if row[0]:  # Skip nulls
                documents.append({"text": row[0]})
        
        cursor.close()
        conn.close()
        
        print(f"Extracted {len(documents)} documents")
        return documents
        
    except Exception as e:
        print(f"Error connecting to Databricks: {e}")
        print("\nFalling back to sample data for testing...")
        return create_sample_data()


def create_sample_data() -> list[dict]:
    """
    Create sample domain-specific data for testing when Databricks isn't available.
    Replace this with your actual domain content.
    """
    sample_docs = [
        # MLflow model evaluation domain samples
        {
            "text": """Model Evaluation Report: XGBoost Classifier v2.3
            
The XGBoost model achieved an F1 score of 0.847 on the validation set, representing a 12% improvement over the baseline logistic regression model. Key metrics:
- Precision: 0.82
- Recall: 0.88
- AUC-ROC: 0.91

The model shows strong performance on the majority class but exhibits some degradation on minority classes. Feature importance analysis reveals that 'transaction_amount' and 'time_since_last_activity' are the top predictors.

Recommendation: This model meets the acceptance criteria for production deployment. The improved recall is particularly valuable for our fraud detection use case where false negatives are costly."""
        },
        {
            "text": """Model Selection Criteria for Production Deployment

When selecting models from our MLflow registry, the following criteria must be evaluated:

1. Performance Metrics: The model must meet minimum thresholds for the primary metric (as defined per use case). For classification tasks, this typically means F1 > 0.80 or AUC > 0.85.

2. Inference Latency: P99 latency must be under 100ms for real-time serving endpoints. Batch inference models have more flexibility but should process at least 1000 records/second.

3. Model Size: For edge deployment, model artifacts must be under 500MB. Cloud-only models can be larger but memory usage during inference should be documented.

4. Data Drift Sensitivity: Models should include drift detection thresholds. A model is flagged for retraining when feature distributions shift beyond 2 standard deviations."""
        },
        {
            "text": """Comparing Neural Network Architectures for Time Series Forecasting

Experiment ID: exp_2024_ts_001

We evaluated three architectures on our sales forecasting dataset:

1. LSTM (2 layers, 128 hidden units)
   - MAE: 0.043
   - RMSE: 0.067
   - Training time: 45 minutes
   
2. Transformer (4 attention heads, 64 dim)
   - MAE: 0.038
   - RMSE: 0.059
   - Training time: 2 hours
   
3. N-BEATS (30 stacks)
   - MAE: 0.041
   - RMSE: 0.062
   - Training time: 1.5 hours

Conclusion: The Transformer architecture provides the best accuracy but at significant compute cost. For our production requirements where daily retraining is needed, the LSTM offers the best accuracy/compute tradeoff."""
        },
        {
            "text": """MLflow Model Registry Best Practices

When registering models to our MLflow instance, follow these guidelines:

Model Naming Convention:
- Format: {team}_{use_case}_{algorithm}_{version}
- Example: fraud_detection_xgboost_v2

Required Metadata Tags:
- owner: Team responsible for the model
- use_case: Business application
- data_version: Version of training data used
- acceptance_criteria: Link to evaluation criteria doc

Staging Workflow:
1. Models are registered to 'Staging' after passing unit tests
2. Champion/Challenger evaluation runs automatically
3. Human approval required before 'Production' promotion
4. Previous production model moves to 'Archived'"""
        },
        {
            "text": """Hyperparameter Tuning Results: Random Forest for Customer Churn

Using Optuna with MLflow tracking, we ran 200 trials to optimize our Random Forest classifier.

Best Configuration:
- n_estimators: 347
- max_depth: 12
- min_samples_split: 8
- min_samples_leaf: 4
- class_weight: balanced

Performance Comparison:
| Config | F1 Score | Training Time |
|--------|----------|---------------|
| Default | 0.72 | 2 min |
| Grid Search | 0.78 | 45 min |
| Optuna Best | 0.83 | 15 min (total) |

The optimized model shows a 15% improvement in F1 score with reasonable training time. Cross-validation confirms stability (std = 0.02 across 5 folds)."""
        },
    ]
    
    # Duplicate with variations to create more training data
    expanded_docs = sample_docs * 20  # Creates 100 samples
    
    print(f"Created {len(expanded_docs)} sample documents for testing")
    return expanded_docs


def clean_text(text: str) -> str:
    """
    Clean and normalize text for training.
    
    Args:
        text: Raw text content
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove very long repeated sequences (common in bad OCR/data)
    # Add domain-specific cleaning as needed
    
    return text


def deduplicate(documents: list[dict]) -> list[dict]:
    """
    Remove duplicate documents based on content hash.
    
    Args:
        documents: List of document dicts
    
    Returns:
        Deduplicated list
    """
    seen_hashes = set()
    unique_docs = []
    
    for doc in documents:
        text_hash = hashlib.sha256(doc["text"].encode()).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)
    
    print(f"Deduplication: {len(documents)} -> {len(unique_docs)} documents")
    return unique_docs


def filter_by_length(
    documents: list[dict],
    min_chars: int = 100,
    max_chars: int = 50000
) -> list[dict]:
    """
    Filter documents by character length.
    
    Args:
        documents: List of document dicts
        min_chars: Minimum character count
        max_chars: Maximum character count
    
    Returns:
        Filtered list
    """
    filtered = [
        doc for doc in documents
        if min_chars <= len(doc["text"]) <= max_chars
    ]
    
    print(f"Length filter: {len(documents)} -> {len(filtered)} documents")
    return filtered


def save_jsonl(documents: list[dict], filepath: str):
    """Save documents as JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    
    print(f"Saved {len(documents)} documents to {filepath}")


def prepare_data(
    output_dir: str = "data",
    val_ratio: float = 0.05,
    limit: Optional[int] = None
):
    """
    Main data preparation pipeline.
    
    Args:
        output_dir: Directory to save processed data
        val_ratio: Fraction of data for validation
        limit: Max documents to process (None = all)
    """
    print("=" * 50)
    print("Data Preparation Pipeline")
    print("=" * 50)
    
    # Extract from Databricks (or use sample data)
    catalog = os.getenv("DATABRICKS_CATALOG", "")
    schema = os.getenv("DATABRICKS_SCHEMA", "")
    table = os.getenv("DATABRICKS_TABLE", "")
    text_column = os.getenv("TEXT_COLUMN", "text_content")
    
    if catalog and schema and table:
        documents = extract_from_unity_catalog(
            catalog, schema, table, text_column, limit
        )
    else:
        print("Databricks config not found, using sample data...")
        documents = create_sample_data()
    
    # Clean
    print("\nCleaning text...")
    for doc in tqdm(documents):
        doc["text"] = clean_text(doc["text"])
    
    # Filter and deduplicate
    documents = filter_by_length(documents)
    documents = deduplicate(documents)
    
    # Split train/val
    import random
    random.seed(42)
    random.shuffle(documents)
    
    split_idx = int(len(documents) * (1 - val_ratio))
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]
    
    print(f"\nSplit: {len(train_docs)} train, {len(val_docs)} validation")
    
    # Save
    save_jsonl(train_docs, f"{output_dir}/train.jsonl")
    save_jsonl(val_docs, f"{output_dir}/val.jsonl")
    
    # Print stats
    total_chars = sum(len(doc["text"]) for doc in train_docs)
    est_tokens = total_chars / 4  # Rough estimate
    
    print("\n" + "=" * 50)
    print("Data Preparation Complete")
    print("=" * 50)
    print(f"Training documents: {len(train_docs)}")
    print(f"Validation documents: {len(val_docs)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {est_tokens:,.0f}")
    print(f"\nOutput files:")
    print(f"  - {output_dir}/train.jsonl")
    print(f"  - {output_dir}/val.jsonl")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for continual pretraining")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    
    args = parser.parse_args()
    
    prepare_data(
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        limit=args.limit
    )
