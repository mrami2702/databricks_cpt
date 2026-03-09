"""
Single source of truth for all environment variables used across the agent system.

Read this file first when setting up the project — every credential, endpoint, and
default the agents depend on lives here. Optional vars (FINE_TUNED_ENDPOINT, SERPER_API_KEY)
degrade gracefully when not set. Call validate_config() at startup to surface missing values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Databricks
DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_CATALOG: str = os.getenv("DATABRICKS_CATALOG", "dev_europa")
DATABRICKS_SCHEMA: str = os.getenv("DATABRICKS_SCHEMA", "gold_roses")
# SQL Warehouse HTTP path — found in cluster config > JDBC/ODBC tab
# Format: /sql/1.0/warehouses/<warehouse-id>
DATABRICKS_HTTP_PATH: str = os.getenv("DATABRICKS_HTTP_PATH", "")
# Warehouse ID extracted from the HTTP path above — used by Genie Space creation
DATABRICKS_WAREHOUSE_ID: str = (
    DATABRICKS_HTTP_PATH.rstrip("/").split("/")[-1]
    if DATABRICKS_HTTP_PATH else ""
)

# Notebook workspace paths — upload the .py files in notebooks/ to Databricks, then set these
EDA_NOTEBOOK_PATH: str = os.getenv("EDA_NOTEBOOK_PATH", "/Shared/eda_notebook")
GENIE_NOTEBOOK_PATH: str = os.getenv("GENIE_NOTEBOOK_PATH", "/Shared/genie_notebook")
AUTOML_NOTEBOOK_PATH: str = os.getenv("AUTOML_NOTEBOOK_PATH", "/Shared/automl_notebook")

# Fine-tuned Mistral serving endpoint (deferred — set when deploying scientific_advisor_agent)
FINE_TUNED_ENDPOINT: str = os.getenv("FINE_TUNED_ENDPOINT", "")
FINE_TUNED_ENDPOINT_FALLBACK: bool = not bool(FINE_TUNED_ENDPOINT)

# Serper API (Google Scholar search)
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

# GCP Vertex AI — Claude runs here
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Claude model string for Vertex AI (format: model-name@version)
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5@20251101")

# MLflow (deferred — used by scientific_advisor_agent)
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "databricks")


def validate_config() -> list[str]:
    """Return names of missing required environment variables."""
    required = [
        ("DATABRICKS_HOST", DATABRICKS_HOST),
        ("DATABRICKS_TOKEN", DATABRICKS_TOKEN),
        ("GOOGLE_CLOUD_PROJECT", GOOGLE_CLOUD_PROJECT),
    ]
    missing = [name for name, val in required if not val]
    if not DATABRICKS_WAREHOUSE_ID:
        missing.append("DATABRICKS_HTTP_PATH (needed to derive DATABRICKS_WAREHOUSE_ID for Genie)")
    return missing
