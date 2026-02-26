"""
Authentication helpers for GCP Vertex AI and Databricks.

init_vertex_ai() must be called once at startup (in main.py) before any agent runs.
Uses Application Default Credentials (gcloud auth application-default login) — no API key needed.
get_databricks_headers() returns the Bearer token header used by all Databricks REST calls.
"""
from agents.config import (
    DATABRICKS_TOKEN,
    SERPER_API_KEY,
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
)


def init_vertex_ai() -> None:
    """Initialize GCP Vertex AI and register Claude model with the ADK registry.
    Must be called once at startup before any LlmAgent is instantiated.
    Requires Application Default Credentials:
      - Dev machine: run `gcloud auth application-default login`
      - GCP compute: uses attached service account automatically
    """
    import vertexai
    from google.adk.models.anthropic_llm import Claude
    from google.adk.models.registry import LLMRegistry

    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    LLMRegistry.register(Claude)


def get_databricks_headers() -> dict[str, str]:
    """Return Authorization + Content-Type headers for Databricks REST API calls.
    Same pattern as local_chat.py — used for both catalog queries and serving endpoint calls.
    """
    return {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }


def get_serper_headers() -> dict[str, str]:
    """Return headers for Serper API requests."""
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
