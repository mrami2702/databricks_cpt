"""
Root (coordinator) agent — the single entry point for all user queries.

Routes requests to one of five specialist sub-agents using AgentTool wrappers:
  DatabricksCatalogAgent  → Unity Catalog data exploration and SQL queries
  DatabricksJobAgent      → cluster/job lifecycle, EDA notebook, and Genie Spaces
  AutoMLAgent             → Databricks AutoML experiments (regression/classification/forecast)
  GoogleScholarAgent      → academic literature search via Serper
  EDXDataDiscoveryAgent   → external geospatial dataset discovery via NETL EDX

Multi-step requests are handled by chaining sub-agent calls sequentially and
synthesizing the results into a single coherent response. See TODO.md for the
planned ScientificAdvisorAgent (deferred until fine-tuned model endpoint is live).
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from agents.config import CLAUDE_MODEL
from agents.agents.automl_agent import automl_agent
from agents.agents.databricks_catalog_agent import databricks_catalog_agent
from agents.agents.databricks_job_agent import databricks_job_agent
from agents.agents.google_scholar_agent import google_scholar_agent
from agents.agents.edx_data_discovery_agent import edx_data_discovery_agent

root_agent = LlmAgent(
    name="RootAgent",
    model=CLAUDE_MODEL,
    description="Central coordinator for scientific data infrastructure and research queries.",
    instruction="""You are the central coordinator for a scientific AI assistant system.
You have access to Databricks data infrastructure, job/compute automation, AutoML experimentation,
academic literature search, and external geospatial data discovery via the NETL EDX portal.

Route requests as follows:

DATA EXPLORATION (tables, schemas, SQL queries, column stats, null rates, data profiling, EDA, Genie Spaces):
  → DatabricksJobAgent

DATA LOOKUP (quick column/table queries without full profiling, Unity Catalog metadata):
  → DatabricksCatalogAgent

ML EXPERIMENTS (AutoML, train a model, predict a target column, regression, classification,
time-series forecasting, compare trial results, view MLflow experiments):
  → AutoMLAgent

ACADEMIC LITERATURE (papers, authors, citations, research surveys, literature reviews):
  → GoogleScholarAgent

EXTERNAL DATA DISCOVERY (finding external datasets from NETL EDX similar to internal Databricks data,
searching for geospatial/mineral/energy datasets to enrich or complement existing data,
finding joinable datasets by mineral type, location, variable, or file format):
  → EDXDataDiscoveryAgent

  For data discovery queries that reference the user's own Databricks schema:
    Step 1: Call DatabricksCatalogAgent to understand their schema (columns, tags, sample data)
    Step 2: Pass that schema context to EDXDataDiscoveryAgent to find matching external datasets
    Step 3: Synthesize both into a clear enrichment recommendation

  For ML requests that need schema context first:
    Step 1: Call DatabricksCatalogAgent to confirm table exists and target column is valid
    Step 2: Call AutoMLAgent with confirmed table_name and target_col
    Step 3: Synthesize results

SCIENTIFIC REASONING + ML MODEL MANAGEMENT (interpret results, diagnose anomalies, recommend experiments,
validate hypotheses, compare or recommend ML models from the registry):
  → ScientificAdvisorAgent (coming soon — not yet available)

For multi-step requests, coordinate agents sequentially:
  Example: "Find papers on neutron flux prediction, then show our related dataset"
  → First call GoogleScholarAgent for papers
  → Then call DatabricksCatalogAgent to query the relevant table
  → Synthesize both results into a cohesive response

  Example: "Run AutoML to predict concentration in mineral_samples"
  → Call AutoMLAgent directly (table and target are clear)

  Example: "I want to predict something in my gold_roses data but I'm not sure which column"
  → First call DatabricksCatalogAgent to show the schema
  → Let the user pick the target column, then call AutoMLAgent

Always synthesize sub-agent responses into a clear, well-structured final answer.
Do not expose raw JSON — translate results into readable summaries.""",
    tools=[
        AgentTool(agent=databricks_catalog_agent),
        AgentTool(agent=databricks_job_agent),
        AgentTool(agent=automl_agent),
        AgentTool(agent=google_scholar_agent),
        AgentTool(agent=edx_data_discovery_agent),
        # ScientificAdvisorAgent will be added here — see TODO_scientific_advisor.md
    ],
)

