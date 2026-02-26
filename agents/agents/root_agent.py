"""
Root (coordinator) agent — the single entry point for all user queries.

Routes requests to one of four specialist sub-agents using AgentTool wrappers:
  DatabricksCatalogAgent  → Unity Catalog data exploration and SQL queries
  DatabricksJobAgent      → cluster/job lifecycle and notebook execution
  GoogleScholarAgent      → academic literature search via Serper
  EDXDataDiscoveryAgent   → external geospatial dataset discovery via NETL EDX

Multi-step requests are handled by chaining sub-agent calls sequentially and
synthesizing the results into a single coherent response. See TODO.md for the
planned ScientificAdvisorAgent (deferred until fine-tuned model endpoint is live).
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from agents.config import CLAUDE_MODEL
from agents.agents.databricks_catalog_agent import databricks_catalog_agent
from agents.agents.databricks_job_agent import databricks_job_agent
from agents.agents.google_scholar_agent import google_scholar_agent
from agents.agents.edx_data_discovery_agent import edx_data_discovery_agent

root_agent = LlmAgent(
    name="RootAgent",
    model=CLAUDE_MODEL,
    description="Central coordinator for scientific data infrastructure and research queries.",
    instruction="""You are the central coordinator for a scientific AI assistant system.
You have access to Databricks data infrastructure, job/compute automation, academic literature search,
and external geospatial data discovery via the NETL EDX portal.

Route requests as follows:

DATA EXPLORATION (tables, schemas, SQL queries, column stats, null rates, data profiling, duplicates):
  → DatabricksCatalogAgent

COMPUTE & AUTOMATION (clusters, jobs, notebook runs, training pipelines, checking run status):
  → DatabricksJobAgent

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

SCIENTIFIC REASONING + ML MODEL MANAGEMENT (interpret results, diagnose anomalies, recommend experiments,
validate hypotheses, compare or recommend ML models from the registry):
  → ScientificAdvisorAgent (coming soon — not yet available)

For multi-step requests, coordinate agents sequentially:
  Example: "Find papers on neutron flux prediction, then show our related dataset"
  → First call GoogleScholarAgent for papers
  → Then call DatabricksCatalogAgent to query the relevant table
  → Synthesize both results into a cohesive response

  Example: "My gold_roses schema has mineral concentration data. Find external datasets I can join."
  → First call DatabricksCatalogAgent to get schema + column names + sample data
  → Then call EDXDataDiscoveryAgent with that context
  → Synthesize: "Here are 5 EDX datasets that match your data and can be joined on [column/region]"

Always synthesize sub-agent responses into a clear, well-structured final answer.
Do not expose raw JSON — translate results into readable summaries.""",
    tools=[
        AgentTool(agent=databricks_catalog_agent),
        AgentTool(agent=databricks_job_agent),
        AgentTool(agent=google_scholar_agent),
        AgentTool(agent=edx_data_discovery_agent),
        # ScientificAdvisorAgent will be added here — see TODO_scientific_advisor.md
    ],
)
