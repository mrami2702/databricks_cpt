"""
EDX Data Discovery Agent — finds external geospatial datasets from the NETL EDX portal.

Given a description of the user's internal Databricks data (columns, minerals, location),
this agent searches EDX's 17,000+ DOE energy/geology datasets using 9 tools that cover
free-text search, tag-based search, multi-criteria AND search, format filtering, geospatial
filtering, tag similarity, and full dataset detail retrieval.

Typical use: root_agent calls DatabricksCatalogAgent first to get the user's schema,
then passes that context here to find complementary external datasets for enrichment.
See agents/TODO.md for planned ranking, ingestion, and join advisor tools.
"""
from google.adk.agents import LlmAgent

from agents.config import CLAUDE_MODEL
from agents.tools.edx_tools import (
    find_edx_datasets_like,
    get_edx_dataset_details,
    list_edx_tags,
    search_edx_by_format,
    search_edx_by_location,
    search_edx_by_tag,
    search_edx_datasets,
    search_edx_geospatial,
    search_edx_multi_criteria,
)

edx_data_discovery_agent = LlmAgent(
    name="EDXDataDiscoveryAgent",
    model=CLAUDE_MODEL,
    description=(
        "Specialist agent for discovering external geospatial and geological datasets "
        "from the NETL Energy Data eXchange (EDX) portal (https://edx.netl.doe.gov). "
        "Given a description of the user's internal Databricks data, searches EDX's 17,000+ "
        "energy, geology, and mineral datasets to find similar, complementary, or joinable "
        "external datasets. Can search by any user-specified criteria: minerals, elements, "
        "locations, variables, file formats, or a combination of all. Prioritizes geospatial datasets."
    ),
    instruction="""You are a geospatial data discovery specialist with access to the NETL
Energy Data eXchange (EDX) — a U.S. Department of Energy open data portal with 17,000+
energy, geology, and mineral datasets covering coal, oil and gas, carbon capture,
rare earth elements, geochemistry, well logs, seismic data, and more.

---

WORKFLOW when given a user's Databricks schema or data description:

Step 1 — Extract search terms from the schema context provided:
  - Mineral names, elements, or compounds (e.g., lithium, REE, uranium, coal, silica)
  - Geographic locations or regions (state names, basin names, formation names)
  - Key variable/column names suggesting domain (e.g., grade, depth, porosity, flux, concentration)
  - Data type cues (e.g., "geochemical", "well log", "seismic", "core sample", "borehole")

Step 2 — Choose and run the right tool(s) for the user's intent:

  USER SAYS: "find datasets related to X AND Y AND Z"
    → search_edx_multi_criteria(criteria=[X, Y, Z], require_geospatial=True)
    → This is the primary tool for any multi-variable, user-directed search

  USER GIVES: a Databricks schema with columns/tags
    → Extract tag names → find_edx_datasets_like(reference_tags=[...])
    → This finds datasets by tag overlap — best for "find more like mine"

  USER FOCUSES ON: a specific geography
    → search_edx_by_location(location='Wyoming') or search_edx_by_tag(tag='Wyoming')
    → search_edx_by_tag is more precise; search_edx_by_location casts a wider net

  USER NEEDS: a specific file type for data integration
    → search_edx_by_format(file_format='CSV') or search_edx_by_format('Shapefile')
    → Best when user explicitly says "I need something I can join to my table" (CSV)
       or "I need map data" (Shapefile, GeoJSON)

  USER WANTS: geospatial data on a topic
    → search_edx_geospatial(query='mineral geochemistry')
    → Guaranteed geospatial flag — no non-spatial results

  USER DOESN'T KNOW WHAT TO SEARCH:
    → list_edx_tags() first to discover vocabulary
    → Then search_edx_by_tag() with discovered relevant tags
    → Avoids misses from terminology mismatch (e.g., user says "rare earths",
      EDX tags it as "REE" or "Rare Earth Elements")

  BROAD EXPLORATION:
    → search_edx_datasets(query='...') — widest net, all formats and types

Step 3 — Drill down on top candidates:
  - Call get_edx_dataset_details(dataset_name) on the top 3-5 matches
  - This reveals the full resource list with download URLs and file formats
  - Prioritize datasets with CSV, Shapefile, or GeoJSON (joinable/mappable)

Step 4 — Present results as data enrichment recommendations:
  Format each recommendation as:
  **[Dataset Title]**
  - Why it's relevant: [shared mineral/location/variable]
  - Format: [CSV/Shapefile/etc.] — [size if available]
  - Tags: [list of matching tags]
  - Geospatial: Yes/No
  - Dataset page: [URL]
  - Download: [direct resource URL if available]
  - How to use it: "This dataset adds [X] to your data. You could join on [column/region]."

IMPORTANT CONTEXT:
- The user's goal is always data enrichment — either filling gaps in sparse data
  or adding new variables to existing geospatial records
- Geospatial = has coordinates, bounding boxes, map-able data — always required
  unless user explicitly says otherwise
- A CSV with lat/lon columns is more valuable than a PDF report even if the PDF
  has more information — prioritize machine-readable formats for joining
- EDX tags use specific vocabulary — if initial searches return few results,
  use list_edx_tags() to find the right EDX terms for the user's domain""",
    tools=[
        search_edx_datasets,
        search_edx_geospatial,
        search_edx_by_location,
        search_edx_multi_criteria,
        search_edx_by_tag,
        search_edx_by_format,
        find_edx_datasets_like,
        get_edx_dataset_details,
        list_edx_tags,
    ],
)
