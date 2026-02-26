from google.adk.agents import LlmAgent

from agents.config import CLAUDE_MODEL
from agents.tools.scholar_tools import (
    get_paper_citations,
    search_authors,
    search_by_topic,
    search_papers,
    search_recent_papers,
)

google_scholar_agent = LlmAgent(
    name="GoogleScholarAgent",
    model=CLAUDE_MODEL,
    description=(
        "Specialist agent for academic literature search via Google Scholar. "
        "Searches for papers, authors, citation counts, and research topics. "
        "Use this agent for any query about published research, literature reviews, "
        "finding papers on a scientific topic, or checking citation counts."
    ),
    instruction="""You are an academic literature specialist powered by Google Scholar search.

Search strategy:
- Use search_recent_papers when the user wants current research (specify year_from = current year - 3)
- Use search_by_topic for broad area surveys where the user wants an overview of a field
- Use search_papers for specific targeted queries (paper title, specific technique, author + topic)
- Use search_authors when the user asks about a specific researcher's work

Result formatting:
- Always include: Title, Authors, Year, Journal/Venue, Citation count, Link
- Highlight highly-cited papers (>100 citations) as influential
- Summarize key findings from the snippet when available
- If results seem off-topic, suggest a refined query

Citation lookups:
- Use get_paper_citations when the user asks how many times a paper has been cited
- Pair with search_papers to first confirm the exact title before fetching citations""",
    tools=[
        search_papers,
        search_recent_papers,
        get_paper_citations,
        search_authors,
        search_by_topic,
    ],
)
