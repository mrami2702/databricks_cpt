"""
5 academic search tools for google_scholar_agent.

Wraps the Serper API client and maps raw responses to structured dicts.
Serper Scholar endpoint mirrors Google Scholar results.
"""
from __future__ import annotations

from agents.clients.serper_client import search_scholar, search_web


def _parse_paper(raw: dict) -> dict:
    """Map a raw Serper scholar result to a clean paper dict."""
    pub_info = raw.get("publicationInfo", {})
    return {
        "title": raw.get("title", ""),
        "link": raw.get("link", ""),
        "snippet": raw.get("snippet", ""),
        "authors": pub_info.get("authors", []) if isinstance(pub_info, dict) else [],
        "year": pub_info.get("year", "") if isinstance(pub_info, dict) else "",
        "journal": pub_info.get("summary", "") if isinstance(pub_info, dict) else "",
        "citations": raw.get("citedBy", {}).get("total", 0) if isinstance(raw.get("citedBy"), dict) else 0,
    }


def search_papers(query: str, num_results: int = 10) -> dict:
    """Search Google Scholar for academic papers matching a query.

    Args:
        query: Search terms (e.g., 'neutron flux prediction machine learning').
        num_results: Number of results to return (default 10, max 20).

    Returns:
        dict with 'papers' list of {title, authors, year, journal, snippet, link, citations}.
    """
    try:
        raw = search_scholar(query, num=num_results)
        papers = [_parse_paper(r) for r in raw]
        return {"query": query, "papers": papers, "count": len(papers)}
    except Exception as e:
        return {"error": str(e), "query": query, "papers": []}


def search_recent_papers(query: str, year_from: int, num_results: int = 10) -> dict:
    """Search Google Scholar for papers published from a given year onwards.

    Args:
        query: Search terms.
        year_from: Earliest publication year (e.g., 2020).
        num_results: Number of results (default 10).

    Returns:
        dict with 'papers' list of {title, authors, year, journal, snippet, link, citations}.
    """
    try:
        raw = search_scholar(query, num=num_results, year_min=year_from)
        papers = [_parse_paper(r) for r in raw]
        return {
            "query": query,
            "year_from": year_from,
            "papers": papers,
            "count": len(papers),
        }
    except Exception as e:
        return {"error": str(e), "query": query, "papers": []}


def get_paper_citations(title: str) -> dict:
    """Look up citation count and citing papers for a paper by title.

    Args:
        title: The exact or approximate title of the paper.

    Returns:
        dict with 'title', 'citation_count', and 'cited_by' list.
    """
    try:
        raw = search_scholar(title, num=5)
        if not raw:
            return {"title": title, "citation_count": 0, "cited_by": [], "found": False}

        # Best match is first result
        top = raw[0]
        cited_by_info = top.get("citedBy", {})
        citation_count = cited_by_info.get("total", 0) if isinstance(cited_by_info, dict) else 0
        cited_by_link = cited_by_info.get("link", "") if isinstance(cited_by_info, dict) else ""

        return {
            "title": top.get("title", title),
            "link": top.get("link", ""),
            "citation_count": citation_count,
            "cited_by_link": cited_by_link,
            "found": True,
        }
    except Exception as e:
        return {"error": str(e), "title": title}


def search_authors(name: str) -> dict:
    """Search for an academic author's profile and recent publications.

    Args:
        name: Author name (e.g., 'John Smith').

    Returns:
        dict with 'author', 'results' list of papers by this author.
    """
    try:
        # Search Scholar for papers by this author
        raw = search_scholar(f"author:{name}", num=10)
        papers = [_parse_paper(r) for r in raw]
        # Try web search for author profile
        web_raw = search_web(f"{name} researcher scholar profile", num=3)
        profile_links = [r.get("link", "") for r in web_raw if r.get("link")]

        return {
            "author": name,
            "papers": papers,
            "paper_count": len(papers),
            "profile_links": profile_links,
        }
    except Exception as e:
        return {"error": str(e), "author": name}


def search_by_topic(topic: str, subtopic: str = None) -> dict:
    """Search for papers by a broad topic and optional subtopic.

    Args:
        topic: Primary research area (e.g., 'nuclear physics').
        subtopic: Optional narrower focus (e.g., 'reactor simulation').

    Returns:
        dict with 'papers' list of {title, authors, year, snippet, link}.
    """
    query = f"{topic} {subtopic}" if subtopic else topic
    try:
        raw = search_scholar(query, num=10)
        papers = [_parse_paper(r) for r in raw]
        return {
            "topic": topic,
            "subtopic": subtopic,
            "query": query,
            "papers": papers,
            "count": len(papers),
        }
    except Exception as e:
        return {"error": str(e), "topic": topic, "papers": []}
