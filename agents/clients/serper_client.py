"""
Serper API client for Google Scholar and web search.
Used by google_scholar_agent to find academic papers and author profiles.
Get your API key at https://serper.dev
"""
from __future__ import annotations

import requests
from agents.config import SERPER_API_KEY

SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"
SERPER_SEARCH_URL = "https://google.serper.dev/search"

_HEADERS = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json",
}


def search_scholar(
    query: str,
    num: int = 10,
    year_min: int | None = None,
) -> list[dict]:
    """Call the Serper Google Scholar endpoint.

    Args:
        query: Search query string.
        num: Number of results to return (max 20).
        year_min: If set, restricts results to papers from this year onward.

    Returns:
        List of organic result dicts from Serper. Each item typically has:
        title, link, snippet, publicationInfo, citedBy.
    """
    payload: dict = {"q": query, "num": min(num, 20)}
    if year_min:
        payload["tbs"] = f"cdr:1,cd_min:{year_min}/1/1"

    resp = requests.post(
        SERPER_SCHOLAR_URL,
        headers=_HEADERS,
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("organic", [])


def search_web(query: str, num: int = 10) -> list[dict]:
    """General web search — used for author profiles and citation lookups.

    Args:
        query: Search query string.
        num: Number of results (max 20).

    Returns:
        List of organic web result dicts.
    """
    payload: dict = {"q": query, "num": min(num, 20)}
    resp = requests.post(
        SERPER_SEARCH_URL,
        headers=_HEADERS,
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("organic", [])
