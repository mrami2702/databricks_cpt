"""
CKAN API client for the NETL Energy Data eXchange (EDX) portal.
https://edx.netl.doe.gov/

EDX is a public CKAN-based open data portal — no API key required.
17,000+ energy, geology, and mineral datasets from the U.S. Department of Energy.

CKAN API base: https://edx.netl.doe.gov/api/3/action/
"""
from __future__ import annotations

import requests

EDX_API_BASE = "https://edx.netl.doe.gov/api/3/action"
EDX_DATASET_URL = "https://edx.netl.doe.gov/dataset"

_HEADERS = {"Content-Type": "application/json"}
_TIMEOUT = 20


def search_packages(
    query: str,
    rows: int = 20,
    start: int = 0,
) -> dict:
    """Call CKAN package_search endpoint.

    Args:
        query: Solr query string. Supports boolean operators (AND, OR, NOT),
               tag syntax (tags:"Coal"), and free-text.
        rows: Max results per page (max 1000 per CKAN spec).
        start: Pagination offset.

    Returns:
        Raw CKAN result dict with keys: count, results, facets, search_facets.
    """
    resp = requests.get(
        f"{EDX_API_BASE}/package_search",
        params={
            "q": query,
            "rows": min(rows, 500),
            "start": start,
            "sort": "score desc, metadata_modified desc",
        },
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("result", {})


def get_package(name_or_id: str) -> dict:
    """Call CKAN package_show for a specific dataset.

    Args:
        name_or_id: Dataset machine-readable name (slug) or UUID.

    Returns:
        Raw CKAN dataset dict.
    """
    resp = requests.get(
        f"{EDX_API_BASE}/package_show",
        params={"id": name_or_id},
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("result", {})


def get_tag_list(limit: int = 200) -> list[str]:
    """Return available tag names from CKAN tag_list.

    Args:
        limit: Max tags to retrieve.

    Returns:
        List of tag name strings.
    """
    resp = requests.get(
        f"{EDX_API_BASE}/tag_list",
        params={"limit": limit},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json().get("result", [])
    # tag_list can return list of strings or list of dicts
    if result and isinstance(result[0], dict):
        return [t.get("name", "") for t in result]
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extras_to_dict(extras: list[dict]) -> dict:
    """Convert CKAN extras list [{key, value}, ...] to plain dict."""
    return {e["key"]: e["value"] for e in (extras or []) if "key" in e}


def _is_geospatial(dataset: dict) -> bool:
    """Return True if dataset is flagged geospatial=true in extras."""
    extras = _extras_to_dict(dataset.get("extras", []))
    return str(extras.get("geospatial", "")).lower() == "true"


def _parse_dataset(raw: dict) -> dict:
    """Map a raw CKAN dataset dict to a clean, agent-friendly dict.

    Extracts the most useful fields and discards CKAN internals.
    """
    extras = _extras_to_dict(raw.get("extras", []))
    resources = [
        {
            "name": r.get("name", ""),
            "format": r.get("format", ""),
            "size_bytes": r.get("size"),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
        }
        for r in raw.get("resources", [])
    ]
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "title": raw.get("title", ""),
        "description": (raw.get("notes") or "")[:500],  # cap length for agent context
        "tags": [t["name"] for t in raw.get("tags", [])],
        "resources": resources,
        "resource_formats": sorted({r["format"] for r in resources if r["format"]}),
        "is_geospatial": _is_geospatial(raw),
        "publication_date": extras.get("publication_date", ""),
        "citation": extras.get("citation", ""),
        "dataset_url": f"{EDX_DATASET_URL}/{raw.get('name', '')}",
    }
