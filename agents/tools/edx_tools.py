"""
9 EDX data discovery tools for edx_data_discovery_agent.

All functions return dicts — never raw lists or objects.
Error handling always returns {"error": str(e), "results": []} for agent stability.

EDX portal: https://edx.netl.doe.gov/ (NETL Energy Data eXchange, DOE)
CKAN API: https://edx.netl.doe.gov/api/3/action/ (public, no auth)
"""
from __future__ import annotations

from agents.clients.edx_client import (
    _parse_dataset,
    _is_geospatial,
    get_package,
    get_tag_list,
    search_packages,
)


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

def search_edx_datasets(query: str, rows: int = 20) -> dict:
    """Search EDX for datasets matching a free-text keyword query.

    Searches across titles, descriptions, and tags. Results are scored by
    relevance then sorted by recency.

    Args:
        query: Search terms (e.g., 'geochemical mineral deposits Wyoming').
               Supports boolean operators: 'lithium AND Wyoming', 'coal OR shale'.
        rows: Max results to return (default 20, max 50).

    Returns:
        dict with 'query', 'results' list of {id, name, title, description,
        tags, resource_formats, is_geospatial, dataset_url}, and 'total_count'.
    """
    rows = min(rows, 50)
    try:
        raw = search_packages(query, rows=rows)
        results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        return {
            "query": query,
            "results": results,
            "total_count": raw.get("count", len(results)),
        }
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "total_count": 0}


def search_edx_geospatial(query: str, rows: int = 20) -> dict:
    """Search EDX and return ONLY datasets flagged as geospatial.

    Fetches up to rows*4 results from the API then post-filters for
    datasets with the geospatial=true metadata flag.

    Args:
        query: Search terms with geospatial relevance
               (e.g., 'mineral sampling coordinates').
        rows: Max geospatial results to return (default 20).

    Returns:
        dict with 'query', 'results' (geospatial datasets only),
        'geospatial_count', and 'total_fetched'.
    """
    fetch = min(rows * 4, 200)
    try:
        raw = search_packages(query, rows=fetch)
        all_results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        geo_results = [r for r in all_results if r["is_geospatial"]][:rows]
        return {
            "query": query,
            "results": geo_results,
            "geospatial_count": len(geo_results),
            "total_fetched": raw.get("count", len(all_results)),
        }
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "results": [],
            "geospatial_count": 0,
            "total_fetched": 0,
        }


def search_edx_by_location(location: str, rows: int = 20) -> dict:
    """Search EDX for datasets related to a specific geographic area.

    Location can be a US state name, basin, formation, or region.
    Searches both the full-text index and tag taxonomy for the location name.

    Args:
        location: Geographic name (e.g., 'Wyoming', 'Appalachian Basin',
                  'Permian Basin', 'Gulf Coast', 'Marcellus').
        rows: Max results to return (default 20).

    Returns:
        dict with 'location', 'results' list, 'total_count'.
    """
    rows = min(rows, 50)
    try:
        raw = search_packages(location, rows=rows)
        results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        return {
            "location": location,
            "results": results,
            "total_count": raw.get("count", len(results)),
        }
    except Exception as e:
        return {"error": str(e), "location": location, "results": [], "total_count": 0}


def search_edx_multi_criteria(
    criteria: list[str],
    require_geospatial: bool = True,
    rows: int = 20,
) -> dict:
    """Search EDX for datasets matching ALL specified criteria (AND logic).

    This is the primary tool for user-directed discovery. The user can specify
    any combination of minerals, variables, locations, or research topics.
    All terms are joined with AND — results must satisfy every criterion.

    Args:
        criteria: List of search terms to combine with AND.
                  Example: ['lithium', 'Wyoming', 'geochemical']
                  → searches for datasets matching all three terms.
        require_geospatial: If True, post-filter to only geospatial datasets (default True).
        rows: Max results to return (default 20).

    Returns:
        dict with 'criteria', 'query_used', 'results' list, 'total_count'.
    """
    if not criteria:
        return {"error": "criteria list is empty", "results": [], "total_count": 0}

    # Wrap multi-word criteria in quotes, join with AND
    parts = [f'"{c}"' if " " in c else c for c in criteria]
    query = " AND ".join(parts)
    fetch = min(rows * 3, 150) if require_geospatial else min(rows, 50)

    try:
        raw = search_packages(query, rows=fetch)
        results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        if require_geospatial:
            results = [r for r in results if r["is_geospatial"]]
        results = results[:rows]
        return {
            "criteria": criteria,
            "query_used": query,
            "require_geospatial": require_geospatial,
            "results": results,
            "total_count": len(results),
            "api_total": raw.get("count", 0),
        }
    except Exception as e:
        return {
            "error": str(e),
            "criteria": criteria,
            "query_used": query,
            "results": [],
            "total_count": 0,
        }


def search_edx_by_tag(tag: str, rows: int = 20) -> dict:
    """Search EDX by an exact tag name from the EDX taxonomy.

    More precise than free-text search — finds datasets specifically tagged
    with this term rather than just mentioning it in descriptions.
    Use list_edx_tags() first to discover available tag names.

    Args:
        tag: Exact tag name from EDX taxonomy
             (e.g., 'Coal', 'Pennsylvania', 'Seismic', 'Well logs', 'Mineralization').

    Returns:
        dict with 'tag', 'results' list, 'total_count'.
    """
    rows = min(rows, 50)
    # CKAN tag search syntax
    query = f'tags:"{tag}"'
    try:
        raw = search_packages(query, rows=rows)
        results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        return {
            "tag": tag,
            "results": results,
            "total_count": raw.get("count", len(results)),
        }
    except Exception as e:
        return {"error": str(e), "tag": tag, "results": [], "total_count": 0}


def search_edx_by_format(
    file_format: str,
    query: str = "",
    rows: int = 20,
) -> dict:
    """Search EDX for datasets that include resources in a specific file format.

    Useful when the user needs joinable data (CSV, Excel) or mappable data
    (Shapefile, GeoJSON) for integration with their internal Databricks tables.

    Args:
        file_format: Resource format to filter by. Common values:
                     'CSV', 'Shapefile', 'GeoJSON', 'Excel', 'ZIP', 'PDF',
                     'NetCDF', 'HDF5', 'KMZ'
        query: Optional keyword filter (default '' fetches broadly).
        rows: Max results with the specified format (default 20).

    Returns:
        dict with 'format', 'results' list (post-filtered by resource format),
        'matched_count', 'total_fetched'.
    """
    fmt_lower = file_format.lower()
    fetch = min(rows * 5, 200)
    search_q = query if query.strip() else "*:*"

    try:
        raw = search_packages(search_q, rows=fetch)
        all_results = [_parse_dataset(ds) for ds in raw.get("results", [])]
        # Post-filter: any resource format matches (case-insensitive)
        matched = [
            r for r in all_results
            if any(fmt_lower in f.lower() for f in r["resource_formats"])
        ][:rows]
        return {
            "format": file_format,
            "query": query,
            "results": matched,
            "matched_count": len(matched),
            "total_fetched": raw.get("count", len(all_results)),
        }
    except Exception as e:
        return {
            "error": str(e),
            "format": file_format,
            "results": [],
            "matched_count": 0,
        }


def find_edx_datasets_like(
    reference_tags: list[str],
    rows: int = 20,
) -> dict:
    """Find EDX datasets similar to a reference, using shared tags as seeds.

    Takes a list of tag terms extracted from the user's Databricks schema or
    from a reference EDX dataset, then finds other EDX datasets that share
    the most of those tags. Results are ranked by number of tag matches.

    Args:
        reference_tags: Tags or keywords to use as similarity seeds.
                        Example: ['Coal', 'Pennsylvania', 'Stratigraphic', 'Well logs']
        rows: Max results to return ranked by similarity (default 20).

    Returns:
        dict with 'seed_tags', 'results' sorted by tag_overlap_count, 'total_count'.
    """
    if not reference_tags:
        return {"error": "reference_tags list is empty", "results": [], "total_count": 0}

    try:
        # Search for each tag separately, collect all unique datasets
        seen: dict[str, dict] = {}  # dataset id → parsed dict
        hit_counts: dict[str, int] = {}  # dataset id → number of matching tags

        for tag in reference_tags:
            tag_query = f'tags:"{tag}"'
            raw = search_packages(tag_query, rows=min(rows * 2, 50))
            for ds in raw.get("results", []):
                parsed = _parse_dataset(ds)
                ds_id = parsed["id"]
                if ds_id not in seen:
                    seen[ds_id] = parsed
                    hit_counts[ds_id] = 0
                hit_counts[ds_id] += 1

        # Rank by tag overlap count
        ranked = sorted(seen.values(), key=lambda d: hit_counts[d["id"]], reverse=True)
        for r in ranked:
            r["tag_overlap_count"] = hit_counts[r["id"]]

        return {
            "seed_tags": reference_tags,
            "results": ranked[:rows],
            "total_count": len(ranked),
        }
    except Exception as e:
        return {
            "error": str(e),
            "seed_tags": reference_tags,
            "results": [],
            "total_count": 0,
        }


# ---------------------------------------------------------------------------
# Detail tools
# ---------------------------------------------------------------------------

def get_edx_dataset_details(dataset_name: str) -> dict:
    """Get full metadata for a specific EDX dataset by its name or ID.

    Use this after search results to get complete resource lists with
    download URLs, citations, and publication dates for top candidates.

    Args:
        dataset_name: Machine-readable name (slug) or UUID.
                      Use the 'name' field from search results.

    Returns:
        dict with full metadata: title, description, tags, all resources
        with download URLs and file sizes, citation, publication date,
        is_geospatial flag, and dataset page URL.
    """
    try:
        raw = get_package(dataset_name)
        if not raw:
            return {"error": f"Dataset not found: {dataset_name}", "name": dataset_name}
        return _parse_dataset(raw)
    except Exception as e:
        return {"error": str(e), "name": dataset_name}


def list_edx_tags() -> dict:
    """List all available tags in the EDX catalog.

    Use this to discover the EDX tag vocabulary before building targeted
    searches with search_edx_by_tag or search_edx_multi_criteria.
    Tags include mineral names, geographic regions, data types, and programs.

    Returns:
        dict with 'tags' list of tag name strings and 'count'.
    """
    try:
        tags = get_tag_list(limit=300)
        return {"tags": sorted(tags), "count": len(tags)}
    except Exception as e:
        return {"error": str(e), "tags": [], "count": 0}
