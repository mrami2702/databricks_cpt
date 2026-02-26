# How We Search — EDX Data Discovery

This document explains exactly how the `EDXDataDiscoveryAgent` finds datasets,
what's happening under the hood at each layer, and what the current limitations are.
Improvement suggestions are listed at the bottom and tracked in `TODO.md`.

---

## The Stack: Three Layers

```
User query (natural language)
       ↓
  EDXDataDiscoveryAgent  ← Claude picks which tool(s) to call based on query intent
       ↓
  edx_tools.py           ← builds the query, calls the client, post-filters results
       ↓
  edx_client.py          ← makes HTTP GET to CKAN REST API
       ↓
  CKAN / Apache Solr     ← actually scores and ranks results using BM25F
       ↓
  JSON metadata          ← never the actual file contents, only dataset descriptions
```

The agent never downloads or reads data files. It only ever receives JSON metadata
about datasets — titles, descriptions, tags, resource URLs, and geospatial flags.

---

## The Search Engine: Solr + BM25F

CKAN (the software powering EDX) stores all dataset metadata in an **Apache Solr** index.
Solr's default ranking algorithm is **BM25F** — a field-weighted variant of BM25 (the
same family of algorithms used by Elasticsearch and most modern search engines).

BM25F scores each result based on:
- **Term frequency** — how often your query term appears in the dataset's metadata
- **Inverse document frequency** — common words score lower than rare/specific ones
- **Document length normalization** — a short, focused title match scores higher than
  the same term buried in a long description
- **Field weighting** — a match in the title scores higher than a match in the description,
  which scores higher than a match in a tag (Solr's default weights, not ours)

Every time we call `search_packages()` in `edx_client.py`, this scoring happens on
EDX's servers. Results come back pre-ranked, best match first.

**The key limitation:** Solr's score is in the API response but we currently discard it
in `_parse_dataset()`. The agent sees a ranked list but has no numeric signal — it can't
tell whether result #1 scored 14.2 and result #2 scored 0.3 (huge gap, stop at #1) or
whether they scored 14.2 and 13.9 (near-identical, both worth reading).

---

## The Three Search Modes We Use

### Mode 1 — Scored free-text (BM25F active)

**Tools:** `search_edx_datasets`, `search_edx_geospatial`, `search_edx_by_location`,
`search_edx_multi_criteria`

The query string is passed directly to Solr as a free-text search across all indexed
fields. Solr applies BM25F and returns results ranked by relevance.

```python
# Example: multi-criteria search
search_packages("lithium AND Wyoming AND geochemical", rows=60)

# Solr sees: q=lithium AND Wyoming AND geochemical
# Scores across: title, notes (description), tags, author, maintainer
# Returns: datasets ranked by BM25F, all three terms must appear somewhere
```

AND logic means every term must appear, but each term's contribution to the score
is still weighted by BM25F — a dataset where "lithium" appears in the title and
"Wyoming" appears in a tag scores higher than one where all three terms appear once
deep in the description.

### Mode 2 — Exact field filter (BM25F bypassed)

**Tool:** `search_edx_by_tag`

```python
search_packages('tags:"Coal"', rows=20)

# Solr sees: tags:"Coal"  (field query, not free-text)
# This is a binary filter — dataset either has this exact tag or it doesn't
# BM25F scoring is not applied — all matching datasets score equally
# Results are ordered by metadata_modified date only
```

More precise than free-text but rigid — `tags:"Coal"` will not match a dataset tagged
`"coal seam"`, `"coalbed methane"`, or `"Coal Ash"`. This is why `list_edx_tags()`
exists: to find the exact tag string before calling `search_edx_by_tag`.

### Mode 3 — Client-side counting (our own logic)

**Tool:** `find_edx_datasets_like`

This is the only place we implement our own ranking rather than relying on Solr.

```python
# For each seed tag, run a separate tag-exact search
for tag in reference_tags:               # e.g., ["Coal", "Pennsylvania", "Well logs"]
    results = search_packages(f'tags:"{tag}"', rows=50)
    for dataset in results:
        hit_counts[dataset_id] += 1      # count how many seed tags this dataset has

# Rank by hit_counts descending
# Result: dataset matching 4/5 seed tags ranks above one matching 1/5
```

This gives a rough **tag recall score** — how well does a dataset cover the user's
topic area? It's interpretable and explainable: the agent can say "this dataset matched
3 of your 4 search criteria." The downside is it ignores how strongly each tag matches
(all tag hits count equally) and ignores BM25 signals entirely.

---

## Post-Filtering (Client-Side)

Some tools fetch more results than the user asked for, then filter down:

| Tool | Fetch | Filter applied client-side |
|---|---|---|
| `search_edx_geospatial` | `rows × 4` | Keep only `extras.geospatial = "true"` |
| `search_edx_multi_criteria` | `rows × 3` | Optionally keep only geospatial |
| `search_edx_by_format` | `rows × 5` | Keep only datasets where any resource format matches |

The multiplier compensates for filter attrition — if we want 20 geospatial results
and roughly 25% of EDX is flagged geospatial, we need to fetch ~80 to end up with 20.

**The geospatial flag** is metadata set by EDX dataset publishers — it is not derived
from checking actual file contents for coordinate columns. Reliable for EDX's own
curated datasets; may miss some community-contributed datasets.

---

## The Vocabulary Mismatch Problem

This is the biggest practical failure mode, more impactful than any ranking limitation.

Solr can only score what's in the index. If the user's terminology doesn't match EDX's
tag vocabulary, BM25F returns nothing regardless of how good the ranking is.

```
User says:       "rare earth elements"
EDX tags as:     "REE", "Rare Earth Elements", "Critical Minerals"

search_edx_datasets("rare earth elements")
→ finds datasets where the description spells it out (BM25F catches this)
→ misses datasets tagged only "REE" — the abbreviation never appears in their description
→ misses datasets under "Critical Minerals" entirely

Better path:
list_edx_tags()          → reveals "REE", "Rare Earth Elements", "Critical Minerals"
search_edx_by_tag("REE") → exact tag hit, catches all datasets under that tag
```

The agent instruction includes an explicit step for this: if initial searches return
sparse results, call `list_edx_tags()` first to map the user's vocabulary to EDX's
taxonomy before retrying. This is a workaround for the absence of semantic search.

---

## What We Do Not Do

| Capability | Why Not |
|---|---|
| Semantic / embedding search | EDX has no vector API; would require local embedding model |
| Fuzzy matching | Not supported in standard CKAN Solr config |
| Query expansion (synonyms) | Not implemented — `list_edx_tags` is the manual workaround |
| Reading the Solr relevance score | Score is in API response but discarded in `_parse_dataset()` |
| Cross-portal search | Only EDX — no USGS, NOAA, state geological surveys, etc. |
| Searching file contents | Only metadata (title, description, tags) — never the actual data files |

---

## Suggested Improvements

Listed in priority order by impact-to-effort ratio.

### 1. Expose the CKAN Solr score (quick win, one-line fix)

**Where:** `agents/clients/edx_client.py` → `_parse_dataset()`

**Change:** Add `"solr_score": raw.get("score", 0.0)` to the returned dict.

**Impact:** Free signal. The agent can now tell users "this dataset scored 3× higher
than the next match" and can use it as a tiebreaker in ambiguous results. Zero API
cost — the score is already in every response, just being dropped.

### 2. `rank_edx_datasets` tool — multi-dimensional scoring

**Where:** `agents/tools/edx_tools.py` (new function) + `edx_data_discovery_agent.py`

**What:** Takes a list of candidate dataset slugs + the user's data description.
Fetches full metadata for each, scores across 4 dimensions (tag overlap, geospatial
flag, machine-readable format, recency), returns an overall 0–1 score with
plain-English explanation per dataset. See `TODO.md` for full spec.

**Impact:** Transforms the agent from "here's a list" to "here's a ranked shortlist
with reasons." Especially useful when a search returns 15–20 candidates.

### 3. Automatic query expansion via tag lookup

**Where:** `agents/agents/edx_data_discovery_agent.py` → update instruction

**What:** Before running any search, always call `list_edx_tags()` and check whether
the user's key terms appear exactly as EDX tags. If not, find the closest match and
use that tag instead of (or in addition to) the free-text term.

**Impact:** Eliminates the vocabulary mismatch problem for the most common cases
without adding any new infrastructure.

### 4. OR-based multi-term expansion for sparse results

**Where:** `agents/tools/edx_tools.py` → modify `search_edx_multi_criteria`

**What:** If an AND query returns zero results, automatically retry with OR logic
(`term1 OR term2 OR term3`) and flag the results as "partial match" in the return dict.

**Impact:** Prevents dead-end searches. Currently if a user asks for "lithium AND
Wyoming AND borehole" and no dataset covers all three, the agent returns nothing and
has to manually retry with fewer terms.

### 5. Semantic reranking (longer-term)

**What:** After retrieving candidates via Solr, embed both the user's query and each
dataset's title+description using a small local embedding model (e.g.,
`sentence-transformers/all-MiniLM-L6-v2`). Rerank by cosine similarity.

**Impact:** Catches semantic matches that lexical search misses — "subsurface samples"
matching "borehole data", "REE" matching "rare earth elements", etc.

**Cost:** Requires adding `sentence-transformers` as a dependency and local inference.
Latency adds ~100–500ms depending on hardware. Worth it once the other improvements
are in place and vocabulary mismatch is still causing misses.
