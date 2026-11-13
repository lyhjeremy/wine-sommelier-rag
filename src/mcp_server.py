"""MCP server: filtered wine search over stdio, for Claude Desktop or Claude Code.

Uses the official `mcp` Python SDK (version 2, MCPServer). Two tools:

  search_wines       hybrid search (dense + BM25, cross-encoder rerank) over the
                     30,000 indexed reviews, with filters applied before ranking
  parse_constraints  the deterministic rules extractor: a request in, the filters
                     it states out (no model call)

Filters go through the same Constraints schema and vocabulary check as the eval
(src/constraints.py), so an unknown country or grape comes back as an error the
client can correct, rather than an empty result.

    python src/mcp_server.py          # speaks MCP on stdin/stdout
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.constraints import Constraints, Vocab, check_vocab, extract_rules, violations  # noqa: E402

server = MCPServer(
    name="wine-search",
    log_level="WARNING",
    instructions=(
        "Search 30,000 Wine Enthusiast reviews. Put hard limits from the user's request "
        "(price, country, US state or region, grape, color, exclusions) in the filter "
        "arguments, not in the query text; the query should carry style and flavor words. "
        "Quote titles, scores and prices exactly as returned."),
)


@lru_cache(maxsize=1)
def _engine():
    from src.filtered import FilteredHybrid

    fh = FilteredHybrid.from_cache()
    return fh, Vocab(fh.corpus.metas)


def _hit(fh, i, c):
    m = fh.corpus.meta(i)
    return {"title": m.get("title"), "variety": m.get("variety"), "country": m.get("country"),
            "province": m.get("province"), "points": m.get("points"), "price": m.get("price"),
            "review": fh.corpus.doc(i).split("\n", 1)[-1],
            "violates": violations(c, fh.corpus.doc(i), m)}


@server.tool(description=(
    "Search wine reviews by style with hard filters applied before ranking. query: style "
    "and flavor words. Filters (all optional): max_price, min_price (USD), min_points, "
    "countries (e.g. 'US', 'Italy'), provinces (US state or region: 'Oregon', 'Bordeaux'), "
    "varieties (grape names, substring match), color ('red', 'white', 'rosé', 'sparkling'), "
    "exclude_countries, exclude_varieties, exclude_terms (root words such as 'oak'). "
    "Returns up to k reviews (default 6) and the filters as applied."))
def search_wines(query: str, max_price: float | None = None, min_price: float | None = None,
                 min_points: int | None = None, countries: list[str] | None = None,
                 provinces: list[str] | None = None, varieties: list[str] | None = None,
                 color: str | None = None, exclude_countries: list[str] | None = None,
                 exclude_varieties: list[str] | None = None,
                 exclude_terms: list[str] | None = None, k: int = 6) -> dict:
    fh, vocab = _engine()
    raw = {"max_price": max_price, "min_price": min_price, "min_points": min_points,
           "countries": countries or [], "provinces": provinces or [], "varieties": varieties or [],
           "color": color, "exclude_countries": exclude_countries or [],
           "exclude_varieties": exclude_varieties or [], "exclude_terms": exclude_terms or []}
    try:
        c, errs = check_vocab(Constraints(**raw), vocab)
    except Exception as e:  # pydantic validation error
        return {"error": str(e)[:600]}
    if errs:
        return {"error": "filters not in the index", "details": errs}
    ids = fh.search(query, c, k=max(1, min(int(k), 20)))
    return {"filters": {f: v for f, v in c.model_dump().items() if v not in (None, [])},
            "matching_reviews": int(fh.mask(c).sum()),
            "results": [_hit(fh, i, c) for i in ids]}


@server.tool(description=(
    "Extract the hard constraints a wine request states (price, place, grape, color, "
    "exclusions) with the deterministic rules parser. No model call."))
def parse_constraints(request: str) -> dict:
    c = extract_rules(request)
    return {f: v for f, v in c.model_dump().items() if v not in (None, [])}


if __name__ == "__main__":
    server.run("stdio")
