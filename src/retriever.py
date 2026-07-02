"""Semantic retrieval over the wine index, with metadata filters and reranking."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CHROMA_DIR, COLLECTION
from .embedder import embed_query


@dataclass
class WineHit:
    doc: str
    meta: dict
    distance: float
    # Populated by the hybrid pipeline: the cross-encoder's relevance score for
    # this (query, document) pair. ``None`` for plain dense retrieval.
    rerank_score: float | None = None

    @property
    def points(self) -> int | None:
        return self.meta.get("points")

    @property
    def price(self) -> float | None:
        return self.meta.get("price")

    def citation(self) -> str:
        bits = [self.meta.get("title", "Unknown wine")]
        if self.points:
            bits.append(f"{self.points} pts")
        if self.price:
            bits.append(f"${self.price:g}")
        return " · ".join(bits)


def meta_passes(
    meta: dict,
    *,
    max_price: float | None = None,
    min_price: float | None = None,
    country: str | None = None,
    variety: str | None = None,
    min_points: int | None = None,
) -> bool:
    """In-memory equivalent of ``Retriever._where`` for candidates that did not
    come straight from Chroma (e.g. the BM25 side of the hybrid retriever)."""
    price = meta.get("price")
    points = meta.get("points")
    if max_price is not None and (price is None or price > float(max_price)):
        return False
    if min_price is not None and (price is None or price < float(min_price)):
        return False
    if min_points is not None and (points is None or points < int(min_points)):
        return False
    if country and meta.get("country") != country:
        return False
    if variety and meta.get("variety") != variety:
        return False
    return True


class Retriever:
    def __init__(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.coll = client.get_collection(COLLECTION)
        except Exception as exc:  # collection missing
            raise SystemExit(
                "No wine index found. Build it first: python -m src.ingest"
            ) from exc

    def _where(self, max_price, min_price, country, variety, min_points) -> dict | None:
        clauses = []
        if max_price is not None:
            clauses.append({"price": {"$lte": float(max_price)}})
        if min_price is not None:
            clauses.append({"price": {"$gte": float(min_price)}})
        if min_points is not None:
            clauses.append({"points": {"$gte": int(min_points)}})
        if country:
            clauses.append({"country": {"$eq": country}})
        if variety:
            clauses.append({"variety": {"$eq": variety}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(
        self,
        query: str,
        k: int = 6,
        *,
        max_price: float | None = None,
        min_price: float | None = None,
        country: str | None = None,
        variety: str | None = None,
        min_points: int | None = None,
        pool: int = 40,
    ) -> list[WineHit]:
        """Retrieve `pool` candidates by vector similarity, then rerank to top `k`.

        Rerank blends semantic similarity with the reviewer's rating so that,
        among equally relevant wines, better-rated bottles surface first.
        """
        where = self._where(max_price, min_price, country, variety, min_points)
        res = self.coll.query(
            query_embeddings=[embed_query(query)],
            n_results=max(pool, k),
            where=where,
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        hits = [WineHit(d, m, dist) for d, m, dist in zip(docs, metas, dists)]

        def score(h: WineHit) -> float:
            sim = 1.0 - h.distance                      # cosine similarity in [0,1]
            quality = ((h.points or 85) - 80) / 20.0    # ~0..1 across 80–100 pts
            return 0.8 * sim + 0.2 * quality

        hits.sort(key=score, reverse=True)
        return hits[:k]
