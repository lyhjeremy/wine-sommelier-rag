"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank
Fusion, then re-ordered by a cross-encoder reranker.

Why bother, when plain vector search already works?

* **Dense** embeddings are great at paraphrase — *"something like a Rhône but
  cheaper"* — but blur exact lexical signals: a specific grape, winery, or
  appellation the guest named.
* **BM25** nails those exact terms but is deaf to meaning: it can't tell that
  "brooding and structured" answers a request for a "bold" red.

Fusing the two rankings (RRF) gives us the union of their strengths, and a
**cross-encoder** — which reads the query and a candidate *together* rather than
comparing two independent vectors — then re-scores the shortlist for a final,
sharper ordering. `src/eval.py` measures exactly how much each stage helps.

The dense and sparse stages are cheap; the cross-encoder is the expensive part,
so we only ever run it on a small fused shortlist (`pool`), never the whole
corpus.
"""

from __future__ import annotations

import os
import pickle
import re
from functools import lru_cache

from .config import CHROMA_DIR, COLLECTION, DATA_DIR
from .embedder import embed_query
from .retriever import WineHit, meta_passes

# Small, fast cross-encoder trained on MS MARCO passage ranking. ~80 MB, CPU-fine.
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
_BM25_CACHE = DATA_DIR / "bm25.pkl"
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --- corpus + BM25 index -----------------------------------------------------
class Corpus:
    """All documents pulled once from Chroma, plus a cached BM25 index over them.

    BM25 needs the whole corpus in memory. For ~30k short reviews that is a few
    MB and builds in a second or two, so we pickle it next to the Chroma store
    and rebuild only when the collection's document count changes.
    """

    def __init__(self, ids, docs, metas, bm25):
        self.ids = ids
        self.docs = docs
        self.metas = metas
        self.bm25 = bm25
        self._pos = {i: n for n, i in enumerate(ids)}

    def doc(self, _id: str) -> str:
        return self.docs[self._pos[_id]]

    def meta(self, _id: str) -> dict:
        return self.metas[self._pos[_id]]

    def bm25_ranking(self, query: str, top: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [self.ids[i] for i in order[:top] if scores[i] > 0]


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION)
    count = coll.count()

    if _BM25_CACHE.exists():
        with open(_BM25_CACHE, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("count") == count:
            return Corpus(cached["ids"], cached["docs"], cached["metas"], cached["bm25"])

    from rank_bm25 import BM25Okapi

    got = coll.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    with open(_BM25_CACHE, "wb") as fh:
        pickle.dump({"count": count, "ids": ids, "docs": docs, "metas": metas, "bm25": bm25}, fh)
    return Corpus(ids, docs, metas, bm25)


# --- fusion + reranking ------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Combine several ranked id-lists into one. RRF rewards items that rank
    highly in *any* list without needing the raw scores to be comparable."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


class HybridRetriever:
    """Dense + BM25 → RRF → (optional) cross-encoder rerank → top ``k``."""

    def __init__(self, use_rerank: bool = True):
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.coll = client.get_collection(COLLECTION)
        except Exception as exc:
            raise SystemExit(
                "No wine index found. Build it first: python -m src.ingest"
            ) from exc
        self.corpus = load_corpus()
        self.use_rerank = use_rerank

    def _dense_ranking(self, query: str, top: int) -> list[str]:
        res = self.coll.query(query_embeddings=[embed_query(query)], n_results=top)
        return res["ids"][0]

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
        pool: int = 60,
    ) -> list[WineHit]:
        filters = dict(
            max_price=max_price, min_price=min_price, country=country,
            variety=variety, min_points=min_points,
        )
        dense = self._dense_ranking(query, pool)
        sparse = self.corpus.bm25_ranking(query, pool)
        fused = reciprocal_rank_fusion([dense, sparse])

        # Apply metadata filters, then keep a shortlist to (optionally) rerank.
        candidates = [i for i in fused if meta_passes(self.corpus.meta(i), **filters)]
        candidates = candidates[: max(pool, k)]
        if not candidates:
            return []

        if self.use_rerank:
            pairs = [(query, self.corpus.doc(i)) for i in candidates]
            scores = _reranker().predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
            return [
                WineHit(self.corpus.doc(i), self.corpus.meta(i), 0.0, float(s))
                for i, s in ranked[:k]
            ]

        return [
            WineHit(self.corpus.doc(i), self.corpus.meta(i), 0.0)
            for i in candidates[:k]
        ]
