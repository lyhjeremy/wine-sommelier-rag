"""Hybrid retrieval with constraints applied before ranking.

The existing ``HybridRetriever`` fuses the top 60 dense and top 60 BM25 hits and
only then drops the ones that fail a filter, so a strict filter ("Oregon Pinot
under $20") can leave few or no results even when matching reviews exist. Here
the constraint is a mask over the whole corpus first; dense search (exact cosine),
BM25 and the cross-encoder only ever see reviews that satisfy it.
"""

from __future__ import annotations

import numpy as np

from .constraints import Constraints, passes
from .hybrid import _reranker, reciprocal_rank_fusion


class FilteredHybrid:
    def __init__(self, corpus, emb: np.ndarray):
        """``corpus`` is ``hybrid.Corpus``; ``emb`` holds its normalized embeddings, same order."""
        self.corpus = corpus
        self.ids = np.array(corpus.ids)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    @classmethod
    def from_chroma(cls):
        import chromadb

        from .config import CHROMA_DIR, COLLECTION
        from .hybrid import load_corpus

        corpus = load_corpus()
        got = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION).get(
            ids=corpus.ids, include=["embeddings"])
        pos = {i: n for n, i in enumerate(got["ids"])}
        emb = np.asarray(got["embeddings"], dtype=np.float32)[[pos[i] for i in corpus.ids]]
        return cls(corpus, emb)

    @classmethod
    def from_cache(cls):
        """Load from data/bm25.pkl (ids, documents, metadata, BM25) and an embedding
        cache next to it, embedding the documents once if the cache is missing.
        Does not need the Chroma store."""
        import pickle

        from .config import DATA_DIR
        from .hybrid import Corpus

        c = pickle.load(open(DATA_DIR / "bm25.pkl", "rb"))
        corpus = Corpus(c["ids"], c["docs"], c["metas"], c["bm25"])
        path = DATA_DIR / "emb.npy"
        if path.exists():
            emb = np.load(path)
        else:
            from .embedder import embed_documents

            emb = np.asarray(embed_documents(c["docs"]), dtype=np.float32)
            np.save(path, emb)
        return cls(corpus, emb)

    def mask(self, c: Constraints | None) -> np.ndarray:
        if c is None or c.is_empty():
            return np.ones(len(self.ids), dtype=bool)
        return np.array([passes(c, d, m) for d, m in zip(self.corpus.docs, self.corpus.metas)])

    def search(self, query: str, c: Constraints | None = None, k: int = 10, pool: int = 80,
               rerank: bool = True) -> list[str]:
        from .embedder import embed_query

        ok = self.mask(c)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return []
        # numpy's Accelerate matmul on macOS can raise spurious divide-by-zero
        # warnings on finite float32 input; the products are fine.
        with np.errstate(all="ignore"):
            sim = self.emb[idx] @ np.asarray(embed_query(query), dtype=np.float32)
        order = idx[np.lexsort((self.ids[idx], -sim))][:pool]
        dense = self.ids[order].tolist()
        from .hybrid import _tokenize

        scores = np.asarray(self.corpus.bm25.get_scores(_tokenize(query)))
        scores[~ok] = 0
        top = np.argsort(-scores, kind="stable")[:pool]
        sparse = [self.ids[i] for i in top if scores[i] > 0]
        fused = reciprocal_rank_fusion([dense, sparse])[:pool]
        if not rerank:
            return fused[:k]
        s = _reranker().predict([(query, self.corpus.doc(i)) for i in fused])
        return [i for i, _ in sorted(zip(fused, s), key=lambda t: t[1], reverse=True)[:k]]
