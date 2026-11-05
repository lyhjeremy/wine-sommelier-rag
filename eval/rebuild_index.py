"""Rebuild data/chroma from the corpus cached in data/bm25.pkl.

    python eval/rebuild_index.py

data/bm25.pkl holds the exact ids, documents and metadata of the indexed
30,000 reviews. If the Chroma store is lost (on 2026-11-04 the local copy had
its HNSW files but no chroma.sqlite3), this re-embeds those same documents so
the evals run on the same corpus. The benchmarks use exact dense search, so they
give the same numbers on any rebuild; the app's HNSW search does not (see
eval/rebuild_spread.py).
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from src.config import CHROMA_DIR, COLLECTION  # noqa: E402
from src.embedder import embed_documents  # noqa: E402

c = pickle.load(open(ROOT / "data" / "bm25.pkl", "rb"))
coll = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
    COLLECTION, metadata={"hnsw:space": "cosine"})
for s in range(0, len(c["ids"]), 512):
    coll.add(ids=c["ids"][s:s + 512], documents=c["docs"][s:s + 512], metadatas=c["metas"][s:s + 512],
             embeddings=embed_documents(c["docs"][s:s + 512]))
print(coll.count())
