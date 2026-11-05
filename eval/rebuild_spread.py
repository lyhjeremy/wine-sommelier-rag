"""How far the retrieval metrics move when the Chroma index is rebuilt.

    python eval/rebuild_spread.py [N]     # default N = 5; writes eval/rebuild_spread.csv

The app's dense search is Chroma's HNSW index, which is approximate. Its graph
comes out different on every build, even from identical embeddings added in the
same order, so for some queries it returns different neighbors, and the dense
and hybrid metrics move with it. This script takes the embeddings already stored
in data/chroma, adds them in 512-row batches to N fresh collections with the
same settings src/ingest.py uses, and scores both query sets with the app's
HNSW search on each. It also writes one row per set and system for the exact
search that src/eval.py and eval/run_eval.py use, which does not depend on the
build.
"""
import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from src import eval as E  # noqa: E402
from src.config import CHROMA_DIR, COLLECTION  # noqa: E402

spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
METRICS = {
    "P@10": lambda r, rel: E.precision_at_k(r, rel, 10),
    "Recall@10": lambda r, rel: E.recall_at_k(r, rel, 10),
    "MRR": E.mrr,
    "nDCG@10": lambda r, rel: E.ndcg_at_k(r, rel, 10),
}


def score(sysm, sets, rels):
    systems = {"dense": sysm.dense, "hybrid": sysm.hybrid, "hybrid+rerank": sysm.hybrid_rerank}
    out = []
    for set_name, specs in sets.items():
        for name, fn in systems.items():
            vals = {m: [] for m in METRICS}
            for s, rel in zip(specs, rels[set_name]):
                if not rel:
                    continue
                ranked = fn(s["q"], 20)
                for m, f in METRICS.items():
                    vals[m].append(f(ranked, rel))
            out.append({"set": set_name, "system": name,
                        **{m: round(sum(v) / len(v), 4) for m, v in vals.items()}})
    return out


def main():
    exact = E.Systems()
    ids, docs, metas = exact.corpus.ids, exact.corpus.docs, exact.corpus.metas
    sets = {"v1": R.v1_specs(),
            "v2": __import__("json").loads((ROOT / "eval" / "queries_v2.json").read_text())["queries"]}
    rels = {k: [{ids[n] for n in range(len(ids)) if R.failed_clause(s, docs[n], metas[n]) is None}
                for s in specs] for k, specs in sets.items()}
    src = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    got = src.get(include=["embeddings", "documents", "metadatas"])

    rows = [{"index": "exact", **r} for r in score(exact, sets, rels)]
    print(rows[-1])
    with tempfile.TemporaryDirectory() as tmp:
        for b in range(1, N + 1):
            coll = chromadb.PersistentClient(path=f"{tmp}/b{b}").get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})
            for s in range(0, len(got["ids"]), 512):
                coll.add(ids=got["ids"][s:s + 512], embeddings=got["embeddings"][s:s + 512],
                         documents=got["documents"][s:s + 512], metadatas=got["metadatas"][s:s + 512])
            sysm = E.Systems.__new__(E.Systems)
            sysm.coll, sysm.corpus = coll, exact.corpus
            sysm.dense = sysm.dense_hnsw
            for r in score(sysm, sets, rels):
                rows.append({"index": f"hnsw build {b}", **r})
            print(rows[-1])
    path = ROOT / "eval" / "rebuild_spread.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "set", "system", *METRICS])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} (chromadb {chromadb.__version__})")


if __name__ == "__main__":
    main()
