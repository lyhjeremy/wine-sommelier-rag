"""Time the retrieval path per query, stage by stage, on this machine.

    python eval/latency.py

Replays HybridRetriever.search (pool 60, k 6, rerank on, no metadata filters)
for all 42 frozen queries (eval: v1 + v2), after one warm-up pass, three passes.
Writes eval/latency_retrieval.csv (one row per query per pass, in ms) and
prints median and p95 per stage.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import eval as E  # noqa: E402
from src.embedder import _model, embed_query  # noqa: E402
from src.hybrid import HybridRetriever, _reranker, reciprocal_rank_fusion  # noqa: E402

POOL, K, PASSES = 60, 6, 3


def pct(xs, p):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def one(r, q):
    t = [time.perf_counter()]
    vec = embed_query(q); t.append(time.perf_counter())
    dense = r.coll.query(query_embeddings=[vec], n_results=POOL)["ids"][0]; t.append(time.perf_counter())
    sparse = r.corpus.bm25_ranking(q, POOL); t.append(time.perf_counter())
    cands = reciprocal_rank_fusion([dense, sparse])[:POOL]; t.append(time.perf_counter())
    scores = _reranker().predict([(q, r.corpus.doc(i)) for i in cands]); t.append(time.perf_counter())
    sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)[:K]
    ms = [(b - a) * 1000 for a, b in zip(t, t[1:])]
    return dict(zip(["embed", "dense_search", "bm25", "rrf", "rerank"], ms)) | {"total": sum(ms)}


def main():
    r = HybridRetriever(use_rerank=True)
    qs = [s["q"] for s in E.QUERIES] + [s["q"] for s in json.loads((ROOT / "eval" / "queries_v2.json").read_text())["queries"]]
    for q in qs:
        one(r, q)  # warm-up: model load, caches
    rows = []
    for p in range(1, PASSES + 1):
        for qi, q in enumerate(qs, 1):
            rows.append({"pass": p, "qid": qi, "query": q, **{k: round(v, 2) for k, v in one(r, q).items()}})
    with open(ROOT / "eval" / "latency_retrieval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    meta = {"machine": chip, "python": platform.python_version(),
            "embed_device": str(_model().device), "rerank_device": str(_reranker().model.device),
            "n_samples": len(rows), "n_queries": len(qs), "passes": PASSES, "pool": POOL, "k": K,
            "date": time.strftime("%Y-%m-%d")}
    for st in ["embed", "dense_search", "bm25", "rrf", "rerank", "total"]:
        xs = [x[st] for x in rows]
        meta[st] = {"median_ms": round(pct(xs, 50), 1), "p95_ms": round(pct(xs, 95), 1)}
    (ROOT / "eval" / "latency_retrieval_summary.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
