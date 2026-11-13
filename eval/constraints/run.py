"""Constraint extraction and pre-filtered retrieval on the frozen 28-request set.

    python eval/constraints/run.py --extract sonnet haiku   # LLM extraction (resumable)
    python eval/constraints/run.py                          # score every system, write results

Systems scored on eval/constraints/queries.json (all k = 10, same corpus as the
other benchmarks: the 30,000 reviews in data/bm25.pkl):

    none         hybrid + rerank on the raw sentence, no filters (the current app path)
    gold-post    the answer-key filters applied after fusion, the way --max-price etc.
                 are applied today (HybridRetriever filters the fused shortlist)
    rules        regex extractor (src/constraints.extract_rules), filters applied first
    llm-sonnet   Claude Sonnet extracts the schema, validated in code, filters first
    llm-haiku    the same on Claude Haiku
    gold         the answer-key filters applied first (ceiling for any extractor)

LLM calls go through `claude -p --output-format json`; the reported
total_cost_usd is the API-equivalent cost (nothing billed, Claude subscription).
Writes eval/constraints/{extractions.json, per_query.csv, summary.json, top10.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import eval as E  # noqa: E402
from src.constraints import Constraints, Vocab, extract_llm, extract_rules, violations  # noqa: E402
from src.filtered import FilteredHybrid  # noqa: E402
from src.hybrid import Corpus, _reranker, _tokenize, reciprocal_rank_fusion  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / "data-cache" / "wine-sommelier-rag" / "emb.npy"
K = 10
FIELDS = list(Constraints.model_fields)


class CliLLM:
    """`claude -p` with an explicit model and JSON output, logging cost and time."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def complete(self, prompt, system=None):
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json"]
        if system:
            cmd += ["--append-system-prompt", system]
        t0 = time.perf_counter()
        for attempt in range(3):
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=240)
            if r.returncode == 0:
                break
            time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(r.stderr[:300])
        out = json.loads(r.stdout)
        self.calls.append({"wall_s": round(time.perf_counter() - t0, 2),
                           "duration_ms": out.get("duration_ms"),
                           "cost_usd": out.get("total_cost_usd"),
                           "models": "+".join(sorted(out.get("modelUsage", {})))})
        return (out.get("result") or "").strip()


def load():
    c = pickle.load(open(ROOT / "data" / "bm25.pkl", "rb"))
    corpus = Corpus(c["ids"], c["docs"], c["metas"], c["bm25"])
    if CACHE.exists():
        emb = np.load(CACHE)
    else:
        from src.embedder import embed_documents
        emb = np.asarray(embed_documents(c["docs"]), dtype=np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, emb)
    return corpus, emb


def norm(c: Constraints) -> dict:
    d = c.model_dump()
    return {k: (sorted(x.lower() for x in v) if isinstance(v, list) else v) for k, v in d.items()}


def run_extract(models, queries, vocab):
    path = HERE / "extractions.json"
    ex = json.loads(path.read_text()) if path.exists() else {}
    for model in models:
        llm = CliLLM(model)
        ex.setdefault(model, {})
        for q in queries:
            if q["id"] in ex[model]:
                continue
            n0 = len(llm.calls)
            res = extract_llm(q["q"], llm, vocab)
            calls = llm.calls[n0:]
            ex[model][q["id"]] = {
                "constraints": res["constraints"].model_dump() if res["constraints"] else None,
                "valid": res["valid"], "repaired": res["repaired"], "errors": res["errors"],
                "raw": res["raw"], "calls": len(calls),
                "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 5),
                "wall_s": round(sum(c["wall_s"] for c in calls), 2),
                "models": sorted({c["models"] for c in calls}),
            }
            print(model, q["id"], res["valid"], res["repaired"], flush=True)
            path.write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    return ex


def post_filter(fh: FilteredHybrid, query, c, k=K, pool=80):
    """Today's flag path: fuse unfiltered top lists, then filter, then rerank."""
    dense = fh.search(query, None, k=pool, pool=pool, rerank=False)  # fused, unfiltered
    keep = [i for i in dense if not violations(c, fh.corpus.doc(i), fh.corpus.meta(i))]
    if not keep:
        return []
    s = _reranker().predict([(query, fh.corpus.doc(i)) for i in keep])
    return [i for i, _ in sorted(zip(keep, s), key=lambda t: t[1], reverse=True)[:k]]


def pct(xs, p):
    xs = sorted(xs)
    kk = (len(xs) - 1) * p
    lo, hi = int(kk), min(int(kk) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (kk - lo)


def score(queries, corpus, emb, ex):
    fh = FilteredHybrid(corpus, emb)
    docs, metas = corpus.docs, corpus.metas
    systems = ["none", "gold-post", "rules"] + [f"llm-{m}" for m in ex] + ["gold"]
    rows, top10, extract_acc = [], {}, {s: {f: 0 for f in FIELDS + ["all"]} for s in systems}
    timings = {s: [] for s in systems}
    for q in queries:
        gold = Constraints(**q["gold"])
        kw = q.get("kw")
        relevant = {i for i, d, m in zip(corpus.ids, docs, metas)
                    if not violations(gold, d, m) and (not kw or any(w in d.lower() for w in kw))}
        for s in systems:
            c = None
            if s == "rules":
                c = extract_rules(q["q"])
            elif s.startswith("llm-"):
                e = ex[s[4:]][q["id"]]
                c = Constraints(**e["constraints"]) if e["constraints"] else None
            elif s in ("gold", "gold-post"):
                c = gold
            t0 = time.perf_counter()
            ranked = post_filter(fh, q["q"], gold) if s == "gold-post" else fh.search(q["q"], c, k=K)
            timings[s].append(time.perf_counter() - t0)
            bad = [violations(gold, corpus.doc(i), corpus.meta(i)) for i in ranked]
            if c is not None and s not in ("gold", "gold-post"):
                a, b = norm(c), norm(gold)
                for f in FIELDS:
                    extract_acc[s][f] += a[f] == b[f]
                extract_acc[s]["all"] += a == b
            rows.append({"id": q["id"], "system": s, "returned": len(ranked),
                         "violating": sum(bool(v) for v in bad),
                         "clauses": "; ".join(sorted({x for v in bad for x in v})),
                         "relevant_in_top10": sum(i in relevant for i in ranked),
                         "n_relevant_in_corpus": len(relevant),
                         "ndcg10": round(E.ndcg_at_k(ranked, relevant, K), 4),
                         "extracted": json.dumps({k: v for k, v in (c.model_dump() if c else {}).items()
                                                  if v not in (None, [])}, ensure_ascii=False)})
            top10[f"{q['id']}|{s}"] = [{"title": corpus.meta(i)["title"],
                                         "variety": corpus.meta(i).get("variety"),
                                         "country": corpus.meta(i).get("country"),
                                         "province": corpus.meta(i).get("province"),
                                         "price": corpus.meta(i).get("price"),
                                         "violates": v} for i, v in zip(ranked, bad)]
    with open(HERE / "per_query.csv", "w", newline="") as fh_:
        w = csv.DictWriter(fh_, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "top10.json").write_text(json.dumps(top10, indent=1, ensure_ascii=False))

    n = len(queries)
    summ = {"date": time.strftime("%Y-%m-%d"), "machine": "Apple M4", "n_queries": n, "k": K,
            "corpus": len(corpus.ids), "systems": {}}
    for s in systems:
        r = [x for x in rows if x["system"] == s]
        slots = sum(x["returned"] for x in r)
        d = {"violation_rate": round(sum(x["violating"] for x in r) / slots, 4) if slots else None,
             "violating_slots": sum(x["violating"] for x in r), "returned_slots": slots,
             "queries_with_any_violation": sum(x["violating"] > 0 for x in r),
             "queries_returning_under_10": sum(x["returned"] < K for x in r),
             "ndcg10_mean": round(statistics.mean(x["ndcg10"] for x in r), 4),
             "retrieval_ms_median": round(1000 * statistics.median(timings[s]), 1),
             "retrieval_ms_p95": round(1000 * pct(timings[s], 0.95), 1)}
        if s in extract_acc and s not in ("none", "gold", "gold-post"):
            d["extraction_exact_queries"] = f"{extract_acc[s]['all']} of {n}"
            d["extraction_field_accuracy"] = {f: round(extract_acc[s][f] / n, 3) for f in FIELDS}
        if s.startswith("llm-"):
            e = ex[s[4:]]
            costs = [v["cost_usd"] for v in e.values()]
            walls = [v["wall_s"] for v in e.values()]
            d.update({"model_flag": s[4:], "models_seen": sorted({m for v in e.values() for m in v["models"]}),
                      "valid_first_try": sum(v["valid"] and not v["repaired"] for v in e.values()),
                      "repaired": sum(v["repaired"] for v in e.values()),
                      "invalid_after_repair": sum(not v["valid"] for v in e.values()),
                      "llm_calls": sum(v["calls"] for v in e.values()),
                      "cost_per_query_median": round(statistics.median(costs), 4),
                      "cost_per_query_p95": round(pct(costs, 0.95), 4),
                      "extract_s_median": round(statistics.median(walls), 2),
                      "extract_s_p95": round(pct(walls, 0.95), 2)})
        summ["systems"][s] = d
    summ["cost_note"] = ("total_cost_usd from claude -p --output-format json: API-equivalent cost "
                         "reported by the Claude CLI, nothing billed (Claude subscription)")
    (HERE / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", nargs="*")
    a = ap.parse_args()
    queries = json.loads((HERE / "queries.json").read_text())["queries"]
    corpus, emb = load()
    vocab = Vocab(corpus.metas)
    if a.extract:
        run_extract(a.extract, queries, vocab)
        return
    p = HERE / "extractions.json"
    ex = json.loads(p.read_text()) if p.exists() else {}
    score(queries, corpus, emb, ex)


if __name__ == "__main__":
    main()
