"""Per-query retrieval eval on two frozen query sets, with saved rankings.

    python eval/run_eval.py

Set "v1" is the original 12 queries and rubric in src/eval.py (the numbers in
eval/results.csv). Set "v2" is the 30 held-out queries in eval/queries_v2.json,
committed before this script first ran. Both use the same three systems and
metrics as src/eval.py, at k = 10.

Writes:
    eval/per_query.csv      one row per (set, query, system)
    eval/summary.csv        mean metrics per (set, system), plus a paired
                            bootstrap 95% interval for hybrid+rerank minus dense
    eval/top10.json         the hybrid+rerank top 10 for every query, each hit
                            marked relevant or with the rubric clause it fails
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import eval as E  # noqa: E402

K = 10
FAMILIES = {"red": E._RED, "white": E._WHITE}


def v1_specs():
    out = []
    for s in E.QUERIES:
        s2 = dict(s)
        if "family" in s2:
            s2["family"] = "red" if s2["family"] is E._RED else "white"
        if "variety_has" in s2:
            s2["variety_any"] = [s2.pop("variety_has")]
        out.append(s2)
    return out


def failed_clause(spec, doc, meta):
    """The first rubric clause a review fails, or None if it is relevant."""
    v = str(meta.get("variety", "")).lower()
    if "family" in spec and not any(m in v for m in FAMILIES[spec["family"]]):
        return "wrong grape"
    if "variety_any" in spec and not any(x in v for x in spec["variety_any"]):
        return "wrong grape"
    if "country" in spec and meta.get("country") != spec["country"]:
        return "wrong place"
    if "province" in spec and meta.get("province") != spec["province"]:
        return "wrong place"
    price = meta.get("price")
    if "max_price" in spec and (price is None or price > spec["max_price"]):
        return "over budget" if price is not None else "no price listed"
    if "min_price" in spec and (price is None or price < spec["min_price"]):
        return "under price floor"
    if "kw" in spec and not any(k in doc.lower() for k in spec["kw"]):
        return "style words missing"
    return None


def boot_ci(diffs, n=10000, seed=0):
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    sysm = E.Systems()
    ids, docs, metas = sysm.corpus.ids, sysm.corpus.docs, sysm.corpus.metas
    v1_rubrics = v1_specs()
    # check: the translated v1 rubric marks exactly what src/eval.is_relevant marks
    for raw, spec in zip(E.QUERIES, v1_rubrics):
        for n in range(0, len(ids), 97):
            assert (failed_clause(spec, docs[n], metas[n]) is None) == E.is_relevant(raw, docs[n], metas[n])
    v2 = json.loads((ROOT / "eval" / "queries_v2.json").read_text())["queries"]
    sets = {"v1": v1_rubrics, "v2": v2}
    systems = {"dense": sysm.dense, "hybrid": sysm.hybrid, "hybrid+rerank": sysm.hybrid_rerank}
    metric_fns = {
        "P@10": lambda r, rel: E.precision_at_k(r, rel, K),
        "Recall@10": lambda r, rel: E.recall_at_k(r, rel, K),
        "MRR": E.mrr,
        "nDCG@10": lambda r, rel: E.ndcg_at_k(r, rel, K),
    }

    rows, top10 = [], []
    for set_name, specs in sets.items():
        for qi, spec in enumerate(specs, 1):
            rel = {ids[n] for n in range(len(ids)) if failed_clause(spec, docs[n], metas[n]) is None}
            if not rel:
                rows.append({"set": set_name, "qid": qi, "query": spec["q"], "system": "all",
                             "n_relevant": 0, **{m: "" for m in metric_fns}})
                continue
            for name, fn in systems.items():
                ranked = fn(spec["q"], max(K, 20))
                rows.append({"set": set_name, "qid": qi, "query": spec["q"], "system": name,
                             "n_relevant": len(rel),
                             **{m: round(f(ranked, rel), 4) for m, f in metric_fns.items()}})
                if name == "hybrid+rerank":
                    hits = []
                    for i in ranked[:K]:
                        m = sysm.corpus.meta(i)
                        hits.append({"id": i, "title": m.get("title"), "variety": m.get("variety"),
                                     "country": m.get("country"), "province": m.get("province"),
                                     "price": m.get("price"), "points": m.get("points"),
                                     "fails": failed_clause(spec, sysm.corpus.doc(i), m)})
                    top10.append({"set": set_name, "qid": qi, "query": spec["q"],
                                  "rubric": {k: v for k, v in spec.items() if k != "q"}, "hits": hits})

    out = ROOT / "eval"
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summ = []
    for set_name in sets:
        scored = [r for r in rows if r["set"] == set_name and r["system"] != "all"]
        n_q = len({r["qid"] for r in scored})
        for name in systems:
            rs = [r for r in scored if r["system"] == name]
            summ.append({"set": set_name, "system": name, "n_queries": n_q,
                         **{m: round(sum(r[m] for r in rs) / len(rs), 4) for m in metric_fns}})
        for m in ("P@10", "nDCG@10"):
            d = {r["qid"]: r[m] for r in scored if r["system"] == "dense"}
            h = {r["qid"]: r[m] for r in scored if r["system"] == "hybrid+rerank"}
            diffs = [h[q] - d[q] for q in d]
            lo, hi = boot_ci(diffs)
            summ.append({"set": set_name, "system": f"rerank minus dense, {m}, 95% bootstrap",
                         "n_queries": n_q, m: f"{sum(diffs)/len(diffs):.4f} [{lo:.4f}, {hi:.4f}]"})
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "system", "n_queries", *metric_fns])
        w.writeheader()
        w.writerows(summ)
    (out / "top10.json").write_text(json.dumps(top10, indent=1, ensure_ascii=False) + "\n")
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
