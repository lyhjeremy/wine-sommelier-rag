"""Group the hybrid+rerank misses in eval/top10.json by type.

    python eval/error_analysis.py      # writes eval/error_analysis.json

A miss is a top-10 hit that fails its query's rubric. The rubric clause it fails
gives the type. One extra rule: a "wrong grape" hit whose title carries the
appellation the query named (a Chianti Classico labeled "Red Blend" in the
dataset, say) is counted as a rubric miss, not a retrieval miss, because the
wine is what the guest asked for and only the variety field disagrees.
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPELLATIONS = ["Barolo", "Champagne", "Rioja", "Chianti Classico", "Amarone", "Beaujolais",
                "Brunello", "Sancerre", "Sauternes", "Burgundy", "Napa"]


def kind(q, h):
    f = h["fails"]
    if f == "wrong grape":
        for a in APPELLATIONS:
            if a.lower() in q["query"].lower() and a.lower() in (h["title"] or "").lower():
                return "rubric miss: right wine, variety field disagrees"
    return {"over budget": "price in the sentence ignored",
            "no price listed": "price in the sentence ignored",
            "wrong place": "wrong region or country",
            "wrong grape": "wrong grape or color",
            "style words missing": "style words missing"}[f]


def generation():
    """Read eval/generation_runs.jsonl. A $ amount that is no source's price is
    classed as the guest's own budget (it appears in the query) or arithmetic
    (the difference between two source prices, or between a price and the
    budget); anything left over would be an invented price."""
    import re
    runs = [json.loads(l) for l in (ROOT / "eval" / "generation_runs.jsonl").read_text().splitlines() if l.strip()]
    res = {"answers": len(runs), "citation_out_of_range": 0, "invented_price": [], "budget_or_arithmetic": [],
           "says_nothing_fits_fully": [], "points_not_in_sources": 0}
    for r in runs:
        ans = r["answer"] or ""
        n = len(r["sources"])
        if any(not 1 <= int(c) <= n for c in re.findall(r"\[(\d+)\]", ans)):
            res["citation_out_of_range"] += 1
        prices = {s["price"] for s in r["sources"] if s["price"]}
        budget = {float(x) for x in re.findall(r"\$(\d+)", r["query"])}
        ok = prices | budget | {abs(a - b) for a in prices | budget for b in prices | budget}
        for amt in re.findall(r"\$(\d+(?:\.\d+)?)", ans):
            v = float(amt)
            if v in prices:
                continue
            (res["budget_or_arithmetic"] if v in ok else res["invented_price"]).append({"query": r["query"], "amount": v})
        pts = {s["points"] for s in r["sources"]}
        res["points_not_in_sources"] += sum(int(x) not in pts for x in re.findall(r"\b(\d{2,3})\s*(?:points?|pts)\b", ans))
        first = ans.strip().split("\n")[0].lower()
        if re.search(r"none of (these|the)|no (wine|bottle)s? (here )?(meets|matches|fits)|doesn't (quite )?(fit|match)", first):
            res["says_nothing_fits_fully"].append({"query": r["query"], "first_line": ans.strip().split("\n")[0]})
    return res


def main():
    top = json.loads((ROOT / "eval" / "top10.json").read_text())
    out = {}
    for st in ("v1", "v2"):
        qs = [q for q in top if q["set"] == st]
        counts, examples, per_q = collections.Counter(), {}, collections.defaultdict(collections.Counter)
        n = 0
        for q in qs:
            for rank, h in enumerate(q["hits"], 1):
                n += 1
                if h["fails"] is None:
                    continue
                k = kind(q, h)
                counts[k] += 1
                per_q[k][q["qid"]] += 1
                examples.setdefault(k, {"query": q["query"], "rank": rank, "title": h["title"],
                                        "variety": h["variety"], "province": h["province"],
                                        "country": h["country"], "price": h["price"]})
        out[st] = {"queries": len(qs), "hits": n, "misses": sum(counts.values()),
                   "by_type": dict(counts.most_common()),
                   "queries_affected": {k: len(v) for k, v in per_q.items()},
                   "examples": examples}
    out["generation"] = generation()
    (ROOT / "eval" / "error_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({s: {k: v for k, v in d.items() if k != "examples"} for s, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
