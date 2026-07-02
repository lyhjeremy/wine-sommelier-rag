"""Retrieval-quality evaluation: does hybrid + reranking actually beat plain
vector search, and by how much?

There is no human-labelled relevance set for these 130k reviews, so we define
relevance by an explicit, reproducible **rubric** per query: objective metadata
constraints (grape family, price band, country) plus, where a request is about
*style*, a set of descriptor keywords. A review is "relevant" to a query if it
satisfies that rubric. This is a proxy for human judgement — we say so plainly —
but it is transparent, deterministic, and identical across the three systems, so
the *relative* comparison is fair.

Systems compared (all over the same index, no metadata pre-filtering — we are
testing whether a free-text query alone surfaces the right wines):

    dense        vector search only (Chroma / MiniLM)
    hybrid       dense + BM25, fused with Reciprocal Rank Fusion
    hybrid+rr    hybrid, then a cross-encoder reranks the shortlist

Run:
    python -m src.eval            # prints a table, writes eval/results.csv + a figure
"""

from __future__ import annotations

import csv
import math

from .config import CHROMA_DIR, COLLECTION, ROOT
from .embedder import embed_query
from .hybrid import _reranker, load_corpus, reciprocal_rank_fusion

EVAL_DIR = ROOT / "eval"

# Grape families identified by substrings in the `variety` field.
_RED = ("cabernet", "merlot", "syrah", "shiraz", "malbec", "pinot noir", "zinfandel",
        "tempranillo", "sangiovese", "grenache", "nebbiolo", "red blend", "petite sirah",
        "barbera", "mourvèdre", "carmenère", "carmenère", "bordeaux-style red")
_WHITE = ("chardonnay", "sauvignon blanc", "riesling", "pinot gris", "pinot grigio",
          "chenin blanc", "gewürztraminer", "viognier", "albariño", "grüner", "white blend",
          "sémillon", "semillon", "moscato", "chablis", "verdejo", "vermentino")


def _is_family(meta: dict, markers: tuple[str, ...]) -> bool:
    v = str(meta.get("variety", "")).lower()
    return any(m in v for m in markers)


# Each query: a free-text request + a rubric that defines which reviews count as
# relevant. `kw` = descriptor keywords (any-match against the review text).
QUERIES = [
    {"q": "a bold, full-bodied red for a grilled steak dinner",
     "family": _RED, "kw": ("bold", "full-bodied", "full bodied", "powerful", "rich",
                            "robust", "structured", "tannic", "concentrated", "intense")},
    {"q": "a crisp, mineral white to go with fresh oysters",
     "family": _WHITE, "kw": ("crisp", "mineral", "minerality", "citrus", "saline",
                              "zesty", "bright acidity", "lemon", "lime")},
    {"q": "an elegant, silky Pinot Noir with red-berry fruit",
     "variety_has": "pinot noir", "kw": ("silky", "elegant", "cherry", "raspberry",
                                         "red berry", "red-berry", "strawberry", "supple")},
    {"q": "an affordable Argentine Malbec under $20",
     "variety_has": "malbec", "country": "Argentina", "max_price": 20},
    {"q": "a dry Riesling from Germany with racy acidity",
     "variety_has": "riesling", "country": "Germany",
     "kw": ("dry", "racy", "acidity", "citrus", "lime", "petrol", "mineral")},
    {"q": "a rich, oaky California Chardonnay with buttery notes",
     "variety_has": "chardonnay", "country": "US",
     "kw": ("oak", "oaky", "butter", "buttery", "creamy", "toast", "vanilla", "rich")},
    {"q": "a spicy, peppery Syrah from the Rhône",
     "variety_has": "syrah", "kw": ("spice", "spicy", "pepper", "peppery", "smoked",
                                    "savory", "black pepper", "meaty")},
    {"q": "a smooth, easy-drinking red under $15 for a weeknight",
     "family": _RED, "max_price": 15, "kw": ("smooth", "easy", "soft", "juicy",
                                             "approachable", "round", "supple", "everyday")},
    {"q": "a sparkling wine for a celebration",
     "kw": ("sparkling", "bubbles", "brut", "champagne", "mousse", "effervescent", "fizz")},
    {"q": "a sweet dessert wine to serve with a fruit tart",
     "kw": ("sweet", "dessert", "honey", "honeyed", "botrytis", "late harvest",
            "luscious", "sauternes", "unctuous", "sticky")},
    {"q": "a age-worthy Bordeaux-style red blend for the cellar",
     "kw": ("age", "aging", "cellar", "structured", "tannic", "firm", "backbone",
            "long", "will improve", "decade"), "family": _RED},
    {"q": "a light, aromatic white with floral notes for a warm afternoon",
     "family": _WHITE, "kw": ("aromatic", "floral", "flowers", "blossom", "perfumed",
                              "light", "fragrant", "jasmine", "honeysuckle")},
]


def is_relevant(spec: dict, doc: str, meta: dict) -> bool:
    if "family" in spec and not _is_family(meta, spec["family"]):
        return False
    if "variety_has" in spec and spec["variety_has"] not in str(meta.get("variety", "")).lower():
        return False
    if "country" in spec and meta.get("country") != spec["country"]:
        return False
    if "max_price" in spec:
        price = meta.get("price")
        if price is None or price > spec["max_price"]:
            return False
    if "kw" in spec:
        text = doc.lower()
        if not any(k in text for k in spec["kw"]):
            return False
    return True


# --- metrics -----------------------------------------------------------------
def precision_at_k(ranked, relevant, k):
    top = ranked[:k]
    return sum(1 for i in top if i in relevant) / k if top else 0.0


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def mrr(ranked, relevant):
    for rank, i in enumerate(ranked, 1):
        if i in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(r + 1) for r, i in enumerate(ranked[:k], 1) if i in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# --- retrieval systems (return ranked id lists, no metadata filtering) --------
class Systems:
    def __init__(self):
        import chromadb

        self.coll = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
        self.corpus = load_corpus()

    def dense(self, query, top):
        return self.coll.query(query_embeddings=[embed_query(query)], n_results=top)["ids"][0]

    def hybrid(self, query, top, pool=80):
        dense = self.dense(query, pool)
        sparse = self.corpus.bm25_ranking(query, pool)
        return reciprocal_rank_fusion([dense, sparse])[:top]

    def hybrid_rerank(self, query, top, pool=80):
        fused = self.hybrid(query, pool, pool=pool)
        pairs = [(query, self.corpus.doc(i)) for i in fused]
        scores = _reranker().predict(pairs)
        order = sorted(zip(fused, scores), key=lambda t: t[1], reverse=True)
        return [i for i, _ in order[:top]]


def evaluate(k: int = 10):
    sys = Systems()
    ids, docs, metas = sys.corpus.ids, sys.corpus.docs, sys.corpus.metas

    # Precompute the relevant id-set for each query over the whole corpus.
    rel_sets = []
    for spec in QUERIES:
        rel = {ids[n] for n in range(len(ids)) if is_relevant(spec, docs[n], metas[n])}
        rel_sets.append(rel)

    systems = {"dense": sys.dense, "hybrid": sys.hybrid, "hybrid+rerank": sys.hybrid_rerank}
    agg = {name: {"P@%d" % k: [], "Recall@%d" % k: [], "MRR": [], "nDCG@%d" % k: []}
           for name in systems}

    for spec, rel in zip(QUERIES, rel_sets):
        if not rel:
            continue  # skip a query no review satisfies (keeps metrics honest)
        for name, fn in systems.items():
            ranked = fn(spec["q"], max(k, 20))
            agg[name]["P@%d" % k].append(precision_at_k(ranked, rel, k))
            agg[name]["Recall@%d" % k].append(recall_at_k(ranked, rel, k))
            agg[name]["MRR"].append(mrr(ranked, rel))
            agg[name]["nDCG@%d" % k].append(ndcg_at_k(ranked, rel, k))

    means = {name: {m: sum(v) / len(v) if v else 0.0 for m, v in metrics.items()}
             for name, metrics in agg.items()}
    return means, k


def _print_table(means, k):
    metrics = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    head = f"{'system':16s}" + "".join(f"{m:>12s}" for m in metrics)
    print("\n" + head)
    print("-" * len(head))
    for name, mv in means.items():
        print(f"{name:16s}" + "".join(f"{mv[m]:>12.3f}" for m in metrics))
    print()


def _write_csv(means, k):
    EVAL_DIR.mkdir(exist_ok=True)
    path = EVAL_DIR / "results.csv"
    metrics = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system"] + metrics)
        for name, mv in means.items():
            w.writerow([name] + [f"{mv[m]:.4f}" for m in metrics])
    print(f"wrote {path}")


def _write_figure(means, k):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not installed — skipping figure)")
        return

    metrics = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    systems = list(means)
    colors = {"dense": "#9aa0a6", "hybrid": "#6b8fb5", "hybrid+rerank": "#7a1f2b"}
    x = range(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, name in enumerate(systems):
        vals = [means[name][m] for m in metrics]
        offs = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=name, color=colors.get(name, None))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Retrieval quality: dense vs. hybrid vs. hybrid + cross-encoder rerank",
                 pad=30)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    EVAL_DIR.mkdir(exist_ok=True)
    path = EVAL_DIR / "retrieval_quality.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    ap.add_argument("-k", type=int, default=10, help="cutoff for @k metrics")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    means, k = evaluate(args.k)
    _print_table(means, k)
    _write_csv(means, k)
    if not args.no_figure:
        _write_figure(means, k)


if __name__ == "__main__":
    main()
