"""How often does the substring rule for 'not oaky' drop a review that says it is unoaked?

    python eval/constraints/exclusion_check.py     # writes eval/constraints/exclusion_check.json

exclude_terms removes any review whose text contains the root word. For the two
'no oak' requests in the frozen set, count the reviews that pass every other
clause, mention 'oak', and mention it in a negated or neutral phrase.
"""
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.constraints import Constraints, violations  # noqa: E402

NEG = re.compile(r"unoaked|no oak|without oak|stainless|neutral oak|no new oak|not oaky|oak-free|free of oak")


def main():
    c = pickle.load(open(ROOT / "data" / "bm25.pkl", "rb"))
    qs = {q["id"]: q for q in json.loads((ROOT / "eval/constraints/queries.json").read_text())["queries"]}
    out = {}
    for qid in ("c03", "c15"):
        g = qs[qid]["gold"]
        rest = Constraints(**{k: v for k, v in g.items() if k != "exclude_terms"})
        pool = [(d, m) for d, m in zip(c["docs"], c["metas"]) if not violations(rest, d, m)]
        oak = [(d, m) for d, m in pool if "oak" in d.split("\n", 1)[-1].lower()]
        neg = [m["title"] for d, m in oak if NEG.search(d.lower())]
        out[qid] = {"q": qs[qid]["q"], "match_other_clauses": len(pool), "mention_oak": len(oak),
                    "oak_negated_or_neutral": len(neg), "examples": neg[:5]}
    (ROOT / "eval/constraints/exclusion_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
