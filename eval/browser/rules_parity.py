"""Check the JS rules extractor (assets/ai/wine-search.js) against the Python one.

    python eval/browser/rules_parity.py

Runs both on the 28 frozen constrained requests plus the test sentences, through
Node (no packages needed), and writes eval/browser/rules_parity.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constraints import extract_rules  # noqa: E402

EXTRA = ["a Pinot Noir from Oregon for less than thirty bucks",
         "a Spanish white between $12 and $22, not Albariño",
         "a Merlot that's not from Chile, nothing buttery",
         "a rosé from Provence", "a 92-point Napa Cab under a hundred",
         "Grüner from the Wachau, not too sweet", "bubbly under twenty-five"]


def norm(d):
    return {k: (sorted(v) if isinstance(v, list) else v) for k, v in d.items() if v not in (None, [])}


def main():
    qs = [q["q"] for q in json.loads((ROOT / "eval/constraints/queries.json").read_text())["queries"]] + EXTRA
    js = ("import { extractRules } from '" + (ROOT / "assets/ai/wine-search.js").as_uri() + "';"
          "const qs = JSON.parse(process.argv[1]);"
          "console.log(JSON.stringify(qs.map(extractRules)));")
    out = subprocess.run(["node", "--input-type=module", "-e", js, json.dumps(qs)],
                         capture_output=True, text=True, check=True).stdout
    js_res = json.loads(out)
    rows, same = [], 0
    for q, j in zip(qs, js_res):
        p = norm(extract_rules(q).model_dump())
        j = norm({k: (float(v) if k.endswith("price") else v) for k, v in j.items()})
        ok = p == j
        same += ok
        rows.append({"q": q, "python": p, "js": j, "same": ok})
    res = {"n": len(qs), "identical": same, "rows": rows}
    (ROOT / "eval/browser/rules_parity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{same} of {len(qs)} identical")
    for r in rows:
        if not r["same"]:
            print(r)


if __name__ == "__main__":
    main()
