"""Write eval/constraints/labeling_sheet.csv: the Sonnet-filtered top ten for every
constrained request, with blank columns for a person's relevance verdict.
No human labels exist yet.

    python eval/constraints/make_labeling_sheet.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    qs = {q["id"]: q for q in json.loads((HERE / "queries.json").read_text())["queries"]}
    top = json.loads((HERE / "top10.json").read_text())
    with open(HERE / "labeling_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "request", "rank", "title", "variety", "country", "province", "price",
                    "breaks_answer_key", "human_relevant (y/n)", "human_note"])
        for qid, q in qs.items():
            for r, x in enumerate(top.get(f"{qid}|llm-sonnet", []), 1):
                w.writerow([qid, q["q"], r, x["title"], x["variety"], x["country"], x["province"],
                            x["price"], "; ".join(x["violates"]), "", ""])
    print("wrote", HERE / "labeling_sheet.csv")


if __name__ == "__main__":
    main()
