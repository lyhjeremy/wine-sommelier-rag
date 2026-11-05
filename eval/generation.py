"""Run the full RAG chain on 20 frozen queries and record cost, latency and
whether each answer stays inside its retrieved context.

    python eval/generation.py

Queries: the 12 in src/eval.py plus the first 8 in eval/queries_v2.json.
Retrieval is Sommelier's default (hybrid + rerank, k 6, no filters). The prompt
and system prompt are the ones in src/sommelier.py, sent the way src/llm.py's
CLI backend sends them, with --output-format json added so the CLI reports
duration, tokens and the API-equivalent cost of the call.

Checks on each answer (no model judge):
  citations_in_range   every [n] cited is between 1 and the number of sources
  n_cited              distinct sources cited (the prompt asks for 2 or 3)
  prices_grounded      every $ amount in the answer is a price of a cited source
  points_grounded      every "NN points/pts" is the score of a cited source

Writes eval/generation_runs.jsonl (raw) and eval/generation_checks.csv.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import eval as E  # noqa: E402
from src.hybrid import HybridRetriever  # noqa: E402
from src.llm import _claude_command  # noqa: E402
from src.sommelier import PROMPT, SYSTEM, _format_context  # noqa: E402


def main():
    qs = [s["q"] for s in E.QUERIES] + [
        s["q"] for s in json.loads((ROOT / "eval" / "queries_v2.json").read_text())["queries"][:8]]
    r = HybridRetriever(use_rerank=True)
    runs_path = ROOT / "eval" / "generation_runs.jsonl"
    done = set()
    if runs_path.exists():
        done = {json.loads(l)["query"] for l in runs_path.read_text().splitlines() if l.strip()}
    with open(runs_path, "a") as fh:
        for q in qs:
            if q in done:
                continue
            t0 = time.perf_counter()
            hits = r.search(q, k=6)
            t_ret = (time.perf_counter() - t0) * 1000
            prompt = PROMPT.format(query=q, context=_format_context(hits))
            cmd = _claude_command() + ["-p", "--append-system-prompt", SYSTEM, "--output-format", "json"]
            t1 = time.perf_counter()
            res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=300)
            wall = (time.perf_counter() - t1) * 1000
            out = json.loads(res.stdout)
            rec = {"query": q, "retrieval_ms": round(t_ret, 1), "generation_wall_ms": round(wall, 1),
                   "duration_ms": out.get("duration_ms"), "duration_api_ms": out.get("duration_api_ms"),
                   "total_cost_usd": out.get("total_cost_usd"), "usage": out.get("usage"),
                   "models": list((out.get("modelUsage") or {}).keys()), "is_error": out.get("is_error"),
                   "answer": out.get("result"),
                   "sources": [{"n": i, "title": h.meta.get("title"), "price": h.meta.get("price"),
                                "points": h.meta.get("points")} for i, h in enumerate(hits, 1)],
                   "date": time.strftime("%Y-%m-%d")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(q, rec["duration_ms"], rec["total_cost_usd"])

    rows = []
    for line in runs_path.read_text().splitlines():
        rec = json.loads(line)
        ans, src = rec["answer"] or "", rec["sources"]
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans)})
        in_range = all(1 <= n <= len(src) for n in cited)
        cited_src = [s for s in src if s["n"] in cited]
        prices = {float(p.replace(",", "")) for p in re.findall(r"\$(\d[\d,]*(?:\.\d+)?)", ans)}
        ok_prices = {s["price"] for s in cited_src if s["price"]}
        pts = {int(p) for p in re.findall(r"\b(\d{2,3})\s*(?:-?\s*points?|pts)\b", ans)}
        ok_pts = {s["points"] for s in cited_src if s["points"]}
        rows.append({"query": rec["query"], "n_cited": len(cited), "citations_in_range": in_range,
                     "prices_grounded": prices <= ok_prices, "ungrounded_prices": sorted(prices - ok_prices),
                     "points_grounded": pts <= ok_pts, "ungrounded_points": sorted(pts - ok_pts),
                     "duration_ms": rec["duration_ms"], "total_cost_usd": rec["total_cost_usd"],
                     "input_tokens": (rec["usage"] or {}).get("input_tokens"),
                     "cache_read_tokens": (rec["usage"] or {}).get("cache_read_input_tokens"),
                     "cache_write_tokens": (rec["usage"] or {}).get("cache_creation_input_tokens"),
                     "output_tokens": (rec["usage"] or {}).get("output_tokens")})
    with open(ROOT / "eval" / "generation_checks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs checked")


if __name__ == "__main__":
    main()
