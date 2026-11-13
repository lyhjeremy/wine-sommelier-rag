"""Do the browser's top results agree with the Python dense retriever?

    python eval/browser/check_agreement.py [--port 8821]

Serves the repo locally, loads assets/ai/wine-search.js in headless Chromium
(Playwright), embeds each query in eval/browser/python_top10.json with
transformers.js (q8 MiniLM, WASM) and ranks the int8 index with filters off.
Compares each browser top 10 with the Python top 10 (sentence-transformers,
float32, exact cosine over the same 8,000 rows). Also records every request the
page makes, to confirm none leaves the local server.
Writes eval/browser/agreement.json.
"""
import argparse
import functools
import http.server
import json
import statistics
import threading
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8821)
    a = ap.parse_args()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    py = json.loads((ROOT / "eval/browser/python_top10.json").read_text())
    hosts = set()
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.on("request", lambda r: hosts.add(urlparse(r.url).scheme + "://" + urlparse(r.url).netloc))
        page.goto(f"http://127.0.0.1:{a.port}/index.html")
        res = page.evaluate("""async (qs) => {
            const m = await import('/assets/ai/wine-search.js');
            const st = await m.load('/assets/ai/');
            const out = {loadMs: st.loadMs, rows: []};
            for (const q of qs) {
              const r = await m.search(q, {applyFilters: false, k: 10});
              out.rows.push({q, top: r.top.map(t => t.i), embedMs: r.embedMs, rankMs: r.rankMs});
            }
            return out;
        }""", list(py))
        b.close()
    srv.shutdown()
    rows = []
    for r in res["rows"]:
        a10, b10 = py[r["q"]], r["top"]
        rows.append({"q": r["q"], "overlap10": len(set(a10) & set(b10)),
                     "same_top1": a10[0] == b10[0], "python": a10, "browser": b10,
                     "embed_ms": round(r["embedMs"], 1), "rank_ms": round(r["rankMs"], 1)})
    out = {"n_queries": len(rows),
           "mean_overlap_at_10": round(statistics.mean(x["overlap10"] for x in rows), 2),
           "min_overlap_at_10": min(x["overlap10"] for x in rows),
           "same_top1": sum(x["same_top1"] for x in rows),
           "load_ms_cold": round(res["loadMs"]),
           "embed_ms_median": round(statistics.median(x["embed_ms"] for x in rows), 1),
           "rank_ms_median": round(statistics.median(x["rank_ms"] for x in rows), 1),
           "hosts_requested": sorted(hosts),
           "machine": "Apple M4, headless Chromium (Playwright), WASM backend",
           "rows": rows}
    (ROOT / "eval/browser/agreement.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
