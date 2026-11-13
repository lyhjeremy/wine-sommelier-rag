"""Build the in-browser search index: 8,000 reviews, int8 embeddings.

    python eval/browser/build_index.py

Takes a fixed random sample (seed 13) of the 30,000 indexed reviews in
data/bm25.pkl (already de-duplicated on title and description by src/ingest.py),
embeds each with the same all-MiniLM-L6-v2 the Python side uses, and writes:

    assets/ai/index/emb.i8     N x 384 int8, row-major; each row quantized with its own scale
    assets/ai/index/meta.json  {"dim", "n", "scales": [...], "rows": [[title, variety, country,
                               province, price, points, color, note, flags], ...]}
                               flags: bit i set when TERM_FLAGS[i] occurs in the full review text,
                               so the browser can apply exclusions without the full text

It also writes eval/browser/python_top10.json: the Python dense top 10 (float32,
exact cosine) over the same 8,000 rows for the agreement check in
eval/browser/check_agreement.py.
"""
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.constraints import color_of  # noqa: E402
from src.embedder import embed_query  # noqa: E402

N = 8000
TERM_FLAGS = ["oak", "jam", "sweet", "butter", "tannic", "fruity"]
OUT = ROOT / "assets" / "ai" / "index"
CACHE = Path.home() / "data-cache" / "wine-sommelier-rag" / "emb.npy"
AGREE_QUERIES = [
    "a Barolo with firm tannins and notes of tar and roses",
    "a New Zealand Sauvignon Blanc with grapefruit and passion fruit",
    "a dry rosé from Provence for a summer lunch",
    "a Champagne with brioche and toasty notes",
    "an Oregon Pinot Noir with earthy forest-floor notes",
    "a bold red for a steak dinner",
    "a crisp white for oysters",
    "a sweet dessert wine with honey and apricot",
    "a peppery Syrah with smoked meat and olive",
    "an oaky buttery Chardonnay",
    "a light red to serve chilled",
    "a Malbec from Argentina with plum and violet",
]


def main():
    c = pickle.load(open(ROOT / "data" / "bm25.pkl", "rb"))
    emb = np.load(CACHE)
    idx = sorted(random.Random(13).sample(range(len(c["ids"])), N))
    E = emb[idx] / np.linalg.norm(emb[idx], axis=1, keepdims=True)
    scales = np.abs(E).max(axis=1) / 127.0
    q8 = np.round(E / scales[:, None]).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "emb.i8").write_bytes(q8.tobytes())
    rows = []
    for i in idx:
        m, d = c["metas"][i], c["docs"][i]
        note = d.split("\n", 1)[-1]
        note = note if len(note) <= 160 else note[:157].rsplit(" ", 1)[0] + "..."
        body = d.split("\n", 1)[-1].lower()
        flags = sum(1 << b for b, t in enumerate(TERM_FLAGS) if t in body)
        rows.append([m.get("title"), m.get("variety"), m.get("country"), m.get("province"),
                     m.get("price"), m.get("points"), color_of(m), note, flags])
    meta = {"dim": 384, "n": N, "model": "all-MiniLM-L6-v2", "term_flags": TERM_FLAGS,
            "source_ids": [c["ids"][i] for i in idx],
            "scales": [round(float(s), 8) for s in scales], "rows": rows}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    top = {}
    for q in AGREE_QUERIES:
        sim = E @ np.asarray(embed_query(q), dtype=np.float32)
        top[q] = [int(j) for j in np.argsort(-sim, kind="stable")[:10]]
    (ROOT / "eval" / "browser" / "python_top10.json").write_text(json.dumps(top, indent=1))
    print(N, (OUT / "emb.i8").stat().st_size, (OUT / "meta.json").stat().st_size)


if __name__ == "__main__":
    main()
