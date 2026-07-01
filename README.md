# Wine Sommelier RAG

Ask for a wine in plain English — *"a bold red under $25 for steak night"*, *"a
crisp Austrian white with citrus"* — and get **2–3 real recommendations, cited to
professional reviews**, from a searchable index of ~130,000 Wine Enthusiast
tastings. A retrieval-augmented sommelier that never invents a bottle.

> 🌐 **Overview:** https://lyhjeremy.github.io/wine-sommelier-rag/

## Why
Wine search is either keyword filters (useless for *"something like a Rhône but
cheaper"*) or a chatbot that hallucinates vintages and scores. RAG fixes both:
**retrieve** the most relevant real reviews with a local semantic index, then let
**Claude** recommend *only* from what was retrieved — with a `[n]` citation on
every pick so you can trust it.

## How it works
```
your request ─▶ local embedding ─▶ Chroma vector search ─▶ rerank by
              (all-MiniLM-L6-v2)     (+ price/country/         relevance × rating
                                       variety filters)                │
                                                                       ▼
                          cited recommendation ◀── Claude ◀── top wines as context
```
- **Retrieval** is 100% local and free (`sentence-transformers` + Chroma) — no API
  key, no data leaves your machine.
- **Generation** runs on the **Claude CLI** by default (uses your Claude
  subscription, no per-token cost). Set `ANTHROPIC_API_KEY` to use the API instead.
- **Grounded:** the sommelier is instructed to recommend only from retrieved
  reviews and cite each one; if nothing matches, it says so.

## Quick start
```bash
pip install -r requirements.txt

python fetch_data.py                 # download the ~130k-review dataset -> data/
python -m src.ingest --limit 30000   # build the local vector index (or --all)

python -m src.cli ask "a bold red under $25 for a steak dinner" --max-price 25
python -m src.cli chat               # interactive sommelier
```

Filters compose with the natural-language query:
```bash
python -m src.cli ask "elegant and mineral, great with oysters" \
    --country France --variety "Chablis" --max-price 40 --min-points 90
```

## Files
| File | What it is |
|---|---|
| `fetch_data.py` | Download the Wine Enthusiast 130k-review dataset |
| `src/ingest.py` | Clean reviews → documents → local embeddings → Chroma index |
| `src/embedder.py` | Local sentence-transformers embedder (free, offline) |
| `src/retriever.py` | Vector search + metadata filters + relevance×rating rerank |
| `src/sommelier.py` | The RAG chain: retrieve → grounded, cited recommendation |
| `src/llm.py` | LLM wrapper — Claude CLI (default) or Anthropic API |
| `src/cli.py` | `ask` (one-shot) and `chat` (interactive) commands |

## Notes
Ships **code only** — the dataset and vector index are downloaded/built locally and
git-ignored. Recommendations are drawn from professional reviews for personal use.
Tested end to end: a query retrieves real bottles and returns a cited pick.

## License
[MIT](LICENSE) © 2026 Jeremy Lee
