# Building a wine sommelier with RAG

How I turned the Wine Enthusiast review dataset into a recommender that names only real bottles, and what a small benchmark said about hybrid retrieval.

## The problem

Ask a chatbot for "a bold red under $25 for steak" and it will name a producer, vintage, score and price, and some of those may be made up. Ask a wine site and you get keyword filters that cannot read "bold" or "for steak."

Retrieval-augmented generation (RAG) splits the work. A retriever finds real, relevant reviews, and the model answers only from those. The model writes the recommendation; the data supplies the facts.

## The data

The [Wine Enthusiast 130k-review dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews) has 129,971 professional tastings, each with a free-text note, an 80 to 100 score, price, grape, winery and region. `src/ingest.py` de-duplicates on title and description and, by default, indexes the 30,000 highest-scored reviews (`--all` indexes everything).

## The pipeline

![Pipeline diagram](assets/architecture.png)

1. **Documents.** Each review becomes a passage, `title, variety from region, country.` followed by the tasting note, with score, price and origin kept as metadata.
2. **Embeddings.** `all-MiniLM-L6-v2` through sentence-transformers turns each passage into a 384-dimensional vector on the Mac GPU (MPS). No API key is needed and nothing leaves the machine.
3. **Vector store.** Chroma with cosine similarity, persisted to disk.
4. **Hybrid retrieval and rerank.** A query goes to the dense index and to a BM25 keyword index, and Reciprocal Rank Fusion merges the two rankings. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then re-reads each shortlisted review against the query and re-orders them. Price, country and minimum-score filters are applied inside the vector query, so a $25 cap never spends a slot on an $80 bottle.
5. **Grounded generation.** The top reviews go to Claude with a system prompt: recommend only from these wines, cite each with `[n]`, and say so if nothing fits. Generation runs through the Claude CLI on my subscription, or the Anthropic API if a key is set.

## A sample run

Asked for "a bold red under $25 for a steak dinner" with `--max-price 25`, the hybrid retriever and reranker returned four bottles. Claude led with the **Misiones de Rengo Carmenère (91 points, $20)**, "a juggernaut red that cries out for steak," and set aside the Soos Creek as "a touch gentler than the bold profile you're after." Each pick is cited back to its review.

## Does the hybrid pipeline help?

`src/eval.py` benchmarks three retrievers on twelve queries. Relevance is defined by a rule per query (grape or wine family, country, price cap, descriptor keywords), applied the same way to every system.

| System | P@10 | MRR | nDCG@10 |
|---|---|---|---|
| Dense (vector only) | 0.68 | 0.88 | 0.69 |
| Dense + BM25 (RRF) | 0.67 | 0.76 | 0.68 |
| + cross-encoder rerank | **0.82** | **0.92** | **0.83** |

Fusing BM25 into the dense ranking did not help by itself. It brought in wines that share words with the query but miss its meaning, and MRR fell from 0.88 to 0.76. The gain came from the cross-encoder, which reads the query and each candidate together instead of comparing two separately computed vectors: about 21% higher precision and nDCG over the dense baseline. Adding a retrieval signal did not improve retrieval until a stage could judge relevance, and only the measurement showed that.

Recall@10 is about 0.03 for all three, because most rubrics match hundreds of reviews and ten results can hold only ten.

## Design decisions

- Retrieval stays local and free. Embedding the corpus needs no external service.
- Metadata filters go into the vector query instead of being applied afterwards.
- The `[n]` indices map to the exact reviews printed under each answer, so a reader can check every recommendation.

## Limitations and next steps

- The dataset is one publication's snapshot and inherits its house style and coverage gaps.
- Relevance in the benchmark is rule-defined. It supports a relative comparison; a small hand-labelled set would make the absolute numbers more trustworthy.
- Constraints come from CLI flags. A natural-language constraint extractor ("under twenty bucks" to `max_price=20`) is the next step, and it leads into the [Wine Pairing Agent](https://github.com/lyhjeremy/wine-pairing-agent).

Code: [github.com/lyhjeremy/wine-sommelier-rag](https://github.com/lyhjeremy/wine-sommelier-rag)
