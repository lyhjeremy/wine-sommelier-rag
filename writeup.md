# Building a wine sommelier with RAG

Asked for "a bold red under $25 for a steak dinner", this recommender came back with the Misiones de Rengo Carmenère (91 points, $20), "a juggernaut red that cries out for steak," and every word of that came from a Wine Enthusiast review it had retrieved. This is how it works, and what a small benchmark said about hybrid retrieval.

## Why retrieval

Put the same question to a general chatbot and it will name a producer, vintage, score and price, some of which may be made up. A wine site, on the other hand, gives you keyword filters that cannot read "bold" or "for steak."

Retrieval-augmented generation (RAG) splits the work between the two. A retriever finds relevant reviews from an actual dataset, and the model answers only from those, so the model writes the recommendation while the data supplies the facts.

## The data

The [Wine Enthusiast 130k-review dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews) has 129,971 professional tastings, each with a free-text note, an 80 to 100 score, price, grape, winery and region. `src/ingest.py` de-duplicates on title and description and, by default, indexes the 30,000 highest-scored reviews (`--all` indexes everything).

## The pipeline

![Pipeline diagram](assets/architecture.png)

1. Each review becomes a passage, `title, variety from region, country.` followed by the tasting note, with score, price and origin kept as metadata.
2. `all-MiniLM-L6-v2` through sentence-transformers turns each passage into a 384-dimensional vector on the Mac GPU (MPS). It needs no API key, and nothing leaves the machine.
3. Chroma stores the vectors with cosine similarity and persists them to disk.
4. A query goes to the dense index and to a BM25 keyword index, and Reciprocal Rank Fusion merges the two rankings. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then re-reads each shortlisted review against the query and re-orders them. Price, country and minimum-score filters are applied inside the vector query, so a $25 cap never spends a slot on an $80 bottle.
5. The top reviews go to Claude with a system prompt: recommend only from these wines, cite each with `[n]`, and say so if nothing fits. Generation runs through the Claude CLI on my subscription, or the Anthropic API if a key is set.

## The steak run, in full

The opening request ran with `--max-price 25`. The hybrid retriever and reranker returned four bottles; Claude led with the Carmenère and set aside the Soos Creek as "a touch gentler than the bold profile you're after," citing each pick back to its review.

## Does the hybrid pipeline help?

`src/eval.py` benchmarks three retrievers on twelve queries. Relevance is defined by a rule per query (grape or wine family, country, price cap, descriptor keywords), applied the same way to every system.

| System | P@10 | MRR | nDCG@10 |
|---|---|---|---|
| Dense (vector only) | 0.68 | 0.88 | 0.69 |
| Dense + BM25 (RRF) | 0.67 | 0.76 | 0.68 |
| + cross-encoder rerank | **0.82** | **0.92** | **0.83** |

On its own, fusing BM25 into the dense ranking made things slightly worse. It brought in wines that share words with the query but miss its meaning, and MRR fell from 0.88 to 0.76. The gain came from the cross-encoder, which reads the query and each candidate together (a bi-encoder compares two vectors computed separately) and lifted precision and nDCG about 21% over the dense baseline. So the extra retrieval signal only paid off once a later stage could judge relevance, which I would not have guessed without running the benchmark.

Recall@10 is about 0.03 for all three, because most rubrics match hundreds of reviews and ten results can hold only ten.

## Design decisions

Retrieval stays local and free, since embedding the corpus needs no external service. Metadata filters go inside the vector query, where they shape the candidate list from the start. And the `[n]` indices map to the exact reviews printed under each answer, so a reader can check every recommendation.

## Limitations

The dataset is one publication's snapshot, so it inherits that magazine's house style and coverage gaps. Relevance in the benchmark is defined by rules, which supports a relative comparison between retrievers; a small hand-labeled set would make the absolute numbers more trustworthy. Constraints come from CLI flags. A natural-language constraint extractor ("under twenty bucks" to `max_price=20`) is the next step, and it leads into the [Wine Pairing Agent](https://github.com/lyhjeremy/wine-pairing-agent).

Code: [github.com/lyhjeremy/wine-sommelier-rag](https://github.com/lyhjeremy/wine-sommelier-rag)
