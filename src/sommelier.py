"""The RAG chain: retrieve wines, then have Claude recommend with citations."""

from __future__ import annotations

from dataclasses import dataclass

from .hybrid import HybridRetriever
from .llm import LLM
from .retriever import Retriever, WineHit

SYSTEM = (
    "You are a knowledgeable, WSET-trained sommelier. You recommend wines ONLY "
    "from the retrieved reviews provided to you — never invent bottles, scores, "
    "or prices. Cite each wine you recommend by its [n] index. Be warm but "
    "concise, explain *why* each pick fits the request (style, flavour, value), "
    "and if the retrieved wines don't truly match, say so honestly."
)

PROMPT = """A guest asked:
"{query}"

Here are the most relevant wines retrieved from a database of professional
reviews. Use ONLY these.

{context}

Recommend the best 2–3 wines for this guest. For each: give the name, its score
and price if known, and one or two sentences on why it fits — citing it as [n].
End with a one-line tasting-note summary of your top pick."""


@dataclass
class Recommendation:
    answer: str
    sources: list[WineHit]


def _format_context(hits: list[WineHit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        tags = []
        if h.points:
            tags.append(f"{h.points} pts")
        if h.price:
            tags.append(f"${h.price:g}")
        tag = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"[{i}]{tag} {h.doc}")
    return "\n\n".join(lines)


class Sommelier:
    def __init__(self, hybrid: bool = True, rerank: bool = True):
        # Hybrid (dense + BM25 + cross-encoder) is the default; fall back to the
        # plain dense retriever with ``hybrid=False`` for a lighter dependency
        # footprint or an apples-to-apples baseline.
        self.retriever = HybridRetriever(use_rerank=rerank) if hybrid else Retriever()
        self.llm = LLM()

    def recommend(self, query: str, k: int = 6, **filters) -> Recommendation:
        hits = self.retriever.search(query, k=k, **filters)
        if not hits:
            return Recommendation(
                "I couldn't find any wines matching those constraints. "
                "Try relaxing the price or region filter.",
                [],
            )
        prompt = PROMPT.format(query=query, context=_format_context(hits))
        answer = self.llm.complete(prompt, system=SYSTEM)
        return Recommendation(answer, hits)
