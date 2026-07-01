"""Command-line interface for the Wine Sommelier RAG.

    python -m src.cli ask "a bold red under $25 for steak night" --max-price 25
    python -m src.cli chat
"""

from __future__ import annotations

import argparse

from .sommelier import Sommelier


def _print_reco(reco) -> None:
    print("\n" + reco.answer.strip() + "\n")
    if reco.sources:
        print("— Retrieved wines —")
        for i, h in enumerate(reco.sources, 1):
            print(f"  [{i}] {h.citation()}")
    print()


def _filters(args) -> dict:
    return {
        "max_price": args.max_price,
        "min_price": args.min_price,
        "country": args.country,
        "variety": args.variety,
        "min_points": args.min_points,
    }


def cmd_ask(args) -> None:
    som = Sommelier()
    _print_reco(som.recommend(args.query, k=args.k, **_filters(args)))


def cmd_chat(args) -> None:
    som = Sommelier()
    print("🍷 Wine Sommelier — ask for a recommendation (Ctrl-C to quit).")
    try:
        while True:
            q = input("\nyou › ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            _print_reco(som.recommend(q, k=args.k, **_filters(args)))
    except (KeyboardInterrupt, EOFError):
        print("\nCheers! 🥂")


def _add_filters(p) -> None:
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--country", type=str, default=None)
    p.add_argument("--variety", type=str, default=None)
    p.add_argument("--min-points", type=int, default=None)
    p.add_argument("-k", type=int, default=6, help="Wines to retrieve.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Wine Sommelier RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="One-shot recommendation.")
    a.add_argument("query")
    _add_filters(a)
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat", help="Interactive session.")
    _add_filters(c)
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
