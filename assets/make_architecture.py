"""Regenerate assets/architecture.png: the RAG pipeline diagram (cellar palette, square corners).

Run:  python assets/make_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#F1ECE3"
BEIGE, BEIGE_EDGE = "#F7F3EC", "#7E7070"
WINE = "#7A2335"
PURPLE = "#24191B"
GOLD = "#6A5A1E"
INK = "#24191B"
GREY = "#5C4C4E"
CREAM_TXT = "#F1ECE3"

STAGES = [
    ("Your request", '"a bold red under $25"', BEIGE, INK, BEIGE_EDGE),
    ("Embed", "sentence-transformers\nlocal · free", WINE, CREAM_TXT, WINE),
    ("Hybrid retrieval", "dense + BM25 → RRF\ncross-encoder rerank", WINE, CREAM_TXT, WINE),
    ("Generate", "Claude · grounded\nno invented bottles", PURPLE, CREAM_TXT, PURPLE),
    ("Cited answer", "Misiones de Rengo\n91 pts · $20 · [3]", GOLD, CREAM_TXT, GOLD),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.7), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 47)
    ax.axis("off")

    ax.text(62, 43, "Retrieval-augmented generation", ha="center", va="center",
            fontsize=25, fontweight="bold", color=INK, family="serif")
    ax.text(62, 37.5, "the model answers only from reviews it retrieved, and cites every pick",
            ha="center", va="center", fontsize=12.5, color=GREY)

    w, h, gap = 20.5, 11.5, 3.5
    x0 = (124 - (5 * w + 4 * gap)) / 2
    cy = 22
    centers = []
    for i, (title, sub, fc, tc, ec) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, cy - h / 2), w, h,
                             boxstyle="square,pad=0.6",
                             linewidth=1.5, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + w / 2, cy + 2.0, title, ha="center", va="center",
                fontsize=13.2, fontweight="bold", color=tc, family="serif")
        ax.text(x + w / 2, cy - 2.7, sub, ha="center", va="center",
                fontsize=9.3, color=tc, linespacing=1.35)
        centers.append((x, x + w))

    # arrows between boxes
    for i in range(4):
        a = FancyArrowPatch((centers[i][1] + 0.4, cy), (centers[i + 1][0] - 0.4, cy),
                            arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=INK)
        ax.add_patch(a)

    # corpus cylinder feeding the retrieval stage (stage index 2)
    rx = sum(centers[2]) / 2
    cyl_top, cyl_h, cyl_w = 10.5, 7.5, 15
    from matplotlib.patches import Ellipse, Rectangle
    ax.add_patch(Rectangle((rx - cyl_w / 2, cyl_top - cyl_h), cyl_w, cyl_h,
                           facecolor=BEIGE, edgecolor=BEIGE_EDGE, linewidth=1.3))
    ax.add_patch(Ellipse((rx, cyl_top), cyl_w, 2.4, facecolor=BEIGE,
                         edgecolor=BEIGE_EDGE, linewidth=1.3))
    ax.add_patch(Ellipse((rx, cyl_top - cyl_h), cyl_w, 2.4, facecolor=BEIGE,
                         edgecolor=BEIGE_EDGE, linewidth=1.3))
    ax.text(rx, cyl_top - cyl_h / 2 + 0.3, "30,000", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(rx, cyl_top - cyl_h / 2 - 2.0, "reviews indexed", ha="center", va="center",
            fontsize=9, color=GREY)
    a = FancyArrowPatch((rx, cyl_top + 1.3), (rx, cy - h / 2 - 0.4),
                        arrowstyle="-|>", mutation_scale=15, linewidth=2.0,
                        color=GOLD, linestyle=(0, (4, 3)))
    ax.add_patch(a)

    out = Path(__file__).resolve().parent / "architecture.png"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
