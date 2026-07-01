"""Build the wine-review vector index.

Reads the Wine Enthusiast 130k-review CSV, cleans it, turns each review into a
retrievable document (title + variety + origin + tasting note), embeds it
locally, and stores vectors + metadata in a persistent Chroma collection.

Usage:
    python -m src.ingest --limit 30000
    python -m src.ingest --all
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from .config import CHROMA_DIR, COLLECTION, RAW_CSV
from .embedder import embed_documents


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["description", "title"])
    # De-duplicate identical reviews (the dataset has many).
    df = df.drop_duplicates(subset=["title", "description"])
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.reset_index(drop=True)


def _region(row: pd.Series) -> str:
    parts = [row.get("region_1"), row.get("province"), row.get("country")]
    parts = [str(p) for p in parts if isinstance(p, str) and p.strip()]
    # Deduplicate while preserving order.
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)


def _document(row: pd.Series) -> str:
    origin = _region(row)
    variety = row.get("variety") if isinstance(row.get("variety"), str) else "Wine"
    header = f"{row['title']} — {variety}"
    if origin:
        header += f" from {origin}"
    return f"{header}.\n{row['description']}"


def _metadata(row: pd.Series) -> dict:
    def s(v):
        return v if isinstance(v, str) and v.strip() else ""

    meta = {
        "title": s(row.get("title")),
        "variety": s(row.get("variety")),
        "winery": s(row.get("winery")),
        "country": s(row.get("country")),
        "province": s(row.get("province")),
        "region": s(row.get("region_1")),
        "taster": s(row.get("taster_name")),
    }
    pts = row.get("points")
    prc = row.get("price")
    if isinstance(pts, (int, float)) and not math.isnan(pts):
        meta["points"] = int(pts)
    if isinstance(prc, (int, float)) and not math.isnan(prc):
        meta["price"] = float(prc)
    return meta


def build(limit: int | None, batch: int = 512, reset: bool = True) -> int:
    import chromadb

    if not RAW_CSV.exists():
        raise SystemExit(
            f"Wine CSV not found at {RAW_CSV}.\n"
            "Download winemag-data-130k-v2.csv (Kaggle: zynicide/wine-reviews) "
            "into the data/ directory, or run scripts/fetch_data.py."
        )

    print(f"Loading {RAW_CSV.name} ...")
    df = _clean(pd.read_csv(RAW_CSV))
    if limit:
        # Prefer higher-rated reviews when sampling for a snappier demo index.
        df = df.sort_values("points", ascending=False, na_position="last").head(limit)
    df = df.reset_index(drop=True)
    print(f"Indexing {len(df):,} reviews (embed model runs locally)…")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    total = len(df)
    for start in range(0, total, batch):
        chunk = df.iloc[start : start + batch]
        docs = [_document(r) for _, r in chunk.iterrows()]
        metas = [_metadata(r) for _, r in chunk.iterrows()]
        ids = [f"wine-{start + i}" for i in range(len(chunk))]
        embeddings = embed_documents(docs)
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"  {min(start + batch, total):,}/{total:,}")

    print(f"Done. Collection '{COLLECTION}' now holds {coll.count():,} reviews.")
    return coll.count()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the wine vector index.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--limit", type=int, default=30000, help="Max reviews to index.")
    g.add_argument("--all", action="store_true", help="Index the full dataset.")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    build(limit=None if args.all else args.limit, batch=args.batch)


if __name__ == "__main__":
    main()
