"""Shared configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this src/ directory
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
RAW_CSV = DATA_DIR / "winemag-data-130k-v2.csv"

COLLECTION = "wine_reviews"

# Small, fast, CPU/MPS-friendly embedding model (384-dim). No API key required.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def embed_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
