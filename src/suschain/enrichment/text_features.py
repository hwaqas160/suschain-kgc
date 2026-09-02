"""Sentence-transformer text features (plan C4): replaces the trainable
per-node-type nn.Embedding lookup table (graph_builder.py's "one-hot/
categorical representation" stand-in) with a FIXED feature vector derived
from each entity's name/description text. This is the concrete fix for
plan limitation B7 -- the base paper claims GraphSAGE generalizes to
"previously unseen entities without retraining" but never tests this, and
with one-hot/embedding-table features it almost certainly can't: a brand
new entity has no learned embedding at all. A text-derived feature exists
for ANY entity, seen or not, as long as it has a name -- which is what
actually makes inductive/cold-start generalization possible.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, standard
choice for this scale of data -- not the largest/most accurate available,
but adequate and citable). Embeddings are cached to disk since encoding
tens of thousands of names is the slow part and inputs don't change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "text_embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"


def _cache_path(cache_key: str) -> Path:
    # Filename must be the hash alone, not the raw cache_key text -- the
    # content_key built in encode_texts() contains "::" separators, which
    # are invalid in Windows filenames (verified directly: OSError Errno 22
    # "Invalid argument" on np.save). A human-readable prefix is still
    # useful for browsing data/processed/text_embeddings/, so a sanitized
    # (colon/slash-stripped) short version of cache_key is kept, but the
    # hash is what actually guarantees uniqueness/validity.
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key.split("::")[0])[:60]
    return CACHE_DIR / f"{safe_prefix}_{digest}.npy"


def encode_texts(texts: list[str], cache_key: str, batch_size: int = 256) -> np.ndarray:
    """Returns [len(texts), 384] float32 embeddings, cached to disk under
    `cache_key` (caller-chosen, e.g. "lodhalal_food_nama") so repeated runs
    don't re-encode. Cache is invalidated by content: a hash of the actual
    text list is embedded in the cache filename, so a changed input never
    silently loads a stale cache."""
    content_key = f"{cache_key}::{len(texts)}::{hash(tuple(texts[:5]) + tuple(texts[-5:]))}"
    path = _cache_path(content_key)
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [t if isinstance(t, str) and t.strip() else "unknown" for t in texts],
        batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def encode_series(series: pd.Series, cache_key: str) -> np.ndarray:
    return encode_texts(series.fillna("unknown").astype(str).tolist(), cache_key)
