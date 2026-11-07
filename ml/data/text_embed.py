"""Text embeddings of the catalog: items.text_emb (design §4.2, §21.2).

BAAI/bge-small-en-v1.5 (384 dimensions, MIT; pinned and checksummed by ml.data.download)
encodes "title. feature 1; feature 2; ..." of every item with sentence-transformers, normalised
so that cosine similarity is the inner product. The vectors serve the item page's "similar
items" (pgvector KNN in the api), and later search and the shopping assistant. No query
instruction prefix: item-to-item similarity is symmetric.

Runs on the CPU for reproducible output. Writes data/processed/text_emb.npy (row i = item_id
i) and text_emb.json; `make ingest` loads them into PostgreSQL.

Usage: python -m ml.data.text_embed
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from ml import paths
from ml.data import download

MAX_TOKENS = 256  # titles + features are short; the model allows 512
BATCH = 64


def item_text(title: str, features) -> str:
    feats = [f.strip() for f in (features if features is not None else []) if f and f.strip()]
    return title.strip() + (". " + "; ".join(feats) if feats else "")


def main() -> None:
    from sentence_transformers import SentenceTransformer

    for remote in download.TEXT_MODEL_FILES:
        if not remote.dest.exists():
            raise SystemExit("text model missing: run `make data` (ml.data.download)")
    items = pd.read_parquet(paths.CATALOG_PARQUET, columns=["item_id", "title", "features"])
    assert items.item_id.tolist() == list(range(len(items)))
    texts = [item_text(t, f) for t, f in zip(items.title, items.features, strict=True)]

    torch.manual_seed(0)
    t0 = time.monotonic()
    model = SentenceTransformer(str(paths.TEXT_MODEL), device="cpu")
    model.max_seq_length = MAX_TOKENS
    emb = model.encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    assert emb.shape == (len(items), 384), emb.shape

    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(paths.TEXT_EMB_NPY, emb)
    meta = {
        "model": download.TEXT_MODEL_REPO,
        "revision": download.TEXT_MODEL_REVISION,
        "dim": int(emb.shape[1]),
        "items": int(emb.shape[0]),
        "text": "title + '. ' + '; '.join(features)",
        "max_tokens": MAX_TOKENS,
        "normalized": True,
        "sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
    }
    paths.TEXT_EMB_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    norms = np.linalg.norm(emb, axis=1)
    print(
        f"text_emb {emb.shape[0]:,} x {emb.shape[1]} in {time.monotonic() - t0:.0f}s "
        f"(norms {norms.min():.4f}..{norms.max():.4f})"
    )


if __name__ == "__main__":
    main()
