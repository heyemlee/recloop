"""Query encoders for the vector half of `search_items`.

Items were embedded by `ml.data.text_embed` with BAAI/bge-small-en-v1.5 (384 dimensions,
normalised, CPU). Queries must go through the same model files and library versions (the
`encoder` dependency group pins the versions in ml/uv.lock); `BgeQueryEncoder` checks the
files against the SHA-256 pins in `ml.data.download` before loading them.

bge's model card recommends an instruction prefix for short queries against longer passages;
`QUERY_INSTRUCTION` is that prefix (the items were embedded without one, as passages).
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from agent import paths, text

DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class QueryEncoder(Protocol):
    name: str

    def encode(self, query: str) -> np.ndarray:
        """(384,) float32, L2-normalised."""
        ...


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def verify_model_files(model_dir: Path) -> None:
    """Raise unless `model_dir` holds exactly the files pinned by ml.data.download."""
    from ml import paths as ml_paths
    from ml.data import download

    for remote in download.TEXT_MODEL_FILES:
        path = model_dir / remote.dest.relative_to(ml_paths.TEXT_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `make data` (or set RECLOOP_TEXT_MODEL)")
        if path.stat().st_size != remote.size or _sha256(path) != remote.sha256:
            raise ValueError(f"{path} does not match the pin in ml/data/download.py")


class BgeQueryEncoder:
    """bge-small-en-v1.5 through sentence-transformers, on the CPU like the item embeddings."""

    def __init__(self, model_dir: Path | None = None, instruction: str | None = QUERY_INSTRUCTION):
        from sentence_transformers import SentenceTransformer

        model_dir = model_dir or paths.text_model_dir()
        verify_model_files(model_dir)
        self.model = SentenceTransformer(str(model_dir), device="cpu")
        self.model.max_seq_length = 256  # as in ml.data.text_embed
        self.instruction = instruction or ""
        self.name = "bge-small-en-v1.5" + ("+instruction" if instruction else "")

    def without_instruction(self) -> BgeQueryEncoder:
        """The same loaded model, encoding queries as they are."""
        other = copy.copy(self)
        other.instruction = ""
        other.name = "bge-small-en-v1.5"
        return other

    def encode(self, query: str) -> np.ndarray:
        v = self.model.encode(
            [self.instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return v.astype(np.float32)


class HashEncoder:
    """Deterministic stand-in for tests: hashed bag of search tokens, normalised."""

    name = "hash"

    def encode(self, query: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for t in text.tokens(query):
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
            v[h % DIM] += 1.0 if (h >> 32) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v
