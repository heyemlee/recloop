"""MiniOneRec's published Office splits, loaded as arrays.

The evaluation always reads the published CSVs (pinned and checksummed by ml.data.download),
so every model is scored on exactly the samples MiniOneRec reports on. `make data` proves that
the re-run reproduces these files byte for byte.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from ml import paths

NUM_ITEMS = 3459
HISTORY_LEN = 10


@dataclass(frozen=True)
class Split:
    name: str
    user: np.ndarray  # (n,) int  user index (CSV "A123" -> 123)
    target: np.ndarray  # (n,) int  item_id of the next item
    history: list[np.ndarray]  # n arrays of item_ids, oldest first, at most HISTORY_LEN long

    def __len__(self) -> int:
        return len(self.target)

    def history_matrix(self, pad: int = -1) -> np.ndarray:
        """(n, HISTORY_LEN) item ids, right-aligned (most recent last), left-padded with `pad`."""
        out = np.full((len(self), HISTORY_LEN), pad, dtype=np.int64)
        for i, h in enumerate(self.history):
            if len(h):
                out[i, -len(h) :] = h
        return out


@cache
def load_split(name: str) -> Split:
    df = pd.read_csv(
        paths.public_split_csv(name), usecols=["user_id", "history_item_id", "item_id"]
    )
    user = df.user_id.str.removeprefix("A").astype(np.int64).to_numpy()
    history = [np.asarray(ast.literal_eval(h), dtype=np.int64) for h in df.history_item_id]
    return Split(name, user, df.item_id.to_numpy(dtype=np.int64), history)


@cache
def item_sids() -> list[str]:
    """Semantic ID string of every item, indexed by item_id ("<a_1><b_2><c_3>")."""
    index = json.loads(paths.PUBLIC_INDEX_JSON.read_text())
    sids = ["".join(index[str(i)]) for i in range(NUM_ITEMS)]
    return sids


def sid_codes() -> np.ndarray:
    """item_id -> dense code of its SID; items that share a SID share a code."""
    _, codes = np.unique(np.asarray(item_sids()), return_inverse=True)
    return codes.astype(np.int64)


def train_sequences(train: Split | None = None) -> dict[int, np.ndarray]:
    """Each user's training interactions in time order.

    A user's train rows are a prefix of the user's samples (the split is a global prefix in
    time, and each user's samples are in time order). The first row's history is the user's
    first interaction; every row's target is the next one. So the training data seen by any
    model trained on the train CSV is: first history item, then the targets of all train rows.
    """
    train = train or load_split("train")
    seqs: dict[int, list[int]] = {}
    for u, h, t in zip(train.user, train.history, train.target, strict=True):
        u = int(u)
        if u not in seqs:
            if len(h) != 1:
                raise ValueError(f"user {u}: first train row has {len(h)} history items")
            seqs[u] = [int(h[0])]
        expected = seqs[u][-HISTORY_LEN:]
        if h.tolist() != expected:
            raise ValueError(f"user {u}: train row history does not extend the sequence")
        seqs[u].append(int(t))
    return {u: np.asarray(s, dtype=np.int64) for u, s in seqs.items()}
