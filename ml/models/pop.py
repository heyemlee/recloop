"""Most-popular baseline: every user gets the items ranked by training interaction count."""

from __future__ import annotations

import numpy as np

from ml.eval.splits import Split


class Pop:
    name = "Pop"

    def __init__(self) -> None:
        self.counts: np.ndarray | None = None

    def fit(self, sequences: dict[int, np.ndarray], num_items: int) -> Pop:
        items = np.concatenate(list(sequences.values()))
        self.counts = np.bincount(items, minlength=num_items).astype(np.float64)
        return self

    def score(self, split: Split) -> np.ndarray:
        assert self.counts is not None, "fit() first"
        return np.broadcast_to(self.counts, (len(split), len(self.counts)))

    def describe(self) -> str:
        return "training interaction counts"
