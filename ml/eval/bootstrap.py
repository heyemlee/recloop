"""User-level bootstrap confidence intervals.

A user's test samples are correlated (consecutive windows of one sequence), so samples are not
resampled independently: users are. Each replicate draws as many users as the test set has,
with replacement, and recomputes every metric as (sum over the drawn users' samples) /
(number of those samples) -- the same ratio as the point estimate. The 95% interval is the
2.5th..97.5th percentile of B replicates. One fixed seed gives every model the same replicates,
so intervals of different models, and of their paired difference, are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B = 1000
SEED = 20261106
ALPHA = 0.05


@dataclass(frozen=True)
class Resampler:
    users: np.ndarray  # (n,) user of every sample
    weights: np.ndarray  # (B, n_users) how often each user is drawn in each replicate
    inverse: np.ndarray  # (n,) sample -> position of its user in `weights` columns

    @classmethod
    def build(cls, users: np.ndarray, b: int = B, seed: int = SEED) -> Resampler:
        uniq, inverse = np.unique(users, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), size=(b, len(uniq)))
        weights = np.zeros((b, len(uniq)), dtype=np.float64)
        rows = np.repeat(np.arange(b), len(uniq))
        np.add.at(weights, (rows, draws.ravel()), 1.0)
        return cls(users, weights, inverse)

    def replicates(self, values: np.ndarray) -> np.ndarray:
        """(B,) bootstrap replicates of mean(values) for per-sample `values`."""
        n_users = self.weights.shape[1]
        per_user_sum = np.bincount(self.inverse, weights=values, minlength=n_users)
        per_user_n = np.bincount(self.inverse, minlength=n_users).astype(np.float64)
        return (self.weights @ per_user_sum) / (self.weights @ per_user_n)

    def interval(self, values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
        reps = self.replicates(values)
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)
