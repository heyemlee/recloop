"""Full-ranking HR@K / NDCG@K with MiniOneRec's definitions.

Every model ranks all items for every sample; nothing is filtered (items already in the
history stay rankable: 2.45% of test targets are repeats). With 1-based rank r of the target:
    HR@K   = 1[r <= K]
    NDCG@K = 1[r <= K] / log2(r + 1)
averaged over samples. This is what MiniOneRec's calc.py computes from its beam outputs.

Two hit rules:
  * item: the target item itself must be ranked (the main metric);
  * sid:  any item with the target's semantic ID counts. This is how calc.py scores generated
          SIDs; 15 pairs of items share a SID, so the two rules differ slightly.

Ties are broken deterministically by a fixed item order passed in by the caller (training
popularity, then item_id).
"""

from __future__ import annotations

import numpy as np

KS = (3, 5, 10)
CALC_KS = (1, 3, 5, 10, 20, 50)  # what calc.py prints for 50 beams


def rank_items(scores: np.ndarray, tie_order: np.ndarray) -> np.ndarray:
    """Item ids sorted best-first for each row of `scores` (n, num_items).

    Equal scores keep the order given by `tie_order` (a permutation of item ids).
    """
    permuted = scores[:, tie_order]
    order = np.argsort(-permuted, axis=1, kind="stable")
    return tie_order[order]


def target_ranks(
    ranked: np.ndarray, target: np.ndarray, codes: np.ndarray | None = None
) -> np.ndarray:
    """1-based rank of the first hit per row; len(ranked[0]) + 1 if there is none.

    With `codes` (item_id -> SID code), a hit is any item whose code equals the target's code.
    """
    hit = ranked == target[:, None] if codes is None else codes[ranked] == codes[target][:, None]
    found = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    return np.where(found, first, ranked.shape[1] + 1)


def ranks_of(scores: np.ndarray, items: np.ndarray, tie_pos: np.ndarray) -> np.ndarray:
    """1-based rank of items[i] in row i of `scores`, without sorting.

    The order is (score desc, tie_pos asc), where tie_pos[j] is item j's position in the tie
    order; rank = 1 + number of items strictly ahead. Same result as rank_items + target_ranks.
    """
    rows = np.arange(scores.shape[0])
    s = scores[rows, items][:, None]
    ahead = (scores > s) | ((scores == s) & (tie_pos[None, :] < tie_pos[items][:, None]))
    return ahead.sum(axis=1) + 1


def sid_group_matrix(codes: np.ndarray) -> np.ndarray:
    """(num_items, max_group) items sharing each item's SID code, padded with -1."""
    groups: dict[int, list[int]] = {}
    for item, code in enumerate(codes.tolist()):
        groups.setdefault(code, []).append(item)
    width = max(len(g) for g in groups.values())
    out = np.full((len(codes), width), -1, dtype=np.int64)
    for item, code in enumerate(codes.tolist()):
        g = groups[code]
        out[item, : len(g)] = g
    return out


def sid_ranks(
    scores: np.ndarray, target: np.ndarray, tie_pos: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """1-based rank of the best-ranked item that shares the target's SID."""
    members = groups[target]  # (n, max_group)
    best = np.full(len(target), np.iinfo(np.int64).max, dtype=np.int64)
    for col in range(members.shape[1]):
        m = members[:, col]
        valid = m >= 0
        if valid.any():
            r = ranks_of(scores[valid], m[valid], tie_pos)
            best[valid] = np.minimum(best[valid], r)
    return best


def per_sample(ranks: np.ndarray, ks=KS) -> dict[str, np.ndarray]:
    """{"HR@K": 0/1 per sample, "NDCG@K": gain per sample} for each K."""
    out = {}
    for k in ks:
        hit = ranks <= k
        out[f"HR@{k}"] = hit.astype(np.float64)
        out[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    return out


def summarize(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(v.mean()) for name, v in values.items()}
