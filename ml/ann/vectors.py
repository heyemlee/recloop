"""Item and user vectors for the ANN benchmark.

Item vectors stand in for SASRec's 64-d item embeddings at catalog sizes the dataset does not
have (3,459 items). They keep the cluster structure of real product embeddings:

  1. MiniOneRec's text embeddings of the 3,459 Office items (Qwen, 2,560-d) are L2-normalized,
     centered and reduced to 64 dimensions by PCA, then L2-normalized again. These are the
     "real" 3,459 vectors.
  2. A Gaussian mixture with full covariances is fitted to them by EM. The number of components
     and the covariance regularization are chosen by held-out log-likelihood.
  3. Synthetic catalogs are drawn from the mixture and L2-normalized, so inner product is
     cosine similarity. Every draw carries its component, which plays the role of a category.

Without the embedding file, `synthetic_mixture` gives a clustered Gaussian instead; the report
says which source was used.

User (query) vectors are recency-weighted averages of a short history of catalog items, mostly
from one "home" component, then L2-normalized (`make_queries`).

Everything is float64 until the final float32 vectors and is driven by explicit seeds, so the
same seed gives the same vectors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

DIM = 64  # SASRec's embedding size (docs/design.md §4.2)
LOG_2PI = float(np.log(2 * np.pi))


def normalize(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm, as float32 (computed in float64)."""
    x = np.asarray(x, dtype=np.float64)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(x: np.ndarray) -> str:
    """Short SHA-256 of an array's bytes, to show that a rerun produced identical vectors."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class Pca:
    vectors: np.ndarray  # (n, dim) float64, centered projections
    explained: float  # share of the (centered) variance kept by `dim` components
    spectrum: np.ndarray  # (dim,) variance of each kept component


def pca(x: np.ndarray, dim: int = DIM) -> Pca:
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s**2 / (len(x) - 1)
    return Pca(xc @ vt[:dim].T, float(var[:dim].sum() / var.sum()), var[:dim])


def participation_ratio(spectrum: np.ndarray) -> float:
    """Effective number of dimensions of a variance spectrum: (sum v)^2 / sum v^2."""
    v = np.asarray(spectrum, dtype=np.float64)
    return float(v.sum() ** 2 / (v**2).sum())


def real_vectors(path: Path, dim: int = DIM) -> tuple[np.ndarray, Pca]:
    """MiniOneRec's item text embeddings -> L2-normalized, PCA to `dim`, normalized (float64)."""
    raw = np.load(path).astype(np.float64)
    p = pca(raw / np.linalg.norm(raw, axis=1, keepdims=True), dim)
    return p.vectors / np.linalg.norm(p.vectors, axis=1, keepdims=True), p


@dataclass(frozen=True)
class Mixture:
    weights: np.ndarray  # (k,)
    means: np.ndarray  # (k, d)
    chols: np.ndarray  # (k, d, d) lower Cholesky factors of the covariances

    @property
    def k(self) -> int:
        return len(self.weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def component_log_prob(self, x: np.ndarray) -> np.ndarray:
        """(n, k): log w_j + log N(x | mean_j, cov_j)."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty((len(x), self.k))
        for j in range(self.k):
            sol = solve_triangular(self.chols[j], (x - self.means[j]).T, lower=True)
            logdet = 2.0 * np.log(np.diag(self.chols[j])).sum()
            out[:, j] = np.log(self.weights[j]) - 0.5 * (
                self.dim * LOG_2PI + logdet + (sol**2).sum(axis=0)
            )
        return out

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        return logsumexp(self.component_log_prob(x), axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.component_log_prob(x).argmax(axis=1)

    def sample(
        self, n: int, rng: np.random.Generator, components: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """n draws (float64, not normalized) and their components.

        With `components`, only those components are drawn, in proportion to their weights.
        """
        w = self.weights.copy()
        if components is not None:
            keep = np.zeros(self.k, dtype=bool)
            keep[np.asarray(components)] = True
            w[~keep] = 0.0
        labels = rng.choice(self.k, size=n, p=w / w.sum())
        eps = rng.standard_normal((n, self.dim))
        x = np.empty((n, self.dim))
        for j in np.unique(labels):
            rows = labels == j
            x[rows] = self.means[j] + eps[rows] @ self.chols[j].T
        return x, labels


def _kmeans_labels(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding (float64, deterministic for a given rng)."""
    n = len(x)
    sq = (x**2).sum(axis=1)
    centers = [x[rng.integers(n)]]
    d2 = ((x - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        nxt = x[rng.choice(n, p=d2 / d2.sum())]
        centers.append(nxt)
        d2 = np.minimum(d2, ((x - nxt) ** 2).sum(axis=1))
    c = np.array(centers)
    labels = np.full(n, -1)
    for _ in range(iters):
        dist = sq[:, None] - 2 * x @ c.T + (c**2).sum(axis=1)[None, :]
        new = dist.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = labels == j
            # An empty cluster takes the point farthest from its current center.
            c[j] = x[members].mean(axis=0) if members.any() else x[dist.min(axis=1).argmax()]
    return labels


def _m_step(x: np.ndarray, resp: np.ndarray, reg: float) -> Mixture:
    nk = resp.sum(axis=0) + 1e-12
    means = (resp.T @ x) / nk[:, None]
    chols = np.empty((resp.shape[1], x.shape[1], x.shape[1]))
    eye = np.eye(x.shape[1])
    for j in range(resp.shape[1]):
        diff = x - means[j]
        cov = (resp[:, j, None] * diff).T @ diff / nk[j] + reg * eye
        chols[j] = np.linalg.cholesky(cov)
    return Mixture(nk / nk.sum(), means, chols)


@dataclass(frozen=True)
class Fit:
    mixture: Mixture
    log_likelihood: float  # mean per-point log-likelihood on the fitted data
    iterations: int
    trace: tuple[float, ...]  # mean log-likelihood after each EM step


def fit_mixture(
    x: np.ndarray, k: int, reg: float, seed: int, max_iter: int = 500, tol: float = 1e-7
) -> Fit:
    """EM for a full-covariance Gaussian mixture, started from k-means.

    `reg` is added to every covariance's diagonal (keeps small components well-conditioned).
    Stops when the mean log-likelihood improves by less than `tol`.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = _kmeans_labels(x, k, np.random.default_rng(seed))
    mix = _m_step(x, np.eye(k)[labels], reg)
    trace: list[float] = []
    for it in range(1, max_iter + 1):
        logp = mix.component_log_prob(x)
        lse = logsumexp(logp, axis=1)
        trace.append(float(lse.mean()))
        if it > 1 and trace[-1] - trace[-2] < tol:
            break
        mix = _m_step(x, np.exp(logp - lse[:, None]), reg)
    return Fit(mix, trace[-1], it, tuple(trace))


def select_mixture(
    x: np.ndarray, ks: tuple[int, ...], regs: tuple[float, ...], seed: int, folds: int = 5
) -> tuple[Fit, list[dict]]:
    """Choose (k, reg) by cross-validated log-likelihood, then refit on all of x."""
    fold = np.random.default_rng(seed).permutation(len(x)) % folds
    trials = []
    for k in ks:
        for reg in regs:
            held = [
                fit_mixture(x[fold != f], k, reg, seed).mixture.log_prob(x[fold == f]).mean()
                for f in range(folds)
            ]
            trials.append({"k": k, "reg": reg, "cv_ll": float(np.mean(held))})
    best = max(trials, key=lambda t: t["cv_ll"])
    return fit_mixture(x, best["k"], best["reg"], seed), trials


def synthetic_mixture(dim: int = DIM, k: int = 32, seed: int = 0) -> Mixture:
    """Fallback without the embedding file: k anisotropic Gaussian clusters around the sphere.

    Centers are uniform on the unit sphere; each cluster has a random orientation and a
    power-law variance spectrum (effective dimension ~10), scaled so that a member's cosine to
    its center is about 0.8 after normalization. Weights are Dirichlet(2).
    """
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, dim))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    spectrum = 1.0 / np.arange(1, dim + 1) ** 1.2
    spectrum *= 0.55 / spectrum.sum()  # total within-cluster variance
    chols = np.empty((k, dim, dim))
    for j in range(k):
        q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        chols[j] = np.linalg.cholesky((q * spectrum) @ q.T + 1e-9 * np.eye(dim))
    return Mixture(rng.dirichlet(np.full(k, 2.0)), means, chols)


def make_catalog(
    mixture: Mixture, n: int, seed: int, components: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """n unit-norm float32 item vectors drawn from the mixture, and their components."""
    x, labels = mixture.sample(n, np.random.default_rng(seed), components)
    return normalize(x), labels


@dataclass(frozen=True)
class QueryParams:
    max_len: int = 10  # MiniOneRec's history cap
    p_home: float = 0.7  # chance that a history item is from the user's home component
    decay: float = 0.8  # recency weight per step back (ItemCF's tuned history decay)


def make_queries(
    pool: np.ndarray,
    labels: np.ndarray,
    n: int,
    seed: int,
    params: QueryParams | None = None,
    homes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """User vectors built from histories of `pool` items; returns (queries, home components).

    Each user has a history of 1..max_len items (uniform). A home component is drawn in
    proportion to its share of the pool (only among `homes` if given); each history item is a
    random pool item of the home component with probability p_home, otherwise a random pool
    item. The user vector is the recency-weighted sum (weight decay**steps_back, the newest
    item has weight 1), normalized.
    """
    params = params or QueryParams()
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    m = params.max_len
    lengths = rng.integers(1, m + 1, size=n)
    eligible = np.arange(len(pool)) if homes is None else np.flatnonzero(np.isin(labels, homes))
    home = labels[eligible[rng.integers(0, len(eligible), size=n)]]
    in_home = order[
        starts[home][:, None] + (rng.random((n, m)) * counts[home][:, None]).astype(int)
    ]
    anywhere = rng.integers(0, len(pool), size=(n, m))
    items = np.where(rng.random((n, m)) < params.p_home, in_home, anywhere)
    slot = np.arange(m)[None, :]
    weights = np.where(slot < lengths[:, None], params.decay ** (lengths[:, None] - 1 - slot), 0.0)
    q = np.einsum("nm,nmd->nd", weights, pool[items].astype(np.float64))
    return normalize(q), home
