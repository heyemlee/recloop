"""SASRec: self-attentive sequential recommendation (Kang & McAuley, ICDM 2018), design §4.2.

The network (`PointWiseFeedForward`, `SASRecNet`) is adapted from pmixer/SASRec.pytorch
(https://github.com/pmixer/SASRec.pytorch), python/model.py and the initialisation in
python/main.py at commit 26a8fa880f312c3700f8dcca5512d5854b257f4e. That code is licensed under
the Apache License, Version 2.0; a copy is in licenses/SASRec.pytorch-LICENSE.

Changes to the adapted code:
  * inputs are tensors instead of numpy arrays; item ids are shifted by +1 so that 0 is padding;
  * a fixed window of HISTORY_LEN (10) items, left-padded, most recent item last -- the input
    format of the evaluation (`Split.history_matrix`);
  * self-attention also masks padded keys (the original masks only the future); a query may
    always attend to itself, so rows that are all padding stay finite;
  * `batch_first` attention, and `forward` returns the user vector (features at the last
    position) instead of pos/neg logits;
  * scoring is the dot product with the shared item embeddings, as in the original `predict`,
    but trained with a full-softmax cross-entropy over all items instead of binary
    cross-entropy against one sampled negative (3,459 items make the full softmax cheap).

Everything after `SASRecNet` (training with early stopping on the validation split, the
evaluation-harness interface, checkpoints) is new code.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.eval import metrics
from ml.eval.splits import HISTORY_LEN, Split


@dataclass(frozen=True)
class SASRecConfig:
    hidden: int = 64
    blocks: int = 2
    heads: int = 1
    maxlen: int = HISTORY_LEN
    dropout: float = 0.2
    norm_first: bool = False
    # Training
    loss: str = "ce"  # "ce": full softmax over all items; "bce": upstream's BCE, 1 negative
    loss_positions: str = "all"  # "all": next item at every position; "last": final position only
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10  # epochs without a better validation NDCG@10 before stopping
    seed: int = 20261106

    def label(self) -> str:
        return (
            f"d={self.hidden}, blocks={self.blocks}, heads={self.heads}, dropout={self.dropout}, "
            f"{'pre' if self.norm_first else 'post'}-LN, {self.loss.upper()} on "
            f"{self.loss_positions} positions, "
            f"lr={self.lr}, wd={self.weight_decay}, batch={self.batch_size}"
        )


# --------------------------------------------------------------------------- adapted network


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        return outputs.transpose(-1, -2)  # Conv1d wants (N, C, L)


class SASRecNet(nn.Module):
    """History (B, L) of shifted item ids, 0 = padding, most recent last -> user vector (B, d)."""

    def __init__(self, num_items: int, cfg: SASRecConfig):
        super().__init__()
        d, length = cfg.hidden, cfg.maxlen
        self.num_items = num_items
        self.maxlen = length
        self.heads = cfg.heads
        self.norm_first = cfg.norm_first

        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(length + 1, d, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=cfg.dropout)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(d, eps=1e-8)
        for _ in range(cfg.blocks):
            self.attention_layernorms.append(nn.LayerNorm(d, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(d, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(d, cfg.dropout))

        self.register_buffer(
            "future", ~torch.tril(torch.ones(length, length, dtype=torch.bool)), persistent=False
        )
        self.register_buffer("eye", torch.eye(length, dtype=torch.bool), persistent=False)
        self.register_buffer("positions", torch.arange(1, length + 1), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # main.py: xavier_normal_ wherever it applies (matrices), then zero the padding rows.
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def attention_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """(B * heads, L, L), True where attention is not allowed: future or padded keys."""
        pad = ids == 0
        mask = (self.future.unsqueeze(0) | pad.unsqueeze(1)) & ~self.eye
        return mask.repeat_interleave(self.heads, dim=0)

    def log2feats(self, ids: torch.Tensor) -> torch.Tensor:
        seqs = self.item_emb(ids) * self.item_emb.embedding_dim**0.5
        seqs = seqs + self.pos_emb(self.positions.unsqueeze(0) * (ids != 0))
        seqs = self.emb_dropout(seqs)
        mask = self.attention_mask(ids)
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha, _ = self.attention_layers[i](x, x, x, attn_mask=mask, need_weights=False)
                seqs = seqs + mha
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha, _ = self.attention_layers[i](
                    seqs, seqs, seqs, attn_mask=mask, need_weights=False
                )
                seqs = self.attention_layernorms[i](seqs + mha)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        return self.last_layernorm(seqs)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.log2feats(ids)[:, -1, :]

    def item_vectors(self) -> torch.Tensor:
        """(num_items, d): row i is item_id i (the padding row is dropped)."""
        return self.item_emb.weight[1:]


# --------------------------------------------------------------------------- data


def encode_histories(history: np.ndarray) -> torch.Tensor:
    """(n, L) item ids left-padded with -1 (Split.history_matrix) -> shifted ids, 0 = padding."""
    return torch.from_numpy(np.asarray(history, dtype=np.int64) + 1)


def training_windows(
    sequences: dict[int, np.ndarray], maxlen: int = HISTORY_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Every (history, next item) pair of the training sequences.

    For a sequence s and every j >= 1: the history is s[max(0, j - maxlen):j], left-padded with
    -1, and the target is s[j]. These are exactly the rows of the published train CSV.
    """
    rows, targets = [], []
    for seq in sequences.values():
        for j in range(1, len(seq)):
            h = seq[max(0, j - maxlen) : j]
            row = np.full(maxlen, -1, dtype=np.int64)
            row[maxlen - len(h) :] = h
            rows.append(row)
            targets.append(seq[j])
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


# --------------------------------------------------------------------------- training


@dataclass
class TrainLog:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_valid: dict = field(default_factory=dict)
    seconds: float = 0.0
    device: str = "cpu"


def _valid_metrics(
    net: SASRecNet, valid: Split, tie_pos: np.ndarray, device: torch.device
) -> dict[str, float]:
    scores = score_histories(net, valid.history_matrix(), device)
    ranks = metrics.ranks_of(scores, valid.target, tie_pos)
    return metrics.summarize(metrics.per_sample(ranks, (5, 10)))


@torch.no_grad()
def score_histories(
    net: SASRecNet, history: np.ndarray, device: torch.device, batch: int = 2048
) -> np.ndarray:
    """(n, num_items) dot-product scores for left-padded histories (-1 = padding)."""
    net.eval()
    ids = encode_histories(history)
    items = net.item_vectors().detach()
    out = []
    for lo in range(0, len(ids), batch):
        u = net(ids[lo : lo + batch].to(device))
        out.append((u @ items.T).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train(
    cfg: SASRecConfig,
    sequences: dict[int, np.ndarray],
    num_items: int,
    valid: Split,
    tie_pos: np.ndarray,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[SASRecNet, TrainLog]:
    """Train with early stopping on the validation NDCG@10; returns the best epoch's network."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device(device)
    net = SASRecNet(num_items, cfg).to(dev)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=cfg.weight_decay
    )
    if cfg.loss not in ("ce", "bce"):
        raise ValueError(f"loss must be 'ce' or 'bce', not {cfg.loss!r}")
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    neg_gen = torch.Generator().manual_seed(cfg.seed)

    hist, target = training_windows(sequences, cfg.maxlen)
    x_all = encode_histories(hist)  # (n, L), 0 = padding
    # Next item at every position: position p predicts what follows it (the window's next
    # element, or the target at the last position). Labels are shifted ids; 0 = no label.
    y_all = torch.cat([x_all[:, 1:], torch.from_numpy(target + 1)[:, None]], dim=1)
    if cfg.loss_positions == "last":
        y_all[:, :-1] = 0
    elif cfg.loss_positions != "all":
        raise ValueError(f"loss_positions must be 'all' or 'last', not {cfg.loss_positions!r}")
    y_all[x_all == 0] = 0

    log = TrainLog(device=device)
    best_state, best_score, since_best = None, -1.0, 0
    t0 = time.monotonic()
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(len(x_all)))
        total, count = 0.0, 0
        for lo in range(0, len(order), cfg.batch_size):
            idx = order[lo : lo + cfg.batch_size]
            x, y = x_all[idx].to(dev), y_all[idx].to(dev)
            feats = net.log2feats(x)  # (B, L, d)
            keep = y > 0
            if cfg.loss == "ce":
                logits = feats[keep] @ net.item_vectors().T  # (M, num_items)
                loss = ce(logits, y[keep] - 1)
            else:  # upstream: one uniformly sampled negative per position, not the positive
                pos = y[keep]
                neg = torch.randint(1, num_items, pos.shape, generator=neg_gen).to(dev)
                neg = neg + (neg >= pos).long()  # uniform over the other num_items - 1 items
                f = feats[keep]
                pos_logit = (f * net.item_emb(pos)).sum(-1)
                neg_logit = (f * net.item_emb(neg)).sum(-1)
                loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(
                    neg_logit, torch.zeros_like(neg_logit)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * int(keep.sum())
            count += int(keep.sum())
        valid_m = _valid_metrics(net, valid, tie_pos, dev)
        log.epochs.append({"epoch": epoch, "loss": total / max(count, 1), **valid_m})
        if verbose:
            print(
                f"  epoch {epoch:3d} loss {total / max(count, 1):.4f} "
                f"valid HR@10 {valid_m['HR@10']:.4f} NDCG@10 {valid_m['NDCG@10']:.4f}",
                flush=True,
            )
        if valid_m["NDCG@10"] > best_score:
            best_score, since_best = valid_m["NDCG@10"], 0
            best_state = copy.deepcopy(net.state_dict())
            log.best_epoch, log.best_valid = epoch, valid_m
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    log.seconds = time.monotonic() - t0
    net.load_state_dict(best_state)
    net.eval()
    return net, log


# --------------------------------------------------------------------------- eval interface


class SASRec:
    """Evaluation-harness wrapper: fit / score / describe, plus checkpoint save and load."""

    name = "SASRec"

    def __init__(self, cfg: SASRecConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or SASRecConfig()
        self.device = device
        self.net: SASRecNet | None = None
        self.log: TrainLog | None = None

    def fit(
        self,
        sequences: dict[int, np.ndarray],
        num_items: int,
        valid: Split | None = None,
        tie_pos: np.ndarray | None = None,
    ) -> SASRec:
        if valid is None:
            raise ValueError("SASRec.fit needs the validation split for early stopping")
        if tie_pos is None:
            tie_pos = np.arange(num_items)
        self.net, self.log = train(self.cfg, sequences, num_items, valid, tie_pos, self.device)
        return self

    def score(self, split: Split) -> np.ndarray:
        assert self.net is not None, "fit() or load() first"
        return score_histories(self.net.cpu(), split.history_matrix(), torch.device("cpu"))

    def user_vectors(self, history: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.cpu().eval()
        with torch.no_grad():
            return self.net(encode_histories(history)).numpy()

    def item_vectors(self) -> np.ndarray:
        assert self.net is not None
        return self.net.item_vectors().detach().cpu().numpy().copy()

    def describe(self) -> str:
        epoch = f", best epoch {self.log.best_epoch}" if self.log else ""
        return self.cfg.label() + epoch

    # Checkpoints: one torch file with the config, the state dict and the training log.
    def save(self, path: Path) -> None:
        assert self.net is not None and self.log is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "num_items": self.net.num_items,
                "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
                "log": asdict(self.log),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> SASRec:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(SASRecConfig(**ckpt["config"]))
        model.net = SASRecNet(ckpt["num_items"], model.cfg)
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        model.log = TrainLog(**ckpt["log"])
        return model
