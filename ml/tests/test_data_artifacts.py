"""Invariants of `make data` outputs. Skipped (with a reason) until `make data` has run."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from ml import paths
from ml.data.download import MINIONEREC_FILES, sha256_file
from ml.eval.splits import NUM_ITEMS, item_sids, load_split, sid_codes, train_sequences

pytestmark = pytest.mark.skipif(
    not paths.VERIFY_JSON.exists(), reason="data/processed is empty: run `make data` first"
)


def test_verification_passed():
    report = json.loads(paths.VERIFY_JSON.read_text())
    failed = [name for name, c in report["checks"].items() if not c["ok"]]
    assert report["ok"] and not failed, failed


def test_published_files_are_the_pinned_ones():
    for rel, digest in MINIONEREC_FILES.items():
        assert sha256_file(paths.MINIONEREC / rel) == digest, rel


def test_catalog():
    items = pd.read_parquet(paths.CATALOG_PARQUET)
    assert len(items) == NUM_ITEMS
    assert items.item_id.tolist() == list(range(NUM_ITEMS))
    assert items.asin.is_unique
    assert items.asin.map(lambda a: bool(re.fullmatch(r"[0-9A-Z]{10}", a))).all()
    assert items.title.str.len().gt(1).all()
    assert items.sid.tolist() == item_sids()
    priced = items.price_cents.dropna()
    assert (priced > 0).all()
    # every parsed price comes from a single "$x.yy" string; ranges stay unknown
    ranges = items.price_raw.fillna("").str.contains(" - ")
    assert items.loc[ranges, "price_cents"].isna().all()


def test_splits_and_samples_agree():
    samples = pd.read_parquet(paths.SAMPLES_PARQUET)
    for name in paths.SPLITS:
        split = load_split(name)
        s = samples[samples.split == name].sort_values("row")
        assert len(s) == len(split)
        np.testing.assert_array_equal(s.user_id.to_numpy(), split.user)
        np.testing.assert_array_equal(s.item_id.to_numpy(), split.target)
    # chronological 8:1:1 split by target time
    by_split = samples.groupby("split").ts.agg(["min", "max"])
    assert by_split.loc["train", "max"] <= by_split.loc["valid", "min"]
    assert by_split.loc["valid", "max"] <= by_split.loc["test", "min"]


def test_train_sequences_are_the_train_interactions():
    inter = pd.read_parquet(paths.INTERACTIONS_PARQUET)
    seqs = train_sequences()
    train = inter[inter.split == "train"].sort_values(["user_id", "seq_idx"])
    assert sum(len(s) for s in seqs.values()) == len(train)
    for uid, g in train.groupby("user_id"):
        np.testing.assert_array_equal(seqs[uid], g.item_id.to_numpy())


def test_shared_sids():
    codes = sid_codes()
    assert len(np.unique(codes)) == NUM_ITEMS - 15
