"""Purchase history in the store: orders plus the dataset users' training period (history.py)."""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agent import paths
from agent.agent import ShoppingAgent
from agent.history import DatasetPurchases, StorePurchases
from agent.parsing import FixedParser
from agent.schema import ShoppingRequest
from agent.tests.conftest import USER

DAY = 86_400
T0 = int(datetime(2017, 7, 14, tzinfo=UTC).timestamp())  # review times are whole days (UTC)
DATASET_USER = 3  # no store orders


def write_interactions(path, rows):
    """rows: (user_id, item_id, day, seq_idx, split), in the columns of ml/'s interactions table."""
    users, items, days, seq, split = zip(*rows, strict=True)
    table = pa.table(
        {
            "user_id": pa.array(users, pa.int64()),
            "item_id": pa.array(items, pa.int64()),
            "ts": pa.array([T0 + d * DAY for d in days], pa.int64()),
            "rating": pa.array([5.0] * len(rows)),
            "seq_idx": pa.array(seq, pa.int64()),
            "split": pa.array(split, pa.large_string()),
        }
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def dataset(tmp_path):
    path = write_interactions(
        tmp_path / "interactions.parquet",
        [
            (DATASET_USER, 9, 0, 0, "train"),  # Pentel gel pen
            (DATASET_USER, 12, 5, 1, "train"),  # sticky notes, the same day as the notebook
            (DATASET_USER, 11, 5, 2, "train"),  # notebook, later in the sequence
            (DATASET_USER, 8, 40, 3, "test"),  # the test period has not happened in the store
            (USER, 9, 0, 0, "train"),  # Pentel gel pen, long before USER's store orders
            (5, 7, 0, 0, "valid"),
        ],
    )
    return DatasetPurchases.load(path)


def test_dataset_purchases_are_the_training_period_most_recent_first(dataset):
    assert len(dataset) == 4
    got = dataset(DATASET_USER, 10)
    assert [p.item_id for p in got] == [11, 12, 9]  # same day: the later in the sequence first
    assert got[0].purchased_at.date() == datetime(2017, 7, 19, tzinfo=UTC).date()
    assert [p.item_id for p in dataset(DATASET_USER, 2)] == [11, 12]
    assert dataset(5, 10) == [] and dataset(99, 10) == [] and dataset(DATASET_USER, 0) == []


def test_store_history_puts_orders_and_the_dataset_in_one_list(catalog, dataset):
    store = StorePurchases(catalog, dataset)
    # USER ordered items 7 (2026-10-01) and 11 (2026-10-20) in the store; item 9 is from 2017.
    assert [p.item_id for p in store(USER, 10)] == [11, 7, 9]
    assert [p.item_id for p in store(USER, 2)] == [11, 7]
    assert [p.item_id for p in store(DATASET_USER, 10)] == [11, 12, 9]


def run(catalog, dataset, user_id, **request):
    parser = FixedParser(ShoppingRequest(search_text="gel pens", **request))
    agent = ShoppingAgent(catalog, parser, purchases=StorePurchases(catalog, dataset))
    return agent.run("msg", user_id)


def test_last_purchase_of_a_dataset_user_comes_from_the_training_period(catalog, dataset):
    r = run(catalog, dataset, DATASET_USER, category="gel_pens", last_purchase="same_brand")
    assert [s.item.item_id for s in r.shown] == [9]  # Pentel, as bought in 2017
    # USER's latest gel pens are the store order (Pilot, 2026), not the 2017 Pentel.
    r = run(catalog, dataset, USER, category="gel_pens", last_purchase="different_brand")
    assert r.trace[1].args["filters"]["exclude_brands"] == ["Pilot"]


def test_the_interactions_file_can_be_moved(monkeypatch, tmp_path):
    monkeypatch.setenv("RECLOOP_INTERACTIONS", str(tmp_path / "x.parquet"))
    assert paths.interactions_parquet() == tmp_path / "x.parquet"
