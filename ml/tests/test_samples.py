"""The sample job (ml.samples.job) on a hand-made event log.

Runs in the job's container (`make test-samples`: PySpark and a JVM); in the ml environment,
which has neither, it is skipped.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.samples import export

pyspark = pytest.importorskip("pyspark", reason="PySpark runs in the job's container: make test-samples")

from ml.samples import job  # noqa: E402

DAY = dt.date(2026, 11, 8)
T0 = dt.datetime(2026, 11, 8, 10, 0, tzinfo=dt.UTC)
FEATURES = ["sasrec_score", "sasrec_rank"]


def uuid7(at: dt.datetime, n: int) -> str:
    ms = int(at.timestamp() * 1000)
    raw = ms.to_bytes(6, "big") + bytes([0x70, n % 256]) + bytes([0x80, n // 256 % 256]) + n.to_bytes(6, "big")
    return str(uuid.UUID(bytes=raw))


class Log:
    def __init__(self):
        self.rows: dict[tuple[dt.date, str], list[dict]] = {}
        self.n = 0

    def add(self, typ: str, at: dt.datetime, user: int, *, event_id=None, request_id=None, impression_id=None, item=None, payload=None, server_ts=None):
        self.n += 1
        eid = event_id or uuid7(at, self.n)
        row = {
            "event_id": eid,
            "user_id": user,
            "request_id": request_id,
            "impression_id": impression_id,
            "item_id": item,
            "payload": json.dumps(payload) if payload is not None else None,
            "client_ts": at,
            "server_ts": server_ts or at + dt.timedelta(milliseconds=5),
        }
        self.rows.setdefault((at.date(), typ), []).append(row)
        return eid

    def served(self, at, user, items, offset=0, variant="control", ranker="rrf"):
        rid = uuid7(at, 10_000 + self.n)
        payload = {
            "surface": "home",
            "offset": offset,
            "model_ver": "v1",
            "exp": {"recall": {"bucket": 3, "exp_id": "r", "variant": "control"}, "rank": {"bucket": 7, "exp_id": "x", "variant": variant}},
            "config_version": 2,
            "degraded": [],
            "hv": 1,
            "ranker": ranker,
            "features": FEATURES,
            "items": [
                {"item_id": it, "pos": offset + i + 1, "slot": "organic", "source": "sasrec", "score": 0.1, "ranks": {"sasrec": i + 1}, "f": [1.0 + i, float(i + 1)]}
                for i, it in enumerate(items)
            ],
        }
        self.add("served", at, user, request_id=rid, payload=payload)
        return rid

    def write(self, root):
        for (day, typ), rows in self.rows.items():
            d = root / f"event_day={day.isoformat()}" / f"type={typ}"
            d.mkdir(parents=True, exist_ok=True)
            cols = {f.name: [r[f.name] for r in rows] for f in export.SCHEMA}
            pq.write_table(pa.table(cols, schema=export.SCHEMA), d / "part-0.parquet")


@pytest.fixture(scope="module")
def spark():
    s = job.spark_session(threads=1)
    yield s
    s.stop()


def imp_id(rid, pos):
    return f"{rid}:{pos}"


def build_log():
    log = Log()
    m = dt.timedelta(minutes=1)
    # A page served late on the day before, seen after midnight.
    early = log.served(dt.datetime(2026, 11, 7, 23, 59, 50, tzinfo=dt.UTC), 1, [10, 11])
    log.add("impression", dt.datetime(2026, 11, 8, 0, 0, 5, tzinfo=dt.UTC), 1, impression_id=imp_id(early, 1), item=10)
    # User 2: four impressions; a click (sent twice: same event_id), a double click, a late click.
    rid = log.served(T0, 2, [20, 21, 22, 23])
    for pos, it in enumerate([20, 21, 22, 23], start=1):
        log.add("impression", T0 + dt.timedelta(seconds=pos), 2, impression_id=imp_id(rid, pos), item=it)
    c1 = log.add("click", T0 + 2 * m, 2, impression_id=imp_id(rid, 1), item=20)
    log.add("click", T0 + 2 * m, 2, event_id=c1, impression_id=imp_id(rid, 1), item=20, server_ts=T0 + 3 * m)
    log.add("click", T0 + 3 * m, 2, impression_id=imp_id(rid, 1), item=20)  # double click
    log.add("click", T0 + 45 * m, 2, impression_id=imp_id(rid, 2), item=21)  # outside 30 minutes
    log.add("add_to_cart", T0 + 4 * m, 2, item=20)
    log.add("purchase", T0 + 5 * m, 2, item=20)
    log.add("purchase", T0 + 50 * m, 2, item=21)
    # A click whose impression was never reported (implied), and an impression with no served
    # record, and one whose item is not the served item.
    rid3 = log.served(T0 + 10 * m, 3, [30, 31])
    log.add("click", T0 + 10 * m + dt.timedelta(seconds=1), 3, impression_id=imp_id(rid3, 2), item=31)
    log.add("impression", T0 + 11 * m, 3, impression_id=imp_id(str(uuid.uuid4()), 1), item=99)
    log.add("impression", T0 + 11 * m, 3, impression_id=imp_id(rid3, 1), item=77)
    # Late on the day: an impression clicked after midnight; one still inside its window.
    late = log.served(dt.datetime(2026, 11, 8, 23, 50, tzinfo=dt.UTC), 4, [40, 41])
    log.add("impression", dt.datetime(2026, 11, 8, 23, 50, 1, tzinfo=dt.UTC), 4, impression_id=imp_id(late, 1), item=40)
    log.add("click", dt.datetime(2026, 11, 9, 0, 5, tzinfo=dt.UTC), 4, impression_id=imp_id(late, 1), item=40)
    log.add("impression", dt.datetime(2026, 11, 8, 23, 50, 1, tzinfo=dt.UTC), 4, impression_id=imp_id(late, 2), item=41)
    # Many unclicked impressions for the downsampling.
    for u in range(100, 140):
        r = log.served(T0 + 20 * m, u, [50, 51, 52, 53, 54])
        for pos, it in enumerate([50, 51, 52, 53, 54], start=1):
            log.add("impression", T0 + 20 * m + dt.timedelta(seconds=pos), u, impression_id=imp_id(r, pos), item=it)
    return log, {"early": early, "rid": rid, "rid3": rid3, "late": late}


def test_job_attributes_dedups_and_labels(spark, tmp_path):
    log, ids = build_log()
    log.write(tmp_path / "raw")
    through = dt.datetime(2026, 11, 9, 0, 10, tzinfo=dt.UTC)  # the 23:50 impressions' windows are still open
    meta = job.run(DAY, tmp_path / "raw", tmp_path / "out", complete_through=through, rate=0.25, spark=spark)
    imps = pq.read_table(tmp_path / "out" / "impressions" / "day=2026-11-08").to_pandas().set_index("impression_id")
    train = pq.read_table(tmp_path / "out" / "train" / "day=2026-11-08").to_pandas()

    assert meta["technical_duplicates"]["click"] == 1  # the click sent twice
    assert meta["dropped"] == {"no_served_record": 1, "item_mismatch": 1}
    early = imps.loc[imp_id(ids["early"], 1)]
    assert early.item_id == 10 and early.exp_rank == "control" and not early.clicked

    first = imps.loc[imp_id(ids["rid"], 1)]
    assert first.clicked == 1 and first.clicks == 2 and first.carted == 1 and first.purchased == 1
    second = imps.loc[imp_id(ids["rid"], 2)]
    assert second.clicked == 0 and second.late_clicks == 1 and second.purchased == 0
    assert list(first.features) == pytest.approx([1.0, 1.0]) and list(first.feature_names) == FEATURES

    implied = imps.loc[imp_id(ids["rid3"], 2)]
    assert implied.implied and implied.clicked == 1
    assert meta["impressions"]["implied"] == 1

    late = imps.loc[imp_id(ids["late"], 1)]
    assert late.clicked == 1 and not late.final  # its click came on the next day; window open
    assert imp_id(ids["late"], 1) not in set(train.impression_id)

    # Downsampling: every positive kept with weight 1, negatives at 1 / rate.
    assert set(train[train.label == 1].impression_id) == set(imps[(imps.clicked == 1) & imps.final].index)
    neg = train[train.label == 0]
    assert (neg.weight == 4.0).all() and (train[train.label == 1].weight == 1.0).all()
    eligible_neg = int(((imps.clicked == 0) & imps.final).sum())
    assert meta["train"]["negatives_before"] == eligible_neg
    assert 0 < len(neg) < eligible_neg
    # Deterministic: the same rows again.
    meta2 = job.run(DAY, tmp_path / "raw", tmp_path / "out2", complete_through=through, rate=0.25, spark=spark)
    train2 = pq.read_table(tmp_path / "out2" / "train" / "day=2026-11-08").to_pandas()
    assert sorted(train2.impression_id) == sorted(train.impression_id)
    assert meta2["train"] == meta["train"]


def test_complete_through_defaults_to_the_newest_event(spark, tmp_path):
    log, ids = build_log()
    log.write(tmp_path / "raw")
    meta = job.run(DAY, tmp_path / "raw", tmp_path / "out", rate=1.0, spark=spark)
    assert meta["complete_through"].startswith("2026-11-09T00:05")
    imps = pq.read_table(tmp_path / "out" / "impressions" / "day=2026-11-08").to_pandas().set_index("impression_id")
    assert not imps.loc[imp_id(ids["late"], 2)].final
    assert imps.loc[imp_id(ids["rid"], 1)].final
