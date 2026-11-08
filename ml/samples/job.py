"""Training samples from the event log (design §7.5): PySpark in local mode, one day at a time.

For the impressions of one UTC day D (`make samples DAY=YYYY-MM-DD`):

  1. read the exported partitions (ml.samples.export): served records of D-1 and D (a page
     served just before midnight is seen after it), impressions of D-1 and D, clicks, carts and
     purchases of D and D+1 (a click up to 30 minutes after an impression late on D);
  2. technical duplicates: one row per event_id, the first the api accepted;
  3. business duplicates: one impression per impression_id ({request_id}:{position}), the
     earliest. A click whose impression was never reported stands for it ("implied"): the web
     client reports an impression after 1 s of visibility, a quick click can come first;
  4. each impression joins its served record on (request_id, position). The features are the
     served record's, the values at serving time; nothing computed later enters a sample, so
     training sees what serving saw. An impression without a served record, or whose item is
     not the served item at that position, is dropped and counted;
  5. label: a click on the same impression within 30 minutes of it (event times are the
     UUIDv7 times of the event ids); a second click on one impression is a business
     duplicate, counted once (`clicks` keeps the count). carted / purchased: the user carted /
     bought that item within 30 minutes after the click;
  6. label delay: until the window of an impression has passed (--complete-through, by default
     the export's newest event), it is `final = false` and stays out of the training samples;
  7. negatives are downsampled: kept when a hash of the impression id falls below the rate,
     with weight 1 / rate; positives are kept with weight 1. The rate and the counts before and
     after are written next to the samples.

Outputs, partitioned by day (day=YYYY-MM-DD, Parquet):
  artifacts/samples/impressions/   every attributed impression: the A/B analysis reads these
  artifacts/samples/train/         training samples: final, home feed, served features present
  artifacts/samples/meta/<day>.json  counts after every step, the rate, the window, the inputs

Runs in its container (deploy/spark.Dockerfile): `make samples DAY=2026-11-08`, which exports the
partitions first. Inside it: python -m ml.samples.job --day 2026-11-08 [--complete-through ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path

from ml import paths
from ml.samples import export

SAMPLES = paths.ARTIFACTS / "samples"
WINDOW_MINUTES = 30
NEGATIVE_RATE = 0.1
HASH_SEED = 20261108
TRAIN_SURFACE = "home"

EXP = "struct<bucket:int,exp_id:string,variant:string>"
ITEM = (
    "struct<item_id:int,pos:int,slot:string,source:string,score:double,ranks:map<string,int>,"
    "campaign_id:bigint,price:int,pctr_ppm:bigint,ad_token:string,f:array<float>,lgbm:double>"
)
SERVED = (
    f"struct<surface:string,seed_item:int,offset:int,model_ver:string,exp:map<string,{EXP}>,"
    "config_version:int,degraded:array<string>,backfill_count:int,candidate_cache:string,"
    f"hv:bigint,items:array<{ITEM}>,features:array<string>,ranker:string,pctr_model:string>"
)
LAYERS = ("recall", "rank", "ads", "agent")


SHUFFLE_PARTITIONS = 64


def spark_session(threads: int = 2):
    from pyspark.sql import SparkSession

    if not os.environ.get("JAVA_HOME") and shutil.which("java") is None:
        raise SystemExit("no Java runtime here: run the job in its container (`make samples`)")
    # PySpark turns timestamps into naive datetimes in the process's local time zone: run the
    # driver, Python side and JVM, in UTC like the event days.
    os.environ["TZ"] = "UTC"
    time.tzset()
    return (
        SparkSession.builder.master(f"local[{threads}]")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .appName("recloop-samples")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # Many small shuffle partitions (adaptive execution merges them when the data is small):
        # with 4, a day of ~2.4 M impressions made ~260 MB partitions that overflowed the heap.
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.sql.adaptive.enabled", "true")
        # The container has 4 GB (Makefile SPARK_RUN): heap 3 GB, the rest for the JVM and Python.
        .config("spark.driver.memory", "3g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


def _uuid7_ts(col):
    """Event time = the first 48 bits of a UUIDv7 (ms since the epoch), as a timestamp."""
    from pyspark.sql import functions as F

    ms = F.conv(F.substring(F.regexp_replace(col, "-", ""), 1, 12), 16, 10).cast("long")
    return F.timestamp_millis(ms)


def read_events(spark, raw: Path, days: dict[str, list[dt.date]]):
    """One DataFrame per type over the given days; missing partitions read as empty."""
    from pyspark.sql import functions as F

    out = {}
    for typ, ds in days.items():
        parts = [str(raw / f"event_day={d.isoformat()}" / f"type={typ}") for d in ds]
        parts = [p for p in parts if Path(p).exists()]
        if parts:
            df = spark.read.schema(_arrow_schema_ddl()).parquet(*parts)
        else:
            df = spark.createDataFrame([], _arrow_schema_ddl())
        out[typ] = df.withColumn("event_ts", _uuid7_ts(F.col("event_id")))
    return out


def _arrow_schema_ddl() -> str:
    return (
        "event_id string, user_id bigint, request_id string, impression_id string, item_id int, "
        "payload string, client_ts timestamp, server_ts timestamp"
    )


def first_by(df, key: str, order: str):
    """One row per `key`, the one with the smallest `order` (then event_id)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy(key).orderBy(F.col(order).asc(), F.col("event_id").asc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def build(
    spark,
    day: dt.date,
    raw: Path = export.RAW,
    complete_through: dt.datetime | None = None,
    rate: float = NEGATIVE_RATE,
):
    """(impressions, samples, meta) for the impressions of `day`."""
    from pyspark.sql import functions as F

    one = dt.timedelta(days=1)
    ev = read_events(
        spark,
        raw,
        {
            "served": [day - one, day],
            "impression": [day - one, day],
            "click": [day, day + one],
            "add_to_cart": [day, day + one],
            "purchase": [day, day + one],
        },
    )
    meta: dict = {
        "day": day.isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "negative_rate": rate,
        "hash_seed": HASH_SEED,
    }
    read = {t: df.count() for t, df in ev.items()}
    ev = {t: first_by(df, "event_id", "server_ts").cache() for t, df in ev.items()}  # 2.
    unique = {t: df.count() for t, df in ev.items()}
    meta["events_read"] = read
    meta["technical_duplicates"] = {t: read[t] - unique[t] for t in read}
    if complete_through is None:
        newest = [df.agg(F.max("event_ts")).first()[0] for df in ev.values()]
        newest = [x for x in newest if x is not None]
        complete_through = (
            max(newest) if newest else dt.datetime.combine(day + one, dt.time(), dt.UTC)
        )
    if complete_through.tzinfo is None:  # the process runs in UTC (spark_session)
        complete_through = complete_through.replace(tzinfo=dt.UTC)
    meta["complete_through"] = complete_through.astimezone(dt.UTC).isoformat()

    # 3. impressions: reported ones, then implied ones (a click without an impression).
    imp_all = first_by(
        ev["impression"].filter(F.col("impression_id").isNotNull()), "impression_id", "event_ts"
    )
    clicks = ev["click"].filter(F.col("impression_id").isNotNull())
    first_click = first_by(clicks, "impression_id", "event_ts")
    implied = first_click.join(imp_all.select("impression_id"), "impression_id", "left_anti")
    imp = imp_all.select(
        "impression_id", "user_id", "item_id", "event_ts", F.lit(False).alias("implied")
    ).unionByName(
        implied.select(
            "impression_id", "user_id", "item_id", "event_ts", F.lit(True).alias("implied")
        )
    )
    imp = imp.filter(F.to_date("event_ts") == F.lit(day))  # this day's impressions only
    meta["impressions"] = {
        "reported": imp.filter(~F.col("implied")).count(),
        "implied": imp.filter("implied").count(),
    }
    imp = (
        imp.withColumn("request_id", F.split("impression_id", ":").getItem(0))
        .withColumn("pos", F.split("impression_id", ":").getItem(1).cast("int"))
        .withColumnRenamed("item_id", "imp_item_id")
        .withColumnRenamed("user_id", "imp_user_id")
        .withColumnRenamed("event_ts", "t_impression")
    )

    # 4. served records, one row per position.
    served = ev["served"].withColumn("p", F.from_json("payload", SERVED))
    served_items = (
        served.select(
            "request_id",
            "user_id",
            F.col("event_ts").alias("t_served"),
            "p.surface",
            "p.model_ver",
            "p.config_version",
            "p.degraded",
            "p.candidate_cache",
            "p.ranker",
            "p.pctr_model",
            F.col("p.features").alias("feature_names"),
            F.col("p.exp").alias("exp"),
            F.explode("p.items").alias("it"),
        )
        .select("*", "it.*")
        .drop("it")
        .withColumnRenamed("item_id", "served_item_id")
    )
    # A page rendered twice (a client fetching the same cursor again) logs its positions twice.
    served_items = (
        served_items.withColumn("rp", F.concat_ws(":", "request_id", F.col("pos").cast("string")))
        .withColumn("event_id", F.col("rp"))
        .transform(lambda d: first_by(d, "rp", "t_served"))
        .drop("rp", "event_id")
    )
    joined = imp.join(served_items, ["request_id", "pos"], "left")
    no_served = joined.filter(F.col("served_item_id").isNull())
    mismatch = joined.filter(
        F.col("served_item_id").isNotNull() & (F.col("served_item_id") != F.col("imp_item_id"))
    )
    meta["dropped"] = {"no_served_record": no_served.count(), "item_mismatch": mismatch.count()}
    att = joined.filter(
        F.col("served_item_id").isNotNull() & (F.col("served_item_id") == F.col("imp_item_id"))
    )

    # 5. labels.
    window = F.expr(f"INTERVAL {WINDOW_MINUTES} MINUTES")
    click_stats = clicks.groupBy("impression_id").agg(
        F.count("*").alias("clicks_all"), F.collect_list("event_ts").alias("click_times")
    )
    att = att.join(click_stats, "impression_id", "left")
    in_window = F.filter("click_times", lambda t: t <= F.col("t_impression") + window)
    att = (
        att.withColumn("in_window", F.coalesce(in_window, F.array()))
        .withColumn("clicks", F.size("in_window"))
        .withColumn("clicked", (F.col("clicks") > 0).cast("int"))
        .withColumn("t_click", F.array_min("in_window"))
        .withColumn("late_clicks", F.coalesce(F.col("clicks_all"), F.lit(0)) - F.col("clicks"))
        .drop("click_times", "in_window", "clicks_all")
    )
    for typ, name in (("add_to_cart", "carted"), ("purchase", "purchased")):
        follow = ev[typ].select(
            F.col("user_id").alias("f_user"),
            F.col("item_id").alias("f_item"),
            F.col("event_ts").alias("f_ts"),
        )
        hit = (
            att.filter("clicked = 1")
            .select("impression_id", "imp_user_id", "imp_item_id", "t_click")
            .join(
                follow,
                (F.col("f_user") == F.col("imp_user_id"))
                & (F.col("f_item") == F.col("imp_item_id"))
                & (F.col("f_ts") >= F.col("t_click"))
                & (F.col("f_ts") <= F.col("t_click") + window),
            )
            .select("impression_id")
            .distinct()
            .withColumn(name, F.lit(1))
        )
        att = att.join(hit, "impression_id", "left").withColumn(
            name, F.coalesce(F.col(name), F.lit(0))
        )

    # 6. label delay.
    att = att.withColumn(
        "final", F.col("t_impression") + window <= F.lit(complete_through.astimezone(dt.UTC))
    )
    impressions = att.select(
        F.lit(day.isoformat()).alias("day"),
        "impression_id",
        "request_id",
        "pos",
        F.col("imp_user_id").alias("user_id"),
        F.col("imp_item_id").alias("item_id"),
        "t_served",
        "t_impression",
        "implied",
        "surface",
        "slot",
        "source",
        "score",
        "ranks",
        "campaign_id",
        "price",
        "pctr_ppm",
        "model_ver",
        "config_version",
        (F.size(F.coalesce("degraded", F.array())) > 0).alias("degraded"),
        "candidate_cache",
        "ranker",
        "pctr_model",
        *[
            F.col("exp").getItem(layer).getField("variant").alias(f"exp_{layer}")
            for layer in LAYERS
        ],
        *[
            F.col("exp").getItem(layer).getField("exp_id").alias(f"exp_id_{layer}")
            for layer in LAYERS
        ],
        *[
            F.col("exp").getItem(layer).getField("bucket").alias(f"bucket_{layer}")
            for layer in LAYERS
        ],
        "feature_names",
        F.col("f").alias("features"),
        "lgbm",
        "clicked",
        "clicks",
        "late_clicks",
        "t_click",
        "carted",
        "purchased",
        "final",
    ).cache()

    # 7. training samples: final, home feed, with served features; negatives downsampled.
    u = F.pmod(F.xxhash64(F.lit(HASH_SEED), F.col("impression_id")), F.lit(1_000_000)) / F.lit(
        1_000_000.0
    )
    eligible = impressions.filter(
        F.col("final") & (F.col("surface") == TRAIN_SURFACE) & F.col("features").isNotNull()
    )
    samples = (
        eligible.filter((F.col("clicked") == 1) | (u < F.lit(rate)))
        .withColumn(
            "weight", F.when(F.col("clicked") == 1, F.lit(1.0)).otherwise(F.lit(1.0 / rate))
        )
        .select(
            "day",
            "impression_id",
            "request_id",
            "user_id",
            "item_id",
            "pos",
            "slot",
            "t_impression",
            "campaign_id",
            "pctr_ppm",
            "model_ver",
            "exp_rank",
            "feature_names",
            "features",
            F.col("clicked").alias("label"),
            "weight",
        )
    )
    n_imp = impressions.count()
    n_final = impressions.filter("final").count()
    n_elig = eligible.count()
    n_pos = eligible.filter("clicked = 1").count()
    n_samples = samples.count()
    meta["attributed"] = {
        "impressions": n_imp,
        "final": n_final,
        "censored": n_imp - n_final,
        "clicked": impressions.filter("clicked = 1").count(),
        "double_clicks": impressions.filter("clicks > 1").count(),
        "late_clicks": impressions.agg(F.sum("late_clicks")).first()[0] or 0,
        "carted": impressions.filter("carted = 1").count(),
        "purchased": impressions.filter("purchased = 1").count(),
    }
    meta["train"] = {
        "surface": TRAIN_SURFACE,
        "eligible": n_elig,
        "positives": n_pos,
        "negatives_kept": n_samples - n_pos,
        "negatives_before": n_elig - n_pos,
        "rows": n_samples,
        "without_features": impressions.filter(
            F.col("final") & (F.col("surface") == TRAIN_SURFACE) & F.col("features").isNull()
        ).count(),
    }
    return impressions, samples, meta


def write(df, out: Path, day: dt.date) -> None:
    """Replace the day=<day> partition of `out` (written aside, then moved in)."""
    target = out / f"day={day.isoformat()}"
    staging = out / f".day={day.isoformat()}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    df.drop("day").coalesce(1).write.parquet(str(staging))
    shutil.rmtree(target, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    staging.rename(target)


def run(
    day: dt.date,
    raw: Path = export.RAW,
    out: Path = SAMPLES,
    complete_through=None,
    rate: float = NEGATIVE_RATE,
    spark=None,
) -> dict:
    t0 = time.monotonic()
    own = spark is None
    spark = spark or spark_session()
    try:
        impressions, samples, meta = build(spark, day, raw, complete_through, rate)
        write(impressions, out / "impressions", day)
        write(samples, out / "train", day)
    finally:
        if own:
            spark.stop()
    meta["seconds"] = round(time.monotonic() - t0, 1)
    meta["spark"] = _spark_version()
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / f"{day.isoformat()}.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )
    return meta


def _spark_version() -> str:
    import pyspark

    return pyspark.__version__


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day", required=True, type=dt.date.fromisoformat)
    p.add_argument("--complete-through", type=dt.datetime.fromisoformat, default=None)
    p.add_argument("--negative-rate", type=float, default=NEGATIVE_RATE)
    p.add_argument("--raw", type=Path, default=export.RAW)
    p.add_argument("--out", type=Path, default=SAMPLES)
    args = p.parse_args(argv)
    meta = run(args.day, args.raw, args.out, args.complete_through, args.negative_rate)
    a, t = meta["attributed"], meta["train"]
    print(
        f"samples  {meta['day']}: {a['impressions']:,} impressions ({meta['impressions']['implied']} implied, "
        f"{a['censored']:,} still in their window), {a['clicked']:,} clicked, "
        f"{meta['dropped']['no_served_record']} without a served record; train {t['rows']:,} rows "
        f"({t['positives']:,} positives, {t['negatives_kept']:,} of {t['negatives_before']:,} negatives "
        f"at rate {meta['negative_rate']}) in {meta['seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
