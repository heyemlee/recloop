"""Check the re-run against MiniOneRec's published data and the checkpoint vocabulary.

Writes data/processed/verify.json and exits non-zero if any check fails. Checks:
  * the CSVs and info file rebuilt from the re-run are byte-identical to the published ones
    (and so is the re-run's item.json);
  * sizes: 3,459 items; 48,656 samples = 38,924 / 4,866 / 4,866;
  * the rebuild of timestamps reproduced the upstream ordering and all three .inter files;
  * ASIN <-> item_id is one-to-one and covers exactly the published item ids;
  * every semantic-ID token in index.json is in the checkpoint's tokenizer vocabulary;
  * items that share a semantic ID are listed (expected: 15 pairs, 3,444 distinct SIDs).

Usage: python -m ml.data.verify
"""

from __future__ import annotations

import ast
import filecmp
import json
import re
from collections import defaultdict

import pandas as pd

from ml import paths

EXPECTED_ITEMS = 3459
EXPECTED_SAMPLES = {"train": 38924, "valid": 4866, "test": 4866}
EXPECTED_SHARED_SID_PAIRS = 15
ASIN_RE = re.compile(r"[0-9A-Z]{10}")


def _plain(value):
    """json.dumps fallback for numpy scalars."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _identical_outputs() -> dict[str, bool]:
    out = {}
    for split in paths.SPLITS:
        ours = paths.CONVERTED / split / f"{paths.SPLIT_STEM}.csv"
        out[f"{split}.csv"] = filecmp.cmp(ours, paths.public_split_csv(split), shallow=False)
    ours_info = paths.CONVERTED / "info" / f"{paths.SPLIT_STEM}.txt"
    out["info.txt"] = filecmp.cmp(ours_info, paths.PUBLIC_INFO, shallow=False)
    ours_items = paths.RERUN_DATASET_DIR / f"{paths.DATASET}.item.json"
    out["item.json"] = filecmp.cmp(ours_items, paths.PUBLIC_ITEM_JSON, shallow=False)
    return out


def read_info() -> pd.DataFrame:
    rows = [line.rstrip("\n").split("\t") for line in paths.PUBLIC_INFO.open(encoding="utf-8")]
    return pd.DataFrame(
        {
            "sid": [r[0] for r in rows],
            "title": [r[1] for r in rows],
            "item_id": [int(r[2]) for r in rows],
        }
    )


def vocab_check(index: dict[str, list[str]]) -> dict:
    added = json.loads((paths.HF_CKPT / "added_tokens.json").read_text())
    tok_cfg = json.loads((paths.HF_CKPT / "tokenizer_config.json").read_text())
    model_cfg = json.loads((paths.HF_CKPT / "config.json").read_text())
    decoder = {v["content"]: int(k) for k, v in tok_cfg["added_tokens_decoder"].items()}
    tokens = sorted({t for toks in index.values() for t in toks})
    missing_added = [t for t in tokens if t not in added]
    missing_decoder = [t for t in tokens if t not in decoder]
    id_mismatch = [t for t in tokens if t in added and t in decoder and added[t] != decoder[t]]
    max_id = max((added[t] for t in tokens if t in added), default=-1)
    per_level = defaultdict(int)
    for t in tokens:
        per_level[t[1]] += 1
    return {
        "ok": not missing_added
        and not missing_decoder
        and not id_mismatch
        and max_id < model_cfg["vocab_size"],
        "distinct_sid_tokens": len(tokens),
        "tokens_per_level": dict(sorted(per_level.items())),
        "sid_tokens_in_checkpoint": sum(1 for t in added if re.fullmatch(r"<[a-z]_\d+>", t)),
        "missing_from_added_tokens": missing_added,
        "missing_from_tokenizer_config": missing_decoder,
        "id_mismatch": id_mismatch,
        "max_token_id": max_id,
        "vocab_size": model_cfg["vocab_size"],
        "tokens_per_item": sorted({len(v) for v in index.values()}),
    }


def main() -> None:
    items = pd.read_parquet(paths.CATALOG_PARQUET)
    samples = pd.read_parquet(paths.SAMPLES_PARQUET)
    rebuild = json.loads((paths.PROCESSED / "rebuild_checks.json").read_text())
    index = json.loads(paths.PUBLIC_INDEX_JSON.read_text())
    info = read_info()
    checks: dict[str, dict] = {}

    identical = _identical_outputs()
    checks["published_files_identical"] = {"ok": all(identical.values()), **identical}

    sizes = samples.split.value_counts().to_dict()
    checks["sizes"] = {
        "ok": len(items) == EXPECTED_ITEMS and sizes == EXPECTED_SAMPLES,
        "items": len(items),
        "samples": {s: int(sizes.get(s, 0)) for s in paths.SPLITS},
        "total_samples": len(samples),
        "users": int(pd.read_parquet(paths.USERS_PARQUET).shape[0]),
        "interactions": int(pd.read_parquet(paths.INTERACTIONS_PARQUET).shape[0]),
    }
    checks["timestamp_rebuild"] = {
        "ok": all(v for k, v in rebuild.items() if k != "kept_reviews"),
        **rebuild,
    }

    ids = set(items.item_id)
    checks["asin_item_id"] = {
        "ok": items.asin.is_unique
        and items.item_id.is_unique
        and ids == set(range(EXPECTED_ITEMS)) == {int(k) for k in index} == set(info.item_id)
        and all(ASIN_RE.fullmatch(a) for a in items.asin)
        and items.asin.notna().all(),
        "distinct_asins": int(items.asin.nunique()),
        "distinct_item_ids": int(items.item_id.nunique()),
        "item_ids_equal_index_json_and_info": ids == {int(k) for k in index} == set(info.item_id),
    }
    merged = items.merge(info, on="item_id", suffixes=("", "_info"))
    checks["titles_and_sids_match_info"] = {
        "ok": len(merged) == len(items)
        and bool((merged.title == merged.title_info).all())
        and bool((merged.sid == merged.sid_info).all()),
    }
    checks["sid_tokens_in_checkpoint_vocab"] = vocab_check(index)

    groups = items.groupby("sid").item_id.apply(list)
    shared = groups[groups.map(len) > 1]
    by_id = items.set_index("item_id")
    pairs = [
        {
            "sid": sid,
            "items": [
                {"item_id": int(i), "asin": by_id.at[i, "asin"], "title": by_id.at[i, "title"]}
                for i in members
            ],
        }
        for sid, members in shared.items()
    ]
    checks["shared_sids"] = {
        "ok": len(shared) == EXPECTED_SHARED_SID_PAIRS and set(shared.map(len)) == {2},
        "distinct_sids": int(items.sid.nunique()),
        "groups": len(shared),
        "same_title_groups": sum(len({m["title"] for m in p["items"]}) == 1 for p in pairs),
        "pairs": pairs,
    }

    test = pd.read_csv(paths.public_split_csv("test"))
    hist = test.history_item_id.map(ast.literal_eval)
    repeat = [t in h for t, h in zip(test.item_id, hist, strict=True)]
    checks["test_set"] = {
        "ok": len(test) == EXPECTED_SAMPLES["test"],
        "samples": len(test),
        "users": int(test.user_id.nunique()),
        "target_in_history": int(sum(repeat)),
        "target_in_history_rate": sum(repeat) / len(repeat),
    }

    for c in checks.values():
        c["ok"] = bool(c["ok"])
    report = {"ok": all(c["ok"] for c in checks.values()), "checks": checks}
    paths.VERIFY_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_plain) + "\n"
    )

    for name, c in checks.items():
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {name}")
    print(f"sizes: {checks['sizes']['items']:,} items, samples {checks['sizes']['samples']}")
    v = checks["sid_tokens_in_checkpoint_vocab"]
    print(f"SID tokens: {v['distinct_sid_tokens']} used, all in checkpoint vocab: {v['ok']}")
    s = checks["shared_sids"]
    print(
        f"shared SIDs: {s['groups']} groups ({s['same_title_groups']} with identical titles), "
        f"{s['distinct_sids']:,} distinct SIDs"
    )
    for p in pairs:
        a, b = p["items"]
        print(
            f"  {p['sid']}  {a['item_id']:>4} {a['asin']}  {b['item_id']:>4} {b['asin']}  "
            f"{a['title'][:60]}"
        )
    t = checks["test_set"]
    print(
        f"test: {t['samples']:,} samples, {t['users']:,} users, target already in history: "
        f"{t['target_in_history']} ({t['target_in_history_rate']:.2%})"
    )
    print(f"wrote {paths.VERIFY_JSON.relative_to(paths.ROOT)}")
    if not report["ok"]:
        raise SystemExit("data verification FAILED")


if __name__ == "__main__":
    main()
