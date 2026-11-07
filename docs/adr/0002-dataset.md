# ADR-0002: Dataset — Amazon 2018 Office Products with MiniOneRec's preprocessing and split

- Status: accepted
- Date: 2026-11-06
- Design: `docs/design.md` §2, §11, §19; results: `docs/results/offline.md`

## Context

RecLoop needs one dataset that serves every milestone:

- v1 compares Pop, ItemCF, SASRec, RRF and LightGBM offline, and serves them online.
- v3 adds MiniOneRec (generative retrieval over semantic IDs) as a retrieval source. Training it
  needs 4-8 A100/H100s; the only machine is an M4 Pro MacBook without CUDA. So v3 must use a
  published checkpoint, and MiniOneRec only publishes checkpoints for Amazon 2018 Industrial and
  Office, whose semantic IDs cover exactly the items of its own preprocessing.
- Comparisons must be on one fixed test set, and the paper's numbers on that test set (e.g.
  SASRec) are a check on our evaluation code.
- The online system needs real product metadata (ASIN, title, brand, price, image) and
  timestamped interactions; the published MiniOneRec files only have internal ids, titles and
  semantic IDs.

## Decision

- Use Amazon Review Data (2018), Office Products, with MiniOneRec's preprocessing
  (`data/amazon18_data_process.py`, README parameters: 5-core, 2017-10..2018-11, widened by the
  script to 2016-10) and its chronological 8:1:1 split: 3,459 items, 38,924 / 4,866 / 4,866
  samples.
- Evaluate on the published test split (MiniOneRec's GitHub repo, which matches the Hugging Face
  copy), ranking all 3,459 items, without filtering history items.
- Re-run the preprocessing from the raw UCSD files to recover ASINs and review timestamps, and
  prove the re-run is the published data: rebuild the published CSVs and info file with the
  release-time converter and require byte equality.
- Pin every input: MiniOneRec commit `0cbd94de` (release converter from `679f9702`), Hugging Face
  revision `365a03fc`, SHA-256 of every downloaded file (`ml/data/download.py`). Download the
  3.56 GB checkpoint weights only when M7 needs them (`make data-weights`).

## Alternatives considered

- **A larger category or Amazon Reviews 2023**: more items, but no published MiniOneRec
  checkpoint, so v3 would need GPU training. Scale questions are answered by load tests and a
  separate ANN benchmark (§4.6) instead.
- **Only the published preprocessed files**: no ASINs (no metadata, prices or images) and no
  timestamps.
- **MovieLens / other public benchmarks**: no product metadata for a shop, no checkpoint.

## Consequences and findings from M0

- Reproduction: with `TZ=UTC` the re-run reproduces the published train/valid/test CSVs, info
  file and item.json byte for byte. Run in the machine's zone (America/Los_Angeles) the script
  drops the 534 reviews of 2016-10-01 and ends with 3,446 items and 48,435 samples, so
  `TZ=UTC` is part of the pipeline.
- The metadata is pre-filtered to the ASINs of the 5-core reviews (lines copied verbatim, order
  and duplicates kept). The upstream script only looks those ASINs up, so its output is
  unchanged, and the 5 GB decompressed file is never written.
- All 600 semantic-ID tokens are in the Office checkpoint's tokenizer. 15 pairs of items share a
  semantic ID (3,444 distinct); every pair is one product listed twice. A generative model
  cannot tell the two apart, so results are reported under an item rule (main) and a SID rule
  (what MiniOneRec's calc.py computes, and what the cross-check uses).
- 2.45% of test targets are already in the user's history (re-purchases and same-day duplicate
  reviews); they stay rankable, as in MiniOneRec.
- Review times are day-granular, and inside a day the sequence order is the raw file's order.
  51.5% of test targets were reviewed on the same day as the last history item; ItemCF's HR@10
  is 0.27 on those and 0.04 on the rest. Results on this split partly measure "items reviewed
  together"; reports should keep that segment visible.
- Prices: 2,967 of 3,459 items (85.8%) have a single dollar price; ranges and missing or
  malformed prices are stored as NULL ("unknown").
- The data is for research use only and must be cited; it is never committed to the repository.
