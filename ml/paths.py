"""Repository paths shared by the ml/ modules. Everything is relative to the repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
UCSD = RAW / "ucsd"
HF = DATA / "hf"
HF_CKPT = HF / "Office_ckpt"
TEXT_MODEL = HF / "bge-small-en-v1.5"  # sentence encoder for items.text_emb (M1)
PROCESSED = DATA / "processed"

THIRD_PARTY = ROOT / "third_party"
MINIONEREC = THIRD_PARTY / "MiniOneRec"
# convert_dataset.py as of the data release (git show <release commit>:convert_dataset.py)
MINIONEREC_RELEASE = THIRD_PARTY / "minionerec_release"

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "docs" / "results"

DATASET = "Office_Products"
# File stem MiniOneRec uses for the published splits and info file.
SPLIT_STEM = f"{DATASET}_5_2016-10-2018-11"

# Published MiniOneRec data (inside the pinned clone).
PUBLIC = MINIONEREC / "data" / "Amazon"
PUBLIC_INDEX_JSON = PUBLIC / "index" / f"{DATASET}.index.json"
PUBLIC_ITEM_JSON = PUBLIC / "index" / f"{DATASET}.item.json"
PUBLIC_INFO = PUBLIC / "info" / f"{SPLIT_STEM}.txt"
SPLITS = ("train", "valid", "test")


def public_split_csv(split: str) -> Path:
    return PUBLIC / split / f"{SPLIT_STEM}.csv"


# Outputs of `make data` (all under data/processed/).
INTERIM = PROCESSED / "interim"  # decompressed reviews, filtered metadata
RERUN = PROCESSED / "rerun"  # amazon18_data_process.py output (+ published index.json)
RERUN_DATASET_DIR = RERUN / DATASET
CONVERTED = PROCESSED / "converted"  # release convert_dataset.py run on the re-run
CATALOG_PARQUET = PROCESSED / "items.parquet"
USERS_PARQUET = PROCESSED / "users.parquet"
INTERACTIONS_PARQUET = PROCESSED / "interactions.parquet"
SAMPLES_PARQUET = PROCESSED / "samples.parquet"
TEXT_EMB_NPY = PROCESSED / "text_emb.npy"  # (3459, 384) float32, row i = item_id i
TEXT_EMB_JSON = PROCESSED / "text_emb.json"  # model, revision, text recipe, checksum
VERIFY_JSON = PROCESSED / "verify.json"
