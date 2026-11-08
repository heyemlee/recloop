"""Fetch and verify every external input of the M0 data pipeline.

Sources (all pinned):
  * MiniOneRec GitHub repo, full history, checked out at MINIONEREC_COMMIT. The converter used for
    the data release is extracted from RELEASE_COMMIT with `git show` (main has a later edit).
  * Amazon Review Data (2018), Office Products: 5-core reviews and product metadata from UCSD.
  * MiniOneRec Office checkpoint on Hugging Face (Apache-2.0): tokenizer/config files only by
    default. Generation (M7) also needs model.safetensors (3.56 GB) and the rest of the
    tokenizer (vocab, merges, tokenizer.json, generation_config); --with-weights fetches them.
  * BAAI/bge-small-en-v1.5 (MIT) on Hugging Face: the sentence encoder of items.text_emb (M1).

Every file is checked against a SHA-256 recorded on first download. Re-running is cheap: files
that already verify are skipped.

Usage: python -m ml.data.download [--with-weights]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ml import paths

MINIONEREC_URL = "https://github.com/AkaliKong/MiniOneRec.git"
MINIONEREC_COMMIT = "0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1"  # main, 2026-11-06
RELEASE_COMMIT = "679f9702e20b1ee0cc6a99e15f0e694b4761b234"  # 2025-10-24, data release

# Files of the pinned clone the pipeline depends on.
MINIONEREC_FILES = {
    "data/amazon18_data_process.py": "b0125b4985ce7a2ea65a3125250277551e52c8c6a40396ae0ca443cde5bbd960",
    "calc.py": "3e5dc6bf10c2e23f50180f520993462c7756b717dd60d646d8b540274a3fce82",
    "data/Amazon/index/Office_Products.index.json": "f599d2b858beb5a2f660d6253896746c25a65f3a7d0c2f3ae6fd4ec2091eebc8",
    "data/Amazon/index/Office_Products.item.json": "398cb2bb7868bedf2ed8ed6b24d25e172f21b7022554d1f8ea52200478494c1c",
    "data/Amazon/info/Office_Products_5_2016-10-2018-11.txt": "7e8bae0a051a377713364b9e669590f86feb869fc269157c64b2189c0f4bcfa4",
    "data/Amazon/train/Office_Products_5_2016-10-2018-11.csv": "7d02ec09b3d04b6ce1c1a0c68df5dd29810c1887655d90f60aad166adc48934e",
    "data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv": "a7d8f049ae425f413def22687cebcd172c65833633ff76dd3d90a83bca98234f",
    "data/Amazon/test/Office_Products_5_2016-10-2018-11.csv": "c1967034e197a7837a8962f2556efaabc12eb5bb84e1fdc79cd0accbce71add5",
}
RELEASE_CONVERTER_SHA256 = "7d16646c0548302e2934d26fc6549c8aec68bed4f8268fb00c5c9022d627fcfb"

UCSD_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
HF_BASE = "https://huggingface.co/kkknight/MiniOneRec/resolve"
HF_REVISION = "365a03fc6601ed36abd69cc9a0a59025a3d31cdc"


@dataclass(frozen=True)
class Remote:
    url: str
    dest: Path
    size: int
    sha256: str


UCSD_FILES = [
    Remote(
        f"{UCSD_BASE}/categoryFilesSmall/Office_Products_5.json.gz",
        paths.UCSD / "Office_Products_5.json.gz",
        111_785_349,
        "325277af44e0566f21a6e03022f83b030e8e6e94300a0f7a2ae58490062d8670",
    ),
    Remote(
        f"{UCSD_BASE}/metaFiles2/meta_Office_Products.json.gz",
        paths.UCSD / "meta_Office_Products.json.gz",
        514_243_589,
        "6912cdf3edf0e36520cf8cfb4ec1f87542423021a48459dbb66193b1c2a1382d",
    ),
]


def _hf(name: str, size: int, sha256: str) -> Remote:
    return Remote(f"{HF_BASE}/{HF_REVISION}/Office_ckpt/{name}", paths.HF_CKPT / name, size, sha256)


HF_FILES = [
    _hf(
        "added_tokens.json",
        12_947,
        "682f42f571bea18c1fa3f4dd8161b0696a5ef802e716d21669742c7f9e678f1e",
    ),
    _hf(
        "tokenizer_config.json",
        108_851,
        "8b398f3d12a0d462c5cc583e5db6fe70c1c30d4df1e0830d1957397a2d972564",
    ),
    _hf("config.json", 1_322, "c2b98b1986dc97a0ee19f6216fc2f7a13f70a78954d5a5957b9d308e35ab56cd"),
    _hf(
        "special_tokens_map.json",
        496,
        "a36726f7fe394c1324021cac4a9ee3ad6c4f6285c0d98721f9b0b95888ec9905",
    ),
]
# Size and SHA-256 from the Hub's x-linked-size / x-linked-etag headers (LFS object hash).
# The redirect target is a CDN host (us.aws.cdn.hf.co at the time of writing).
HF_WEIGHTS = _hf(
    "model.safetensors",
    3_556_236_128,
    "69cddd1c19244e6925fcdc073fb54622f1367f749e193949f9456592f6d437fd",
)
# What generation needs besides the weights (M7): the full Qwen2 tokenizer with the 600 SID
# tokens (tokenizer.json is an LFS object; its SHA-256 is the Hub's LFS oid), the generation
# defaults, and trainer_state.json, which records what the checkpoint is (GRPO step 1038 of 3456).
HF_GENERATION_FILES = [
    _hf(
        "tokenizer.json",
        11_532_202,
        "727f175c5ff824b7cfa45c8577e293b356b255b215a504834c4451298bd5c50a",
    ),
    _hf(
        "vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    ),
    _hf(
        "merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
    ),
    _hf(
        "generation_config.json",
        121,
        "55b2b4462aedccc5858edd8d7aa0fa7aebf8a3bf9245f9a095ddeba8a9a1650a",
    ),
    _hf(
        "trainer_state.json",
        502_245,
        "a460e160ec6582f05622523e50f3fdf28e8a22b7405694c75247095b86f15f07",
    ),
]


TEXT_MODEL_REPO = "BAAI/bge-small-en-v1.5"
TEXT_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # main, 2024-02-22


def _text_model(name: str, size: int, sha256: str) -> Remote:
    return Remote(
        f"https://huggingface.co/{TEXT_MODEL_REPO}/resolve/{TEXT_MODEL_REVISION}/{name}",
        paths.TEXT_MODEL / name,
        size,
        sha256,
    )


# Only what sentence-transformers needs (no pytorch_model.bin / onnx copies).
TEXT_MODEL_FILES = [
    _text_model(
        "1_Pooling/config.json",
        190,
        "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c",
    ),
    _text_model(
        "config.json", 743, "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"
    ),
    _text_model(
        "config_sentence_transformers.json",
        124,
        "940d5f50db195fa6e5e6a4f122c095f77880de259d74b14a65779ed48bdd7c56",
    ),
    _text_model(
        "modules.json", 349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
    ),
    _text_model(
        "sentence_bert_config.json",
        52,
        "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
    ),
    _text_model(
        "special_tokens_map.json",
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    _text_model(
        "tokenizer.json",
        711_396,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    _text_model(
        "tokenizer_config.json",
        366,
        "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    ),
    _text_model(
        "vocab.txt", 231_508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    ),
    # LFS object: size and SHA-256 as listed by the Hub API.
    _text_model(
        "model.safetensors",
        133_466_304,
        "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _verify(remote: Remote) -> bool:
    if not remote.dest.exists() or remote.dest.stat().st_size != remote.size:
        return False
    return sha256_file(remote.dest) == remote.sha256


def fetch(remote: Remote) -> None:
    """Download `remote` unless an identical copy is already on disk."""
    if _verify(remote):
        print(f"ok        {remote.dest.relative_to(paths.ROOT)}")
        return
    remote.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = remote.dest.with_name(remote.dest.name + ".part")
    print(f"download  {remote.url}  ({remote.size / 1e6:,.1f} MB)")
    req = urllib.request.Request(remote.url, headers={"User-Agent": "recloop-data/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    size = tmp.stat().st_size
    digest = sha256_file(tmp)
    if size != remote.size or digest != remote.sha256:
        raise SystemExit(
            f"checksum mismatch for {remote.url}\n"
            f"  expected {remote.size} bytes, sha256 {remote.sha256}\n"
            f"  got      {size} bytes, sha256 {digest}\n"
            f"  partial file kept at {tmp}"
        )
    tmp.replace(remote.dest)
    print(f"verified  {remote.dest.relative_to(paths.ROOT)}")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(paths.MINIONEREC), *args], check=True, capture_output=True, text=True
    ).stdout


def fetch_minionerec() -> None:
    """Clone MiniOneRec (full history), pin the commit, extract the release-time converter."""
    if not (paths.MINIONEREC / ".git").exists():
        paths.THIRD_PARTY.mkdir(parents=True, exist_ok=True)
        print(f"clone     {MINIONEREC_URL}")
        subprocess.run(["git", "clone", MINIONEREC_URL, str(paths.MINIONEREC)], check=True)
    if _git("rev-parse", "HEAD").strip() != MINIONEREC_COMMIT:
        subprocess.run(
            [
                "git",
                "-C",
                str(paths.MINIONEREC),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--detach",
                MINIONEREC_COMMIT,
            ],
            check=True,
        )
    for rel, expected in MINIONEREC_FILES.items():
        got = sha256_file(paths.MINIONEREC / rel)
        if got != expected:
            raise SystemExit(f"third_party/MiniOneRec/{rel}: sha256 {got}, expected {expected}")
    print(f"ok        third_party/MiniOneRec @ {MINIONEREC_COMMIT[:12]}")

    converter = paths.MINIONEREC_RELEASE / "convert_dataset.py"
    paths.MINIONEREC_RELEASE.mkdir(parents=True, exist_ok=True)
    converter.write_text(_git("show", f"{RELEASE_COMMIT}:convert_dataset.py"))
    got = sha256_file(converter)
    if got != RELEASE_CONVERTER_SHA256:
        raise SystemExit(f"{converter}: sha256 {got}, expected {RELEASE_CONVERTER_SHA256}")
    print(f"ok        {converter.relative_to(paths.ROOT)} @ {RELEASE_COMMIT[:12]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="also download Office_ckpt/model.safetensors (3.56 GB) and the remaining tokenizer "
        "files (first needed in M7)",
    )
    args = parser.parse_args(argv)

    fetch_minionerec()
    for remote in UCSD_FILES + HF_FILES + TEXT_MODEL_FILES:
        fetch(remote)
    if args.with_weights:
        for remote in HF_GENERATION_FILES:
            fetch(remote)
        fetch(HF_WEIGHTS)
    else:
        print("skip      Office_ckpt/model.safetensors (pass --with-weights to fetch, 3.56 GB)")


if __name__ == "__main__":
    sys.exit(main())
