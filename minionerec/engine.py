"""MiniOneRec's generation as a library: the offline replay and the nearline worker call it.

What evaluate.py (MiniOneRec @ 0cbd94de, Apache-2.0; licenses/MiniOneRec-LICENSE) does for one
batch, in a form a long-running process can call:

  * the checkpoint and the tokenizer load as in evaluate.py (patched for Apple Silicon, see
    upstream.py: the dtype, eager attention, then .to(device));
  * the SID prefix tree and the batch's generate() call are evaluate.py's code, copied below
    (`_prefix_tree`, `Generator.generate_encoded`) with the same GenerationConfig and arguments;
  * the prompts come from the staged data.EvalSidDataset and the constraint from the staged
    LogitProcessor.ConstrainedLogitsProcessor, imported from artifacts/minionerec/upstream (the
    pinned files, see upstream.stage), not copied.

One change, `prefill_once` (on by default): Hugging Face's beam search repeats each prompt once
per beam before its first forward pass, so a 50-beam batch of 8 prompts encodes 400 identical
copies -- 86% of a 50-beam request (docs/results/minionerec.md, Cost). share_prompt_pass() wraps
the model's forward: the first pass (empty cache, rows = prompts x beams, each prompt repeated
num_beams times) runs on one copy of each prompt, and the attention cache and logits are then
repeated num_beams times, which is what the full pass would have produced if every row were
computed alike. Decoding, the constraint, the beam bookkeeping and the order of the returned
sequences are transformers' own. Whether the GPU computes the rows alike is an empirical
question: minionerec.replay compares every SID sequence with the main run's.
"""

from __future__ import annotations

import functools
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from minionerec import upstream
from ml import paths

STAGE = paths.ARTIFACTS / "minionerec" / "upstream"
# evaluate.py's category name for the Office checkpoint and its fixed seed.
CATEGORY_TEXT = "office products"
SEED = 42
MAX_LEN = 2560  # evaluate.py's EvalSidDataset(max_len=...)
# Arguments of the model's forward that carry one row per sequence.
_ROW_ARGS = ("input_ids", "attention_mask", "position_ids", "inputs_embeds", "token_type_ids")


def import_upstream(stage_dir: Path = STAGE):
    """Stage the pinned upstream files (hash-checked) and import data / LogitProcessor."""
    upstream.stage(stage_dir)
    if str(stage_dir) not in sys.path:
        sys.path.insert(0, str(stage_dir))
    import data  # the staged data.py
    import LogitProcessor

    return data, LogitProcessor


def cap_mps_memory(gb: float) -> dict | None:
    """Cap what MPS may allocate in this process (torch.mps.set_per_process_memory_fraction of the
    recommended maximum): an allocation past it raises instead of pushing the machine into swap.
    None when gb <= 0 or the GPU is not MPS. Results do not depend on it."""
    import torch

    if gb <= 0 or not torch.backends.mps.is_available():
        return None
    limit = torch.mps.recommended_max_memory()
    fraction = min(1.0, gb * 2**30 / limit)
    torch.mps.set_per_process_memory_fraction(fraction)
    return {"gb": gb, "fraction": fraction, "recommended_max": limit}


def device_name() -> str:
    """evaluate.py's choice (patched): cuda, else mps, else cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def share_prompt_pass(model, num_beams: int) -> None:
    """Make the first forward pass of beam search encode each prompt once (see the module
    docstring). Calls that are not that pass -- a non-empty cache, rows that are not num_beams
    copies of each prompt -- go to the model unchanged."""
    import torch

    inner = model.forward

    @functools.wraps(inner)  # keeps the signature generate() inspects (logits_to_keep, ...)
    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        cache = kwargs.get("past_key_values")
        if (
            args
            or ids is None
            or cache is None
            or cache.get_seq_length() > 0
            or ids.shape[0] % num_beams
        ):
            return inner(*args, **kwargs)
        grouped = ids.view(ids.shape[0] // num_beams, num_beams, -1)
        if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
            return inner(*args, **kwargs)
        rows = slice(None, None, num_beams)
        one = {
            k: v[rows] if k in _ROW_ARGS and torch.is_tensor(v) else v for k, v in kwargs.items()
        }
        out = inner(**one)
        cache.batch_repeat_interleave(num_beams)  # the same cache object generate() holds
        out.logits = out.logits.repeat_interleave(num_beams, dim=0)
        return out

    model.forward = forward


def _get_hash(x) -> str:
    # evaluate.py's get_hash
    return "-".join(str(_) for _ in x)


def _prefix_tree(tokenizer, semantic_ids: list[str], prefix_index: int) -> dict[str, list[int]]:
    """evaluate.py's hash_dict: token-id prefix of a "### Response:\\n<sid>\\n" -> next tokens.

    Copied from evaluate.py's main() (the non-Llama branch: no BOS to drop)."""
    info_semantic = [f"""### Response:\n{_}""" for _ in semantic_ids]
    prefix_ids = [tokenizer(_).input_ids for _ in info_semantic]
    hash_dict: dict[str, set[int]] = {}
    for ids in prefix_ids:
        ids.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ids)):
            if i == prefix_index:
                hash_number = _get_hash(ids[:i])
            else:
                hash_number = _get_hash(ids[prefix_index:i])
            hash_dict.setdefault(hash_number, set()).add(ids[i])
    return {k: list(v) for k, v in hash_dict.items()}


@dataclass
class Batch:
    """One generate() call: per sample the num_beams SIDs, best first, and their beam scores."""

    sids: list[list[str]]
    scores: list[list[float]]
    seconds: float
    mps_allocated: int | None = None
    mps_driver_allocated: int | None = None


class Generator:
    def __init__(
        self,
        base_model: Path = paths.HF_CKPT,
        info_file: Path = paths.PUBLIC_INFO,
        *,
        num_beams: int = upstream.EVALUATE_SH["num_beams"],
        max_new_tokens: int = upstream.EVALUATE_SH["max_new_tokens"],
        length_penalty: float = upstream.EVALUATE_SH["length_penalty"],
        dtype: str = "bfloat16",
        attn_implementation: str = upstream.ATTN_IMPLEMENTATION,
        prefill_once: bool = True,
        device: str | None = None,
        stage_dir: Path = STAGE,
    ):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.data, self.logit_processor = import_upstream(stage_dir)
        # evaluate.py: random.seed(seed); set_seed(seed) before anything else.
        random.seed(SEED)
        np.random.seed(SEED)  # noqa: NPY002 (evaluate.py seeds numpy's global generator)
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        self.base_model = str(base_model)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.dtype = dtype
        self.prefill_once = prefill_once
        self.device = device or device_name()
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation or None,
        )
        model.eval()
        lines = Path(info_file).read_text().splitlines(keepends=True)
        # evaluate.py: semantic_id \t item_title \t item_id per line; the SID keeps a newline.
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in lines]
        tok = AutoTokenizer.from_pretrained(base_model)
        prefix_index = 4 if self.base_model.lower().find("gpt2") > -1 else 3
        self.hash_dict = _prefix_tree(tok, semantic_ids, prefix_index)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        self.tokenizer = tok
        model.config.pad_token_id = model.config.eos_token_id = tok.eos_token_id
        model.config.bos_token_id = tok.bos_token_id
        self.model = model.to(self.device)
        if prefill_once:
            share_prompt_pass(self.model, num_beams)
        self.attn_implementation = self.model.config._attn_implementation
        self.setup_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ prompts

    def dataset(self, csv: Path):
        """evaluate.py's dataset over a test-format CSV (EvalSidDataset, same arguments)."""
        return self.data.EvalSidDataset(
            train_file=str(csv),
            tokenizer=self.tokenizer,
            max_len=MAX_LEN,
            category=CATEGORY_TEXT,
            test=True,
            K=0,
            seed=SEED,
        )

    def encode_histories(self, histories: list[list[str]]) -> list[dict]:
        """Prompts for SID histories (oldest first), built by EvalSidDataset.pre exactly as for a
        CSV row whose history_item_sid is that list (the target is not part of a test prompt)."""
        import pandas as pd

        ds = self.data.EvalSidDataset.__new__(self.data.EvalSidDataset)
        # BaseDataset.__init__ without the seed: a worker must not reseed the global RNG.
        self.data.BaseDataset.__init__(
            ds, self.tokenizer, max_len=MAX_LEN, test=True, category=CATEGORY_TEXT, seed=None
        )
        ds.data = pd.DataFrame(
            {
                "history_item_sid": [repr(list(h)) for h in histories],
                "item_sid": [""] * len(histories),
            }
        )
        return [ds.pre(i) for i in range(len(histories))]

    # ------------------------------------------------------------------ generation

    def _allowed(self, batch_id, input_ids):
        # evaluate.py's prefix_allowed_tokens_fn_semantic
        hash_number = _get_hash(input_ids)
        if hash_number in self.hash_dict:
            return self.hash_dict[hash_number]
        return []

    def generate_encoded(self, encodings: list[dict]) -> Batch:
        """evaluate.py's evaluate() on one batch of encodings: left padding, the constrained
        beam search, the decoded SIDs; plus the beam scores (sequences_scores)."""
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        t0 = time.perf_counter()
        tok, model, nb = self.tokenizer, self.model, self.num_beams
        max_len = max(len(e["input_ids"]) for e in encodings)
        input_ids, attention_mask = [], []
        for e in encodings:
            n = len(e["input_ids"])
            input_ids.append([tok.pad_token_id] * (max_len - n) + e["input_ids"])
            attention_mask.append([0] * (max_len - n) + [1] * n)
        config = GenerationConfig(
            num_beams=nb,
            length_penalty=self.length_penalty,
            num_return_sequences=nb,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
        )
        with torch.no_grad():
            clp = self.logit_processor.ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=self._allowed,
                num_beams=nb,
                base_model=self.base_model,
                eos_token_id=model.config.eos_token_id,
            )
            out = model.generate(
                torch.tensor(input_ids).to(self.device),
                attention_mask=torch.tensor(attention_mask).to(self.device),
                generation_config=config,
                do_sample=False,  # as a kwarg on purpose: see evaluate.py
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList([clp]),
            )
        completions = out.sequences[:, max_len:]
        text = tok.batch_decode(completions, skip_special_tokens=True)
        text = [t.split("Response:\n")[-1].strip() for t in text]
        scores = out.sequences_scores.float().cpu().tolist()
        k = len(encodings)
        batch = Batch(
            sids=[text[i * nb : (i + 1) * nb] for i in range(k)],
            scores=[scores[i * nb : (i + 1) * nb] for i in range(k)],
            seconds=0.0,
        )
        if self.device == "mps":
            batch.mps_allocated = torch.mps.current_allocated_memory()
            batch.mps_driver_allocated = torch.mps.driver_allocated_memory()
            torch.mps.empty_cache()  # as the patched evaluate.py: batch lengths differ
        batch.seconds = time.perf_counter() - t0
        return batch

    def settings(self) -> dict:
        return {
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "length_penalty": self.length_penalty,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "prefill": "once" if self.prefill_once else "per_beam",
        }
