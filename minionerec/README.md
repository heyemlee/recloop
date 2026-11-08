# minionerec

MiniOneRec's own evaluation on Apple Silicon (M7): generation with the published Office
checkpoint on the Mac's GPU (MPS), and its cost. A uv project of its own with only what
MiniOneRec's evaluation imports, at the versions its `requirements.txt` pins (PyTorch 2.6.0,
transformers 4.57.1, ...); no CUDA wheels, deepspeed or bitsandbytes. Apple Silicon only.

- `upstream.py` copies `evaluate.py` and the modules it imports from the pinned clone
  (`third_party/MiniOneRec`, untouched) into `artifacts/minionerec/upstream/`, checks their
  SHA-256, and applies `evaluate.py.patch`: the `mps` device, no `bitsandbytes` import, a
  `--dtype` option, per-batch timing. Prompts, the SID prefix tree and the constrained beam
  search run as published.
- `generate.py` runs it over the test split or a seeded subset of users, then MiniOneRec's
  `calc.py` (`make minionerec-gen`).
- `cost.py` times 10 / 20 / 50 beams at batch 1 and 8, plus a prefill-only run
  (`make minionerec-cost`).

Scoring with RecLoop's metrics, the segments and the report are in the ml environment
(`ml/eval/minionerec.py`, `make minionerec` -> `docs/results/minionerec.md`). The checkpoint
comes from `make data-weights`.

MiniOneRec (github.com/AkaliKong/MiniOneRec) is Apache-2.0; the patch modifies a copy of its
`evaluate.py` and keeps that licence (`licenses/MiniOneRec-LICENSE`). The checkpoint
(huggingface.co/kkknight/MiniOneRec) is Apache-2.0 per its model card.
