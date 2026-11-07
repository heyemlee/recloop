"""Our SID-rule metrics against MiniOneRec's calc.py on a small hand-made result file."""

import numpy as np
import pytest

from ml import paths
from ml.eval import calc_check, metrics
from ml.eval.splits import Split, item_sids

pytestmark = pytest.mark.skipif(
    not calc_check.CALC_PY.exists() or not paths.PUBLIC_INFO.exists(),
    reason="third_party/MiniOneRec missing: run `make data` first",
)


def test_calc_py_agrees_on_a_small_case(tmp_path):
    sids = item_sids()
    rng = np.random.default_rng(7)
    n = 40
    top = np.stack([rng.permutation(3459)[: calc_check.TOP_N] for _ in range(n)])
    target = np.where(
        rng.random(n) < 0.6, top[np.arange(n), rng.integers(0, 50, n)], rng.integers(0, 3459, n)
    )
    # include a pair of items that share a SID (112 and 118): predicting 118 hits target 112
    top[0, 2], target[0] = 118, 112
    split = Split("test", np.arange(n), target, [np.array([1, 2])] * n)

    codes = np.unique(np.asarray(sids), return_inverse=True)[1]
    ranks = metrics.target_ranks(top, target, codes)
    ours = metrics.summarize(metrics.per_sample(ranks, metrics.CALC_KS))
    result = calc_check.check("toy", split, top, ours, sids, tmp_path)
    assert result["ok"], result["rows"]
    assert result["invalid_predictions"] == 0
