import numpy as np

from ml.eval.bootstrap import Resampler


def test_replicates_are_user_level_ratios():
    users = np.array([7, 7, 7, 9])
    values = np.array([1.0, 1.0, 1.0, 0.0])
    r = Resampler.build(users, b=500, seed=0)
    reps = r.replicates(values)
    # drawing user 7 twice -> 1.0; user 9 twice -> 0.0; one each -> 3/4
    assert set(np.round(reps, 6)) <= {0.0, 0.75, 1.0}
    assert np.isclose(r.weights.sum(axis=1), 2).all()


def test_interval_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(3)
    users = rng.integers(0, 300, size=2000)
    values = (rng.random(2000) < 0.2).astype(float)
    a = Resampler.build(users, seed=11).interval(values)
    b = Resampler.build(users, seed=11).interval(values)
    assert a == b
    assert a[0] < values.mean() < a[1]


def test_clustered_samples_widen_the_interval():
    # Same values; in `clustered` each user has 10 identical samples.
    rng = np.random.default_rng(5)
    per_user = (rng.random(200) < 0.3).astype(float)
    clustered_users = np.repeat(np.arange(200), 10)
    clustered_values = np.repeat(per_user, 10)
    iid_users = np.arange(2000)
    iid_values = clustered_values.copy()
    lo_c, hi_c = Resampler.build(clustered_users, seed=1).interval(clustered_values)
    lo_i, hi_i = Resampler.build(iid_users, seed=1).interval(iid_values)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)
