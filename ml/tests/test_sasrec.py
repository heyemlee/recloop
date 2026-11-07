import numpy as np
import pytest
import torch

from ml import paths
from ml.eval.splits import HISTORY_LEN, Split, load_split, train_sequences
from ml.models.sasrec import (
    SASRec,
    SASRecConfig,
    SASRecNet,
    encode_histories,
    score_histories,
    train,
    training_windows,
)

N = 30
TINY = SASRecConfig(
    hidden=16, blocks=2, heads=2, dropout=0.0, max_epochs=2, patience=5, batch_size=8
)


def _net(cfg=TINY, seed=0) -> SASRecNet:
    torch.manual_seed(seed)
    return SASRecNet(N, cfg).eval()


def test_training_windows_are_the_published_train_rows():
    seqs = {7: np.array([1, 2, 3]), 9: np.arange(14)}
    hist, target = training_windows(seqs, maxlen=HISTORY_LEN)
    assert hist.shape == (2 + 13, HISTORY_LEN)
    assert hist[0].tolist() == [-1] * 9 + [1] and target[0] == 2
    assert hist[1].tolist() == [-1] * 8 + [1, 2] and target[1] == 3
    assert hist[-1].tolist() == list(range(3, 13)) and target[-1] == 13  # last 10 only


@pytest.mark.skipif(not paths.public_split_csv("train").exists(), reason="no data: run `make data`")
def test_training_windows_match_train_csv():
    train_split = load_split("train")
    hist, target = training_windows(train_sequences(train_split))
    got = {(tuple(h[h >= 0]), t) for h, t in zip(hist, target, strict=True)}
    want = {(tuple(h), t) for h, t in zip(train_split.history, train_split.target, strict=True)}
    assert len(hist) == len(train_split) and got == want


def test_attention_is_causal():
    net = _net()
    x = torch.randint(1, N + 1, (4, HISTORY_LEN))
    y = x.clone()
    y[:, -1] = (y[:, -1] % N) + 1  # change only the last item
    with torch.no_grad():
        fx, fy = net.log2feats(x), net.log2feats(y)
    torch.testing.assert_close(fx[:, :-1], fy[:, :-1])
    assert not torch.allclose(fx[:, -1], fy[:, -1])


def test_padding_is_masked():
    """Whatever the padding row holds, it must not change the user vector."""
    net = _net()
    x = torch.zeros((3, HISTORY_LEN), dtype=torch.long)
    x[0, -2:] = torch.tensor([3, 4])
    x[1, -5:] = torch.tensor([1, 2, 3, 4, 5])
    x[2, :] = torch.arange(1, HISTORY_LEN + 1)
    with torch.no_grad():
        before = net(x)
        net.item_emb.weight[0] = torch.randn(TINY.hidden) * 5
        after = net(x)
    torch.testing.assert_close(before, after)
    assert torch.isfinite(before).all()


def test_scores_are_dot_products_with_shared_item_embeddings():
    net = _net()
    history = np.array([[-1] * 8 + [0, 5], [-1] * 9 + [7]])
    scores = score_histories(net, history, torch.device("cpu"))
    with torch.no_grad():
        users = net(encode_histories(history))
        want = (users @ net.item_emb.weight[1:].T).numpy()
    assert scores.shape == (2, N)
    np.testing.assert_allclose(scores, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("loss", ["ce", "bce"])
@pytest.mark.parametrize("positions", ["all", "last"])
def test_training_runs_and_is_reproducible(loss, positions):
    rng = np.random.default_rng(0)
    seqs = {u: rng.integers(0, N, size=rng.integers(2, 15)) for u in range(40)}
    valid = Split(
        "v", np.arange(5), rng.integers(0, N, 5), [rng.integers(0, N, 3) for _ in range(5)]
    )
    cfg = SASRecConfig(**{**TINY.__dict__, "loss": loss, "loss_positions": positions})
    tie = np.arange(N)
    a, log_a = train(cfg, seqs, N, valid, tie)
    b, log_b = train(cfg, seqs, N, valid, tie)
    assert len(log_a.epochs) == cfg.max_epochs and np.isfinite(log_a.epochs[-1]["loss"])
    assert log_a.epochs == log_b.epochs  # same seed, same CPU -> same run
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k])


def test_checkpoint_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    seqs = {u: rng.integers(0, N, size=6) for u in range(20)}
    valid = Split(
        "v", np.arange(4), rng.integers(0, N, 4), [rng.integers(0, N, 2) for _ in range(4)]
    )
    model = SASRec(TINY).fit(seqs, N, valid=valid)
    model.save(tmp_path / "m.pt")
    loaded = SASRec.load(tmp_path / "m.pt")
    np.testing.assert_array_equal(model.score(valid), loaded.score(valid))
    assert loaded.describe() == model.describe()


def test_onnx_export_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    net = _net(SASRecConfig(hidden=16, blocks=2, heads=1, dropout=0.0))
    example = torch.randint(0, N + 1, (4, HISTORY_LEN))
    path = tmp_path / "enc.onnx"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            (example,),
            str(path),
            input_names=["history"],
            output_names=["user_vector"],
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=64)},),
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for b in (1, 7):
        x = torch.randint(0, N + 1, (b, HISTORY_LEN))
        x[0, :6] = 0
        got = sess.run(None, {"history": x.numpy()})[0]
        with torch.no_grad():
            np.testing.assert_allclose(got, net(x).numpy(), atol=1e-5)
