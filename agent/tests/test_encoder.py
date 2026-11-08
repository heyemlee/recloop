"""Query encoders. The bge test needs the `encoder` group, the model files and the database."""

import numpy as np
import pytest

from agent import paths
from agent.encoder import DIM, HashEncoder


def test_hash_encoder_is_deterministic_and_normalised():
    enc = HashEncoder()
    a, b = enc.encode("Brother HL-2240 toner"), enc.encode("Brother HL-2240 toner")
    assert a.shape == (DIM,) and np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)
    assert float(a @ enc.encode("brother hl2240 toners")) > 0.9  # same tokens, other spelling


def test_bge_reproduces_the_stored_item_embeddings():
    pytest.importorskip("sentence_transformers")
    psycopg = pytest.importorskip("psycopg")
    if not (paths.text_model_dir() / "model.safetensors").exists():
        pytest.skip("bge-small-en-v1.5 not downloaded (make data, or set RECLOOP_TEXT_MODEL)")
    try:
        conn = psycopg.connect(paths.database_url(), connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("database not reachable")
    with conn:
        row = conn.execute(
            "SELECT title, features, text_emb::text FROM items WHERE text_emb IS NOT NULL ORDER BY item_id LIMIT 1"
        ).fetchone()
    if row is None:
        pytest.skip("no text embeddings in the items table")
    from agent.encoder import BgeQueryEncoder

    title, features, stored = row
    # The item text recipe of ml.data.text_embed.item_text.
    feats = [f.strip() for f in (features or []) if f and f.strip()]
    text = title.strip() + (". " + "; ".join(feats) if feats else "")
    stored_vec = np.array([float(x) for x in stored.strip("[]").split(",")], dtype=np.float32)
    plain = BgeQueryEncoder().without_instruction()  # items were embedded without the prefix
    v = plain.encode(text)
    assert float(v @ stored_vec) > 0.9999
