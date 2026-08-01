import numpy as np
import pytest

from core.centering import mean_center


def test_none_returns_copy(rng):
    emb = {"en": rng.randn(10, 8).astype(np.float32)}
    out = mean_center(emb, mode="none")
    np.testing.assert_array_equal(out["en"], emb["en"])
    # but it should be a different object (we promise not to mutate inputs)
    assert out["en"] is not emb["en"]


def test_per_layer_per_lang_zeros_mean(rng):
    emb = {
        "en": rng.randn(10, 8).astype(np.float32) + 5.0,  # shifted off-origin
        "es": rng.randn(12, 8).astype(np.float32) - 3.0,
    }
    out = mean_center(emb, mode="per_layer_per_lang")
    np.testing.assert_allclose(out["en"].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out["es"].mean(axis=0), 0.0, atol=1e-5)


def test_global_uses_grand_mean(rng):
    en = rng.randn(10, 8).astype(np.float32) + 5.0
    es = rng.randn(10, 8).astype(np.float32) - 5.0
    emb = {"en": en, "es": es}
    out = mean_center(emb, mode="global")
    # The grand mean of out["en"] and out["es"] combined should be ~0.
    combined = np.concatenate([out["en"], out["es"]], axis=0)
    np.testing.assert_allclose(combined.mean(axis=0), 0.0, atol=1e-5)
    # But neither language alone need be zero-mean.
    assert not np.allclose(out["en"].mean(axis=0), 0.0, atol=1e-3)


def test_layer_dim(rng):
    n_layers, n_words, dim = 3, 8, 4
    emb = {"en": rng.randn(n_layers, n_words, dim).astype(np.float32) + 2.0}
    out = mean_center(emb, mode="per_layer_per_lang", layer_dim=True)
    # mean over the words axis per layer should be ~0
    for layer in range(n_layers):
        np.testing.assert_allclose(out["en"][layer].mean(axis=0), 0.0, atol=1e-5)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        mean_center({"en": np.zeros((3, 4))}, mode="nope")  # type: ignore[arg-type]
