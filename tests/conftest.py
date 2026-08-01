"""Shared test fixtures."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def toy_pairs():
    """Tiny MUSE-style pair list, with one many-to-one case (border/boundary -> frontera)."""
    return [
        ("cat", "gato"),
        ("dog", "perro"),
        ("house", "casa"),
        ("water", "agua"),
        ("book", "libro"),
        ("border", "frontera"),
        ("boundary", "frontera"),  # many-to-one
        ("frontier", "frontera"),  # many-to-one
        ("king", "rey"),
        ("queen", "reina"),
    ]


@pytest.fixture
def aligned_embeddings(rng):
    """Two 'languages' that are perfectly aligned: tgt = src + small noise.
    P@1 should be ~1.0 on any reasonable metric."""
    n_words, dim = 20, 16
    src = rng.randn(n_words, dim).astype(np.float32)
    tgt = src + 0.01 * rng.randn(n_words, dim).astype(np.float32)
    words = [f"w{i}" for i in range(n_words)]
    return src, tgt, words
