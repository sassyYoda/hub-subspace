"""Mean-centering of contextual embeddings.

Chang, Tu, Bergen (EMNLP 2022), "The Geometry of Multilingual Language Model
Representations", show that mean-centering per (layer, language) before measuring
cosine similarity removes the dominant language-sensitive axis and lets the
remaining geometry reveal cross-lingual structure. Without it, cosine similarity
saturates near 1.0 and the cross-lingual structure is hidden.

Modes:
    - 'per_layer_per_lang': subtract the mean of {this layer, this language}.
      Recommended default per Chang et al. 2022.
    - 'per_lang':           subtract this language's mean (averaged over layers).
      Useful when working with a single-layer extraction.
    - 'global':             subtract the grand mean over all (layer, language) samples.
      Negative control; should not fix anisotropy on its own.
    - 'none':               return as-is.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

CenteringMode = Literal["per_layer_per_lang", "per_lang", "global", "none"]


def mean_center(
    embeddings_by_lang: dict[str, np.ndarray],
    mode: CenteringMode = "per_layer_per_lang",
    layer_dim: bool = False,
) -> dict[str, np.ndarray]:
    """Mean-center a dict of {lang: embeddings}.

    For single-layer use, pass `{lang: (n_words, hidden)}`. For multi-layer use
    `layer_dim=True` and pass `{lang: (n_layers, n_words, hidden)}`.

    Returns a new dict; does not mutate inputs.
    """
    if mode == "none":
        return {lang: emb.copy() for lang, emb in embeddings_by_lang.items()}

    if mode == "per_layer_per_lang":
        out: dict[str, np.ndarray] = {}
        for lang, emb in embeddings_by_lang.items():
            if layer_dim:
                mean = emb.mean(axis=1, keepdims=True)  # (n_layers, 1, hidden)
            else:
                mean = emb.mean(axis=0, keepdims=True)  # (1, hidden)
            out[lang] = emb - mean
        return out

    if mode == "per_lang":
        # Same as per_layer_per_lang when single-layer; over layers, collapse first.
        out = {}
        for lang, emb in embeddings_by_lang.items():
            if layer_dim:
                mean = emb.mean(axis=(0, 1), keepdims=True)  # (1, 1, hidden)
            else:
                mean = emb.mean(axis=0, keepdims=True)
            out[lang] = emb - mean
        return out

    if mode == "global":
        all_embs = np.concatenate(
            [e.reshape(-1, e.shape[-1]) for e in embeddings_by_lang.values()], axis=0
        )
        global_mean = all_embs.mean(axis=0)
        return {lang: emb - global_mean for lang, emb in embeddings_by_lang.items()}

    raise ValueError(f"unknown centering mode: {mode}")
