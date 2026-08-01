"""Subword aggregation strategies.

Multilingual transformers tokenize words into multiple subword pieces, so getting a
single word embedding for BLI requires pooling across the subword positions. Most
BLI-on-LLM work uses mean (Feng et al. 2025); `first` mirrors raw BERT-style
extraction and serves as a negative control.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

AggMode = Literal["mean", "first", "last", "sum"]


def aggregate_subwords(
    hidden_states: np.ndarray,
    subword_indices: list[int],
    mode: AggMode = "mean",
) -> np.ndarray:
    """Pool the subwords of one word into a single (hidden_dim,) vector.

    `hidden_states` is the full per-token states (seq_len, hidden_dim);
    `subword_indices` are the positions belonging to this word.
    """
    if not subword_indices:
        raise ValueError("subword_indices must be non-empty")
    sel = hidden_states[subword_indices]
    if mode == "mean":
        return sel.mean(axis=0)
    if mode == "sum":
        return sel.sum(axis=0)
    if mode == "first":
        return sel[0]
    if mode == "last":
        return sel[-1]
    raise ValueError(f"unknown subword agg mode: {mode}")
