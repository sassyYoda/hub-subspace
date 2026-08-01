"""EmbeddingExtractor interface.

Lets experiments iterate over models without branching on the backend. Every
extractor returns the same shape: `dict[layer_idx, (n_words, hidden_dim)]`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .aggregators import AggMode
from .context_pool import ContextPoolSpec


class EmbeddingExtractor(ABC):
    name: str
    n_layers: int          # includes embedding layer (layer 0)
    hidden_dim: int

    @abstractmethod
    def extract(
        self,
        words: Sequence[str],
        lang: str,
        layers: int | Sequence[int] | str = "all",
        context_pool: ContextPoolSpec | None = None,
        subword_agg: AggMode = "mean",
    ) -> dict[int, np.ndarray]:
        """Extract word embeddings.

        `layers` is 'all', a single int, or a list of layer indices, where layer 0
        is the static embedding layer and layer N the final hidden state.

        Returns a dict mapping layer_idx -> (n_words, hidden_dim) ndarray, where row
        i corresponds to words[i]. Words that fail to extract (e.g. tokenizer
        failure) get a NaN row, which downstream code drops.
        """

    def warm_cache(
        self,
        words: Sequence[str],
        lang: str,
        layers: int | Sequence[int] | str = "all",
        context_pool: ContextPoolSpec | None = None,
        subword_agg: AggMode = "mean",
    ) -> None:
        """Optionally write extraction results to disk for cheap reload.

        The default calls .extract(...) and relies on the implementation to cache
        internally. Override for explicit disk-streamed extraction.
        """
        self.extract(words, lang, layers, context_pool, subword_agg)
