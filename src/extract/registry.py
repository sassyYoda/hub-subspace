"""Map a string model name to an EmbeddingExtractor builder."""
from __future__ import annotations

from .base import EmbeddingExtractor
from .hf_encoder import build_mbert, build_xlmr_base, build_xlmr_large

REGISTRY = {
    "mbert": build_mbert,
    "xlmr_base": build_xlmr_base,
    "xlmr_large": build_xlmr_large,
}


def build_extractor(name: str, **kw) -> EmbeddingExtractor:
    if name not in REGISTRY:
        raise ValueError(f"unknown model: {name}. known: {list(REGISTRY)}")
    return REGISTRY[name](**kw)
