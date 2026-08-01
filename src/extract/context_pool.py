"""Context pooling specifications for word-level embedding extraction.

Three regimes:
  - `isolation`: feed the word alone (or in a minimal template). Simplest, and
    most exposed to anisotropy since there is no context averaging to dilute it.
  - `corpus_k`: find K sentences containing the word in a corpus (Flores-200 by
    default), extract the word's embedding in each, average. Peng & Søgaard 2024
    show K >= 16 stabilizes the per-word centroid.
  - `prompt`: wrap the word in a fixed prompt template (e.g. 'The word "X" means').
    Useful for decoder LMs (Llama), where isolation can degenerate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IsolationSpec:
    """Feed the word alone (just the word, no template)."""
    kind: Literal["isolation"] = "isolation"


@dataclass(frozen=True)
class CorpusKSpec:
    """Average over K sentences containing the word from `corpus` (e.g. 'flores200')."""
    k: int = 32
    corpus: str = "flores200"
    kind: Literal["corpus_k"] = "corpus_k"


@dataclass(frozen=True)
class PromptSpec:
    """Wrap word in a per-language prompt template, e.g. 'The word "{w}" refers to'."""
    template: str = 'The word "{w}" refers to'
    kind: Literal["prompt"] = "prompt"


ContextPoolSpec = IsolationSpec | CorpusKSpec | PromptSpec


def from_dict(d: dict) -> ContextPoolSpec:
    """Build a spec from a parsed YAML dict."""
    kind = d.get("kind", "isolation")
    if kind == "isolation":
        return IsolationSpec()
    if kind == "corpus_k":
        return CorpusKSpec(k=d.get("k", 32), corpus=d.get("corpus", "flores200"))
    if kind == "prompt":
        return PromptSpec(template=d.get("template", 'The word "{w}" refers to'))
    raise ValueError(f"unknown context_pool kind: {kind}")
