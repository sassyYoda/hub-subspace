"""HuggingFace encoder extractor: mBERT, XLM-R (base / large).

In isolation context, each word is tokenized standalone (with the model's default
special-token wrapping) and all hidden states are read from a single forward pass.
Subwords belonging to the word are pooled per `subword_agg`.

Caching: each layer is saved under `results/cache/{model}/{lang}/{key}/L{n}.npy`,
where `key` hashes the full extraction config. Reloads are sub-second.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .aggregators import AggMode, aggregate_subwords
from .base import EmbeddingExtractor
from .context_pool import ContextPoolSpec, IsolationSpec


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HFEncoderExtractor(EmbeddingExtractor):
    """Encoder-only models (BERT family, XLM-R) via HuggingFace transformers.

    `name` is the short label used for the cache directory; `device` is 'mps',
    'cuda', 'cpu', or None to autodetect.
    """

    def __init__(
        self,
        hf_id: str,
        name: str,
        cache_dir: str | Path = "results/cache",
        batch_size: int = 64,
        device: str | torch.device | None = None,
    ):
        self.hf_id = hf_id
        self.name = name
        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = torch.device(device) if device else _device()

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

        # n_layers includes the embedding layer (hidden_states[0]).
        # config.num_hidden_layers excludes it.
        self.n_layers = self.model.config.num_hidden_layers + 1
        self.hidden_dim = self.model.config.hidden_size

    # Cache key is per (model, lang, word_hash, context_pool, subword_agg) but not
    # per layer-list; each layer is a separate file under one directory, so a
    # request for layers=[8] and a later layers='all' share the layer-8 file and
    # only the missing layers get extracted.
    def _cache_dir_for(
        self,
        lang: str,
        n_words: int,
        word_hash: str,
        context_pool: ContextPoolSpec,
        subword_agg: AggMode,
    ) -> Path:
        spec = {
            "hf_id": self.hf_id,
            "lang": lang,
            "n_words": n_words,
            "word_hash": word_hash,
            "context": context_pool.__class__.__name__,
            "context_dict": context_pool.__dict__,
            "agg": subword_agg,
        }
        key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / lang / key

    def _resolve_layers(self, layers: int | Sequence[int] | str) -> list[int]:
        if layers == "all":
            return list(range(self.n_layers))
        if isinstance(layers, int):
            return [layers]
        return list(layers)

    def extract(
        self,
        words: Sequence[str],
        lang: str,
        layers: int | Sequence[int] | str = "all",
        context_pool: ContextPoolSpec | None = None,
        subword_agg: AggMode = "mean",
    ) -> dict[int, np.ndarray]:
        if context_pool is None:
            context_pool = IsolationSpec()
        layer_list = self._resolve_layers(layers)

        words = list(words)
        # Word-order-sensitive hash so reordering invalidates cache.
        word_hash = hashlib.sha1("|".join(words).encode()).hexdigest()[:12]
        cdir = self._cache_dir_for(lang, len(words), word_hash, context_pool, subword_agg)

        # Load whatever layers are cached; extract only the missing ones below.
        out: dict[int, np.ndarray] = {}
        missing = []
        for L in layer_list:
            p = cdir / f"L{L}.npy"
            if p.exists():
                out[L] = np.load(p)
            else:
                missing.append(L)

        if not missing:
            return out

        if not isinstance(context_pool, IsolationSpec):
            raise NotImplementedError(
                f"context_pool={context_pool} not yet implemented; "
                "only IsolationSpec works in v0.1"
            )

        # NaN-fill so any word that fails extraction stays flagged.
        n_words = len(words)
        for L in missing:
            out[L] = np.full((n_words, self.hidden_dim), np.nan, dtype=np.float32)

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, n_words, self.batch_size),
                desc=f"{self.name}:{lang}",
                leave=False,
            ):
                batch_words = words[batch_start : batch_start + self.batch_size]
                enc = self.tokenizer(
                    batch_words,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                special_mask = enc["special_tokens_mask"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.hidden_states  # tuple of (B, T, H)

                for L in missing:
                    hs_L = hidden_states[L].cpu().float().numpy()
                    for j, word in enumerate(batch_words):
                        sub_mask = (special_mask[j] == 0) & (attention_mask[j].cpu() == 1)
                        sub_idx = sub_mask.nonzero(as_tuple=True)[0].tolist()
                        if not sub_idx:
                            continue
                        out[L][batch_start + j] = aggregate_subwords(
                            hs_L[j], sub_idx, mode=subword_agg
                        )

        cdir.mkdir(parents=True, exist_ok=True)
        for L in missing:
            np.save(cdir / f"L{L}.npy", out[L])
        return out


def build_mbert(**kw) -> HFEncoderExtractor:
    return HFEncoderExtractor(
        hf_id="bert-base-multilingual-cased", name="mbert", **kw
    )


def build_xlmr_base(**kw) -> HFEncoderExtractor:
    return HFEncoderExtractor(hf_id="xlm-roberta-base", name="xlmr_base", **kw)


def build_xlmr_large(**kw) -> HFEncoderExtractor:
    return HFEncoderExtractor(hf_id="xlm-roberta-large", name="xlmr_large", **kw)
