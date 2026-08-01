"""Cache Llama-3-8B last-token activations for the decoder experiments.

`decoder_ablation.py` and `downstream_patch.py` both read per-word activation
caches at `results/cache/llama3_8b_4bit/{lang}_L{LAYER}_last.npz`. This script
builds them: for every word in the MUSE dictionaries of the requested pairs, it
runs the word in isolation through Llama-3-8B and stores the layer-LAYER hidden
state of the final token.

Runs through Apple MLX, so it needs Apple Silicon. Caching is incremental: words
already present in a cache file are skipped, so reruns are cheap.

    PAIRS=en-es,en-de LAYER=16 python experiments/decoder_extract.py
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
from mlx_lm import load

from data.muse import load_muse_dictionary

PAIRS = os.environ.get("PAIRS", "en-es,en-de").split(",")
LAYER = int(os.environ.get("LAYER", "16"))
MODEL_ID = "mlx-community/Meta-Llama-3-8B-4bit"
CACHE_DIR = "results/cache/llama3_8b_4bit"

_model = None
_tok = None


def get_model():
    global _model, _tok
    if _model is None:
        print(f"loading {MODEL_ID}", flush=True)
        _model, _tok = load(MODEL_ID)
    return _model, _tok


def words_by_language(pairs):
    """Collect the distinct source and target words per language across pairs."""
    words: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")
        for lang in (src_lang, tgt_lang):
            words.setdefault(lang, [])
            seen.setdefault(lang, set())
        for src, tgt in load_muse_dictionary(pair):
            if src not in seen[src_lang]:
                seen[src_lang].add(src)
                words[src_lang].append(src)
            if tgt not in seen[tgt_lang]:
                seen[tgt_lang].add(tgt)
                words[tgt_lang].append(tgt)
    return words


def last_token_state(model, tok, word):
    ids = tok.encode(word)
    if len(ids) == 0:
        ids = tok.encode(" " + word)
    h = model.model.embed_tokens(mx.array([ids]))
    for li, layer in enumerate(model.model.layers):
        h = layer(h, mask=None, cache=None)
        if li == LAYER:
            break
    return np.array(h[0]).astype(np.float32)[-1]


def extract_language(lang, words):
    path = f"{CACHE_DIR}/{lang}_L{LAYER}_last.npz"
    cache: dict[str, np.ndarray] = {}
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        cache = {w: v for w, v in zip(d["words"], d["vecs"])}
    missing = [w for w in words if w not in cache]
    if not missing:
        print(f"{lang}: {len(cache)} words already cached", flush=True)
        return
    model, tok = get_model()
    for i, w in enumerate(missing):
        cache[w] = last_token_state(model, tok, w)
        if i % 500 == 0:
            print(f"  {lang}: {i}/{len(missing)}", flush=True)
    np.savez(
        path,
        words=np.array(list(cache.keys()), dtype=object),
        vecs=np.array(list(cache.values()), dtype=np.float32),
    )
    print(f"{lang}: cached {len(cache)} words ({len(missing)} new)", flush=True)


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    for lang, words in words_by_language(PAIRS).items():
        extract_language(lang, words)
