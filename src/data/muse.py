"""MUSE bilingual dictionary loader.

Handles arbitrary language pairs and builds the many-to-one gold dict used by the
BLI metric in core/metrics.py.

MUSE dictionaries: https://github.com/facebookresearch/MUSE
File format: one whitespace-separated (src_word, tgt_word) pair per line. For en-es
the file is `en-es.txt` and lines are "en_word es_word". MUSE provides both directions.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{pair}.txt"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def load_muse_dictionary(
    pair: str = "en-es",
    dict_dir: str | Path = "data/muse",
    lowercase: bool = True,
    download_if_missing: bool = True,
) -> list[tuple[str, str]]:
    """Return list of (src_word, tgt_word) pairs in MUSE-file order.

    `pair` follows MUSE convention, e.g. 'en-es', 'en-de', 'en-zh'. For BLI we treat
    src as the query language and tgt as the gallery language.
    """
    dict_dir = Path(dict_dir)
    filepath = dict_dir / f"{pair}.txt"
    if not filepath.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"MUSE dict not found: {filepath}")
        _download(DEFAULT_DICT_URL.format(pair=pair), filepath)

    pairs: list[tuple[str, str]] = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if lowercase:
                src, tgt = src.lower(), tgt.lower()
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def build_gold_map(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Many-to-one gold: src_word -> set of acceptable tgt translations.

    Many src words have several valid translations in MUSE, so evaluation checks
    `predicted_tgt_word in gold[src_word]` rather than array-index equality.
    See core/metrics.py.
    """
    gold: dict[str, set[str]] = {}
    for src, tgt in pairs:
        gold.setdefault(src, set()).add(tgt)
    return gold


def split_pairs(
    pairs: list[tuple[str, str]],
    n_anchors: int,
    n_test: int,
    n_validation: int = 0,
    shuffle_seed: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Split pairs into anchors / validation / test deterministically.

    If `shuffle_seed` is None, splits are taken in MUSE-file order. If a seed is
    given, pairs are shuffled deterministically first, for multi-seed runs.
    """
    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        pairs = list(pairs)
        rng.shuffle(pairs)

    total_needed = n_anchors + n_validation + n_test
    if len(pairs) < total_needed:
        raise ValueError(
            f"MUSE dict too small: need {total_needed} pairs, have {len(pairs)}"
        )
    anchors = pairs[:n_anchors]
    validation = pairs[n_anchors : n_anchors + n_validation]
    test = pairs[n_anchors + n_validation : n_anchors + n_validation + n_test]

    # No anchor word may leak into the test set.
    anchor_src = {s for s, _ in anchors}
    anchor_tgt = {t for _, t in anchors}
    test = [(s, t) for s, t in test if s not in anchor_src and t not in anchor_tgt]
    if len(test) < n_test:
        # Top up from the tail to reach n_test.
        extra_pool = pairs[n_anchors + n_validation + n_test :]
        for s, t in extra_pool:
            if s in anchor_src or t in anchor_tgt:
                continue
            test.append((s, t))
            if len(test) >= n_test:
                break

    return {"anchors": anchors, "validation": validation, "test": test[:n_test]}
