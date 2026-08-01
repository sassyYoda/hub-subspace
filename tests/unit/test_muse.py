import os
from pathlib import Path

import pytest

from data.muse import build_gold_map, load_muse_dictionary, split_pairs

# Repo root, overridable with REPO_ROOT; defaults to three levels up from this file.
_REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


def test_build_gold_map(toy_pairs):
    gold = build_gold_map(toy_pairs)
    assert gold["cat"] == {"gato"}
    assert gold["border"] == {"frontera"}
    assert gold["boundary"] == {"frontera"}
    # boundary and frontier both also map to frontera; check the many-to-one direction
    # is recorded (src -> multiple targets is the same data shape)
    assert len(gold) == 10


def test_split_pairs_basic(toy_pairs):
    splits = split_pairs(toy_pairs, n_anchors=4, n_test=3, n_validation=0)
    assert len(splits["anchors"]) == 4
    assert len(splits["test"]) == 3


def test_split_pairs_no_leak(toy_pairs):
    splits = split_pairs(toy_pairs, n_anchors=5, n_test=4, n_validation=0)
    anchor_src = {s for s, _ in splits["anchors"]}
    test_src = {s for s, _ in splits["test"]}
    assert anchor_src.isdisjoint(test_src)


def test_split_pairs_raises_if_too_small(toy_pairs):
    with pytest.raises(ValueError, match="too small"):
        split_pairs(toy_pairs, n_anchors=100, n_test=100, n_validation=0)


@pytest.mark.skipif(
    not (_REPO_ROOT / "data/muse/en-es.txt").exists(),
    reason="MUSE dict not cached yet"
)
def test_load_real_dict():
    pairs = load_muse_dictionary("en-es")
    assert len(pairs) > 5000
    assert all(isinstance(s, str) and isinstance(t, str) for s, t in pairs)
