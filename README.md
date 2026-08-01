# hub-subspace

Multilingual models translate words far less reliably than they translate
sentences. We show the gap is hubness, not missing alignment: ten high-variance
directions of the single-token activation space work as a bidirectional causal
dial. Project them out and stranded words come back; amplify them and retrieval
collapses; matched random directions do neither.

Paper: **Hubness, Not Misalignment: A Causal Dial for Cross-Lingual Word
Retrieval in Multilingual Models** (under review; arXiv link TBD).

Using bilingual lexicon induction (BLI) as a behavioral probe across eleven
languages and two encoders, the paper shows that word-level retrieval is the
worst case of a smooth context-length curve, and that the bottleneck is hubness:
a few hub words are the nearest neighbor of up to 134 queries while 82% of the
gallery is never retrieved, with lexical ambiguity the strongest per-word
predictor under controls. The hub structure is concentrated, not low-dimensional:
the top ten directions hold about 14% of the variance on mBERT and 23% on XLM-R
while effective rank stays at several hundred of 768. The subspace estimated on
en-es transfers unchanged to the other four pairs (113% and 93% of the
pair-native recovery gain on mBERT and XLM-R) and across disjoint word splits.
A frozen Llama-3-8B carries the same subspace: removing it raises the correct
token's log-probability (+0.09 nats, 8/8 seeds) without changing the top-1
output, which we read as a property of the representation, not a
translation-quality gain. CSLS and inverted softmax flatten the same in-degree
distribution the ablation targets. This repository holds the library and the
experiment scripts that produce those results.

## Install

```bash
uv venv --python 3.11 .venv
uv pip install -e .
uv run pytest
```

The library lives under `src/` and imports as the top-level packages `core`,
`data`, and `extract` (with `src` on the path). The experiments are standalone
scripts under `experiments/`.

## Data

- **MUSE** bilingual dictionaries download on first use into `data/muse/`.
- **MNLI / XNLI** are pulled through the Hugging Face `datasets` library when you
  run the XNLI scripts; no manual download.
- **FLORES-200** is needed for the length curve and the per-language hubness
  measurement. Download it and unpack so the files land under
  `data/flores200/flores200_dataset/`:

```bash
mkdir -p data/flores200
curl -L -o flores200.tar.gz \
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar xzf flores200.tar.gz -C data/flores200/
```

## Reproduce the experiments

Each script reads its settings from environment variables and writes JSON to
`results/`. The defaults reproduce the reported runs. Encoder runs default to
mBERT; pass `MODEL=xlmr` for XLM-R.

**Length is the axis (context length controls retrieval).**

```bash
MODEL=mbert uv run python experiments/length_curve.py            # results/length_mbert.json
uv run python experiments/length_confound_control.py            # length vs distinctiveness control
```

**Hubness is the bottleneck.**

```bash
MODEL=mbert uv run python experiments/layer_sweep.py            # accuracy vs in-degree skew, per layer
uv run python experiments/perword_logreg.py                    # which lexical factors predict retrieval
uv run python experiments/xnli_hubness.py                      # per-language single-token hubness
```

**The hubness is concentrated in a few directions, and ablating it is causal.**

```bash
MODEL=mbert uv run python experiments/hub_ablation.py          # ABTT-k vs random-k vs CSLS, per pair
MODEL=mbert uv run python experiments/amplification_grid.py    # amplify vs ablate dose-response
MODEL=mbert uv run python experiments/csls_lineage.py          # CSLS / inverted softmax touch the same subspace
MODEL=mbert uv run python experiments/whitening_stability.py   # subspace stability and ZCA whitening vs ABTT
uv run python experiments/interpretable_axis_control.py        # is the subspace the fertility/length axes?
uv run python experiments/E2_effective_rank.py                 # variance concentration and effective rank
uv run python experiments/E3_perword_mediation.py              # per-word indirect effect (covariation, not mediation)
```

**Transfer and controls (E4, E5).**

```bash
uv run python experiments/E4_subspace_transfer.py   # cross-pair + cross-split transfer, language-ID control, principal angles
uv run python experiments/E5_map_interaction.py     # does de-hubbing help on top of a learned Procrustes map?
```

**Generality: a decoder and a downstream task.**

The decoder experiments run Llama-3-8B through Apple MLX, so they need Apple
Silicon. Install the extra and build the activation cache first:

```bash
uv pip install -e '.[decoder]'
uv run python experiments/decoder_extract.py                   # caches Llama layer-16 activations
uv run python experiments/decoder_ablation.py                  # the ablation, replicated on a decoder
uv run python experiments/downstream_patch.py                  # few-shot translation with the subspace patched out
```

`downstream_patch.py` unit-normalizes activations before estimating the subspace;
on raw decoder activations the top components are the massive-norm outlier
dimensions, and ablating those collapses generation for unrelated reasons.

```bash
MODEL=mbert uv run python experiments/xnli_extract.py          # frozen cross-encoder XNLI features
uv run python experiments/xnli_analyze.py                      # XNLI transfer vs hubness, with causal ablation
```

`xnli_analyze.py` reads the caches written by `xnli_extract.py` and
`xnli_hubness.py`, so run those first.

## Results

Headline numbers from the paper and the command that reproduces each. Encoder rows are
layer 1, identity retrieval (raw cosine, no learned map). Each script's own defaults are the
ones that produced the reported run, except where a variable is given in the command; the
per-script settings are in each file's docstring header. Confidence intervals are Student-t
over seeds.

| Paper claim | Number | Command |
|---|---|---|
| Sentence retrieval is near-perfect, word retrieval is not | `0.96`–`1.00` vs `0.2`–`0.6` P@1 | `MODEL=mbert python experiments/length_curve.py` |
| Length, not distinctiveness, drives the curve | random-bag stays near `0` | `python experiments/length_confound_control.py` |
| Single-token hubness is severe (mBERT en-es) | top hub = NN of `134` queries, `82%` never retrieved | `MODEL=mbert python experiments/hub_ablation.py` |
| Ambiguity predicts failure, not rarity | polysemy is the largest standardised coefficient with all five factors fit jointly | `python experiments/perword_logreg.py` |
| Accuracy and skew move together across depth | mBERT BLI peaks at layer 4 | `MODEL=mbert python experiments/layer_sweep.py` |
| Ablating the hub subspace recovers anti-hubs (mBERT) | `0.07`–`0.12` → `0.14`–`0.19`, random stays at baseline | `MODEL=mbert python experiments/hub_ablation.py` |
| Same on XLM-R, where it beats CSLS | `0.05`–`0.09` → `0.18`–`0.22` vs CSLS `0.11`–`0.13` | `MODEL=xlmr python experiments/hub_ablation.py` |
| Amplifying the subspace collapses retrieval | mBERT en-es P@1 `0.61` → `0.21` at alpha=8 | `MODEL=mbert python experiments/amplification_grid.py` |
| The subspace is concentrated but not low-rank | top-10 hold `14%` (mBERT) / `23%` (XLM-R) of variance | `python experiments/E2_effective_rank.py` |
| It transfers across pairs and word splits | `113%` (mBERT) / `93%` (XLM-R) of pair-native gain | `python experiments/E4_subspace_transfer.py` |
| It is not the language-identity subspace | held-out language probe `0.38`–`0.46`, near chance | `python experiments/E4_subspace_transfer.py` |
| De-hubbing composes with a learned Procrustes map | mBERT `+0.010`–`+0.016` (4 pairs); XLM-R `+0.07`–`+0.09` (5 pairs) | `python experiments/E5_map_interaction.py` |
| CSLS and inverted softmax flatten the same distribution | skew `1.7`–`2.5` → `0.9`–`1.4` under CSLS | `MODEL=mbert python experiments/csls_lineage.py` |
| Whitening corroborates the localization | ZCA matches or beats ABTT | `MODEL=mbert python experiments/whitening_stability.py` |
| The subspace is present in a frozen decoder | `+0.09` nats, `8/8` seeds; top-1 unchanged `0.459` → `0.462` | `N_SEEDS=8 python experiments/downstream_patch.py` |
| Predicted sentence-level effect, one-sided | XLM-R `+0.016` (5/5 seeds), mBERT `-0.017` (0/5) | `python experiments/xnli_analyze.py` |

Most scripts write JSON to `results/` and print the reported fields on completion;
`length_confound_control.py` prints its table without writing a file. The decoder rows need Apple Silicon (MLX). The XNLI rows need
`xnli_extract.py` and `xnli_hubness.py` run first.

**Models.** No custom checkpoints are released because none are trained: everything runs on
public frozen models, `bert-base-multilingual-cased`, `xlm-roberta-base`, and
`Llama-3-8B` (4-bit via MLX), pulled on first use.

## Repo layout

```
src/
  core/centering.py     per-language / global mean-centering of hidden states
  data/muse.py          MUSE dictionary loading, gold map, train/test splits
  extract/              mBERT / XLM-R word hidden-state extraction
experiments/            one script per experiment (see above)
tests/                  unit tests for the library
data/                   MUSE dicts (auto-downloaded) and FLORES-200 (created on demand)
results/                experiment outputs (JSON, caches; created by the scripts)
site-docs/              paper landing page (figures, PDF, HTML)
```

## Citation

```bibtex
@misc{hubnessdial2026,
  title  = {Hubness, Not Misalignment: A Causal Dial for Cross-Lingual Word Retrieval in Multilingual Models},
  author = {Author list to be added on acceptance},
  year   = {2026},
  note   = {Under review},
}
```

## License

MIT. See [LICENSE](LICENSE).
