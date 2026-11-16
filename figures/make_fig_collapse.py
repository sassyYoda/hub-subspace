"""Regenerate the hubness-collapse figure (camera-ready Fig. 3, fig3_hub_collapse.pdf).

Sorted nearest-neighbor in-degree N1 on mBERT en-es FLORES (layer 8, masked-mean pooling) over
the fixed 712-item held-out gallery. Three curves:

  - word-level raw (k=1): first words of the parallel sentences. Peak 134.
  - word-level distinct types (k=1): repeated first-word strings collapsed to one entry per
    type on both sides, then re-retrieved. Peak 17, 35% never retrieved. The raw peak of 134
    equals the number of held-out sentences beginning with "the": identical tokens share an
    embedding and are forced onto a single target, so the raw spike reflects string frequency,
    not geometry. The distinct-type curve is the geometric single-token hubness reported in the
    paper (abstract, Sec. mechanism, App. length / mediation).
  - sentence-level (full): the whole parallel sentence. Peak ~2, almost nothing stranded.

Data provenance. The original plotting script was not kept in this repository (the "Strip repo"
commit). This recomputes the in-degree curves directly from FLORES + cached mBERT, with the same
fixed seed-0 held-out split as experiments/length_curve.py and the paper's dedup control, and
asserts the published peaks (raw 134, distinct-type 17, sentence small) before drawing.

Needs data/flores200/flores200_dataset/devtest and bert-base-multilingual-cased (cached).
Out: <OUT>/fig3_hub_collapse.pdf (+ .png); OUT defaults to this figures/ dir. Run from repo root:
     OUT=hubness_eacl2027_camera_ready/figures/hubness python3 figures/make_fig_collapse.py
Requires matplotlib, numpy, torch, transformers, scikit-learn.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", HERE)
ROOT = os.environ.get("FLORES_ROOT", "data/flores200/flores200_dataset/devtest")
HFID = "bert-base-multilingual-cased"
L, N = 8, 1012
CODE = {"en": "eng_Latn", "es": "spa_Latn"}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.minor.width": 0.4,
    "figure.dpi": 300,
})


def compute_indegree():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(HFID)
    model = AutoModel.from_pretrained(HFID).to(dev).eval()

    def load(lang):
        with open(f"{ROOT}/{CODE[lang]}.devtest") as f:
            return [l.strip() for l in f][:N]

    @torch.no_grad()
    def embed(sents):
        out = []
        for i in range(0, len(sents), 32):
            enc = tok(sents[i:i+32], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[L]
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(out, 0)

    first = lambda s: (s.split()[0] if s.split() else "")
    en_sent, es_sent = load("en"), load("es")
    en_w = [first(s) for s in en_sent]
    es_w = [first(s) for s in es_sent]

    def indeg(X, Y):
        X = X - X.mean(0); Y = Y - Y.mean(0)
        test = np.random.RandomState(0).permutation(len(X))[300:]
        pred = cosine_similarity(X[test], Y[test]).argmax(1)
        return np.bincount(pred, minlength=len(test)), test

    # word-level raw (k=1): first words, full held-out gallery
    n1_word, test = indeg(embed(en_w), embed(es_w))

    # word-level distinct types: one entry per first-word string on each side, within the split
    enT = [en_w[i] for i in test]; esT = [es_w[i] for i in test]
    Xw = embed(en_w); Yw = embed(es_w)
    Xw = Xw - Xw.mean(0); Yw = Yw - Yw.mean(0)
    Xt, Yt = Xw[test], Yw[test]
    def first_idx(strings):
        seen, idx = set(), []
        for j, w in enumerate(strings):
            if w not in seen:
                seen.add(w); idx.append(j)
        return np.array(idx)
    from sklearn.metrics.pairwise import cosine_similarity as cs
    qi, gi = first_idx(enT), first_idx(esT)
    predD = cs(Xt[qi], Yt[gi]).argmax(1)
    n1_dedup = np.bincount(predD, minlength=len(gi))

    # sentence-level (full sentence)
    n1_sent, _ = indeg(embed(en_sent), embed(es_sent))

    from collections import Counter
    the_count = Counter(enT).most_common(1)[0]
    return n1_word, n1_dedup, n1_sent, the_count


def main():
    n1_word, n1_dedup, n1_sent, the_count = compute_indegree()
    peak_raw = int(n1_word.max())
    peak_dedup = int(n1_dedup.max())
    peak_sent = int(n1_sent.max())
    never_raw = float((n1_word == 0).mean())
    never_dedup = float((n1_dedup == 0).mean())
    never_sent = float((n1_sent == 0).mean())
    print(f"raw peak={peak_raw} never={never_raw:.3f} (most common first word {the_count}); "
          f"dedup peak={peak_dedup} never={never_dedup:.3f}; "
          f"sentence peak={peak_sent} never={never_sent:.3f}")

    # Guard against silent drift from the published numbers.
    assert peak_raw == 134, f"raw peak {peak_raw} != 134"
    assert peak_raw == the_count[1], "raw peak should equal the most common first-word count"
    assert peak_dedup == 17, f"dedup peak {peak_dedup} != 17"
    assert round(never_dedup, 2) == 0.35, f"dedup never {never_dedup:.3f} != 0.35"
    assert peak_sent <= 3, f"sentence peak {peak_sent} unexpectedly large"

    def sorted_desc(n1):
        return np.sort(n1)[::-1]

    fig, ax = plt.subplots(figsize=(3.27, 2.4))
    C_RAW, C_DED, C_SENT = "#88969c", "#009e73", "#d55e00"
    ax.plot(sorted_desc(n1_word), color=C_RAW, lw=1.4, label="word-level, raw (k = 1)")
    ax.plot(sorted_desc(n1_dedup), color=C_DED, lw=1.6, label="word-level, distinct types")
    ax.plot(sorted_desc(n1_sent), color=C_SENT, lw=1.4, label="sentence-level (full)")

    ax.set_yscale("log")
    ax.set_ylim(0.8, 200)
    ax.set_xlim(-8, len(n1_word) + 5)
    ax.set_xlabel("gallery item (sorted by in-degree)")
    ax.set_ylabel(r"NN in-degree $N_1$")
    ax.set_xticks([0, 200, 400, 600])

    ax.annotate(f"raw peak {peak_raw}\n(= copies of {the_count[0]!r})", xy=(0, peak_raw),
                xytext=(70, 150), fontsize=6.5, color=C_RAW,
                arrowprops=dict(arrowstyle="-", color=C_RAW, lw=0.5))
    ax.annotate(f"distinct types: peak {peak_dedup}", xy=(0, peak_dedup),
                xytext=(120, 32), fontsize=6.5, color=C_DED,
                arrowprops=dict(arrowstyle="-", color=C_DED, lw=0.5))
    ax.annotate("peak 2, 0% lost", xy=(2, peak_sent), xytext=(150, 3.3),
                fontsize=6.5, color=C_SENT,
                arrowprops=dict(arrowstyle="-", color=C_SENT, lw=0.5))
    ax.text(len(n1_word) * 0.5, 1.33, f"{never_raw*100:.0f}% never retrieved (raw),"
            f" {never_dedup*100:.0f}% (distinct)", fontsize=6, color="#444",
            ha="center", va="bottom")

    ax.legend(fontsize=6.3, frameon=False, loc="upper right", handlelength=1.4,
              borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, "fig3_hub_collapse.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf[:-4] + ".png", bbox_inches="tight", dpi=300)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
