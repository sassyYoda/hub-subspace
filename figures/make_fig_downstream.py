"""Regenerate the decoder-patch figure (camera-ready Fig. 8, fig4_downstream_v5.pdf) at readable sizes.

Two panels, vertically stacked so each spans the full single column: (a) per-condition mean gold
FIRST-TOKEN log-prob (moves up under hub-ablation), (b) first-token top-1 accuracy (flat). Honest to
the E1 result: both panels score the first token of the gold Spanish target, not the whole word; the
full-word metrics (seq-logp, exact match) reverse sign and are reported in the appendix text. Error
bars are 95% CIs from the Student-t quantile at n-1 df, the multi-seed convention used elsewhere.

Data: results/downstream_patch.json (the N=150, 8-seed, unit-normalized run). That file lives under
the gitignored results/ dir; run this on a checkout that has it. Serif/Times, Type-42 fonts.

Out: <OUT>/fig4_downstream_v5.{pdf,png}  (OUT defaults to this script's dir; set OUT to write
elsewhere, e.g. the camera-ready figures/hubness/).
"""
import os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("OUT", os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(REPO, "results", "downstream_patch.json")))
conds = ["baseline", "hub", "random"]
labels = {"baseline": "baseline", "hub": "hub-ablated", "random": "random"}
COL = {"baseline": "#333333", "hub": "#0072B2", "random": "#8C8C8C"}

def ci(v):
    v = np.array(v, float); n = len(v)
    return v.mean(), stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.15, 3.9))
xs = np.arange(3)
for ax, key, ylab, ttl in [
        (ax1, "lp", "gold first-token\nlog-prob", "(a) gold first-token log-probability"),
        (ax2, "acc", "first-token\ntop-1 accuracy", "(b) first-token top-1 accuracy")]:
    means = [ci(d[c][key])[0] for c in conds]; errs = [ci(d[c][key])[1] for c in conds]
    ax.bar(xs, means, yerr=errs, color=[COL[c] for c in conds], width=0.6, capsize=3,
           error_kw={"lw": 1.0})
    ax.set_xticks(xs); ax.set_xticklabels([labels[c] for c in conds], fontsize=8)
    ax.set_ylabel(ylab, fontsize=8); ax.set_title(ttl, fontsize=8.5, loc="left")
    ax.tick_params(axis="y", labelsize=7.5)
lp_means = [ci(d[c]["lp"])[0] for c in conds]
ax1.set_ylim(min(lp_means) * 1.12, 0)
ac_means = [ci(d[c]["acc"])[0] for c in conds]
ax2.set_ylim(0, max(ac_means) * 1.18)
hp = np.array(d["hub"]["lp"]) - np.array(d["random"]["lp"])
da = np.array(d["hub"]["acc"]) - np.array(d["random"]["acc"])
fig.tight_layout(h_pad=1.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig4_downstream_v5.{ext}"),
                bbox_inches="tight", dpi=(160 if ext == "png" else None))
print("top-1 means:", {c: round(ci(d[c]['acc'])[0], 3) for c in conds})
print("paired lp hub-random: +%.3f nats, %d/%d" % (hp.mean(), (hp > 0).sum(), len(hp)))
print("paired acc hub-random: +%.3f, %d/%d" % (da.mean(), (da > 0).sum(), len(da)))
print("wrote fig4_downstream_v5.{pdf,png} to", OUT)
