"""Regenerate the hubness-lineage figure (camera-ready Fig. 7, fig_lineage.pdf) at readable font sizes.

Panel (a): in-degree skewness (log) under cosine / CSLS / inverted softmax for six model x pair cells.
Panel (b): skewness as a fraction of the cosine baseline under CSLS / ABTT / ABTT+CSLS for
mBERT en-es and XLM-R en-zh.

Data provenance. The original plotting script and results/csls_lineage_{mbert,xlmr}.json were deleted
in the "Strip repo" commit. The 5-seed means are in the paper (appendix.tex, tab:lineage, "skew
cos/CSLS/ISF" and "skew b/C/A/A+C" columns). The 95% CIs are not in the paper, so they were recovered
from the published PDF: the content stream of fig_lineage.pdf was decompressed with pypdf and the bar
tops and error-bar endpoints read off in PDF user units, then inverted through the axis mapping
solved from the tick marks (panel a: y=211.022 <-> 10^0, y=284.124 <-> 10^1; panel b: y=28.696 <-> 0,
y=129.601 <-> 1). The decoded means agree with tab:lineage to the printed 2 decimals; the script
asserts this before drawing.

Out: figs_out/fig_lineage.pdf (+ .png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", HERE)

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

# ---------------------------------------------------------------- published means (tab:lineage)
CELLS = ["mB\nes", "mB\nzh", "mB\nde", "XR\nes", "XR\nzh", "XR\nde"]
TAB_SKEW = {  # skew cos / CSLS / ISF
    "cosine":      [1.74, 2.46, 2.44, 9.68, 15.76, 15.32],
    "CSLS":        [0.92, 0.97, 1.40, 7.53, 9.36, 9.23],
    "inv-softmax": [2.45, 0.92, 2.17, 2.14, 1.92, 3.71],
}
TAB_REDUN = {  # skew b / C / A / A+C
    "mBERT en-es": [1.74, 0.92, 1.19, 0.76],
    "XLM-R en-zh": [15.76, 9.36, 5.47, 4.54],
}

# ---------------------------------------------------------------- geometry recovered from the PDF
# panel (a): bar top y and error-bar (lo, hi) y, in PDF units, per method, cells in CELLS order
A_TOP = {
    "cosine":      [228.637, 239.573, 239.384, 283.104, 298.561, 297.677],
    "CSLS":        [208.283, 209.924, 221.745, 275.105, 282.011, 281.566],
    "inv-softmax": [239.439, 208.312, 235.588, 235.151, 231.776, 252.612],
}
A_ERR = {
    "cosine":      [(222.881, 233.507), (237.520, 241.502), (238.286, 240.445),
                    (277.650, 287.757), (289.387, 305.670), (272.878, 311.428)],
    "CSLS":        [(207.354, 209.186), (208.340, 211.432), (220.548, 222.898),
                    (264.519, 283.029), (272.736, 289.181), (268.453, 290.819)],
    "inv-softmax": [(236.540, 242.095), (203.184, 212.726), (233.024, 237.960),
                    (229.764, 239.756), (225.649, 236.910), (233.860, 264.321)],
}
A_Y1, A_Y10 = 211.022, 284.124                     # tick marks for 10^0 and 10^1

def a_val(y):
    return 10 ** ((y - A_Y1) / (A_Y10 - A_Y1))

# panel (b): bar top y and error-bar (lo, hi) y for CSLS / ABTT / ABTT+CSLS
B_TOP = {"mBERT en-es": [81.844, 97.574, 72.440], "XLM-R en-zh": [88.608, 63.713, 57.745]}
B_ERR = {"mBERT en-es": [(80.310, 83.377), (86.977, 108.171), (69.882, 74.999)],
         "XLM-R en-zh": [(73.430, 103.787), (60.695, 66.730), (51.555, 63.934)]}
B_Y0, B_Y1 = 28.696, 129.601

def b_val(y):
    return (y - B_Y0) / (B_Y1 - B_Y0)

# ---------------------------------------------------------------- decode + verify against the paper
skew = {m: np.array([a_val(y) for y in A_TOP[m]]) for m in A_TOP}
skew_ci = {m: np.array([(a_val(hi) - a_val(lo)) / 2 for lo, hi in A_ERR[m]]) for m in A_ERR}
ratio = {c: np.array([b_val(y) for y in B_TOP[c]]) for c in B_TOP}
ratio_ci = {c: np.array([(b_val(hi) - b_val(lo)) / 2 for lo, hi in B_ERR[c]]) for c in B_ERR}

for m in TAB_SKEW:
    d = np.abs(skew[m] - np.array(TAB_SKEW[m]))
    assert d.max() < 0.011, (m, skew[m], TAB_SKEW[m])
for c in TAB_REDUN:
    b, C, A, AC = TAB_REDUN[c]
    tab_ratio = np.array([C, A, AC]) / b
    d = np.abs(ratio[c] - tab_ratio)
    assert d.max() < 0.012, (c, ratio[c], tab_ratio)   # table values are 2-decimal rounded
print("decoded means match tab:lineage")
for m in skew:
    print(f"  {m:12s} " + "  ".join(f"{v:5.2f}+-{e:4.2f}" for v, e in zip(skew[m], skew_ci[m])))
for c in ratio:
    print(f"  {c:12s} " + "  ".join(f"{v:5.3f}+-{e:5.3f}" for v, e in zip(ratio[c], ratio_ci[c])))

# ---------------------------------------------------------------- draw at single-column width
W = 3.03                                   # ACL column width (7.7 cm)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 3.85), gridspec_kw={"hspace": 0.48})
fig.subplots_adjust(left=0.17, right=0.985, top=0.95, bottom=0.075)

GREY, BLUE, ORANGE, VERM = "#8c8c8c", "#0072B2", "#E69F00", "#D55E00"
EB = dict(ecolor="black", elinewidth=0.9, capsize=1.8, capthick=0.9)
FS_TICK, FS_LAB, FS_LEG = 7.5, 8.5, 7.2

# (a)
x = np.arange(6)
w = 0.26
for off, m, col, hatch in [(-w, "cosine", GREY, None), (0, "CSLS", BLUE, None),
                           (w, "inv-softmax", ORANGE, "///")]:
    ax1.bar(x + off, skew[m], w, yerr=skew_ci[m], color=col, hatch=hatch,
            edgecolor="white" if hatch else col, linewidth=0.5, label=m, zorder=3,
            error_kw=EB)
ax1.set_yscale("log")
ax1.set_ylim(0.6, 30)
ax1.set_yticks([1, 10])
ax1.set_yticklabels([r"$10^0$", r"$10^1$"], fontsize=FS_TICK)
ax1.tick_params(axis="y", which="minor", length=1.8)
ax1.set_xticks(x)
ax1.set_xticklabels(CELLS, fontsize=FS_TICK)
ax1.tick_params(axis="x", length=2.5, pad=2)
ax1.set_ylabel("in-degree skewness (log)", fontsize=FS_LAB)
ax1.yaxis.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=FS_LEG, loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cccccc",
           handlelength=1.1, handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
ax1.text(-0.21, 1.03, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="bottom")

# (b)
x2 = np.arange(3)
w2 = 0.34
for off, c, col in [(-w2 / 2, "mBERT en-es", BLUE), (w2 / 2, "XLM-R en-zh", VERM)]:
    ax2.bar(x2 + off, ratio[c], w2, yerr=ratio_ci[c], color=col, label=c, zorder=3, error_kw=EB)
ax2.axhline(1.0, color="#999999", lw=0.8, ls="--", zorder=1)
ax2.text(-0.56, 1.02, "cosine baseline", fontsize=7, color="#777777", ha="left", va="bottom")
ax2.set_ylim(0, 1.22)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=FS_TICK)
ax2.set_xticks(x2)
ax2.set_xticklabels(["CSLS", "ABTT", "ABTT+CSLS"], fontsize=FS_TICK)
ax2.tick_params(axis="x", length=2.5, pad=2)
ax2.set_xlim(-0.6, 3.0)
ax2.set_ylabel("skewness / cosine baseline", fontsize=FS_LAB)
ax2.yaxis.grid(True, color="#dddddd", lw=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(fontsize=FS_LEG, loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=True,
           framealpha=0.95, edgecolor="#cccccc", handlelength=1.1, handletextpad=0.5,
           borderpad=0.4, labelspacing=0.3)
ax2.text(-0.21, 1.03, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="bottom")

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_lineage.{ext}"))
print("wrote", os.path.join(OUT, "fig_lineage.{pdf,png}"))
