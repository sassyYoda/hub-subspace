"""Regenerate the bidirectional-dial figure (camera-ready Fig. 10, fig_amplify.pdf) at readable font sizes.

2 x 2 grid: columns mBERT / XLM-R, rows P@1 / max in-degree. Per panel: faint per-pair curves over
alpha in {1, 1.5, 2, 3, 4, 6, 8}, bold five-pair mean with markers, dashed baseline mean, hub-ablate
(ABTT) triangles at the "ablate" position (faint per pair, bold mean), random-amplify x4 open circles
at alpha=4 (faint per pair, bold mean).

Data provenance. The original plotting script and results/amplification_grid_{mbert,xlmr}.json were
deleted in the "Strip repo" commit. Every number is parsed from the full 20-seed grid log that
survives in the repo, paper_acl/EXPERIMENT_amplify_RESULTS.md (10 cells x 10 conditions, mean +- CI),
and cross-checked against the paper's tab:amplify in appendix.tex before drawing (P@1 ABTT / base /
amp x8 / rnd x4 and max N1 base -> amp x8 for all ten cells).

Out: figs_out/fig_amplify.pdf (+ .png)
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", HERE)
ROOT = os.environ.get("ROOT", os.path.dirname(HERE))  # figures/ -> repo root; override with ROOT=
MD = os.path.join(ROOT, "paper_acl", "EXPERIMENT_amplify_RESULTS.md")
TEX = os.path.join(ROOT, "hubness_eacl2027_camera_ready", "appendix.tex")

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300,
})

ALPHAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
PAIRS = ["en-es", "en-zh", "en-de", "en-fr", "en-it"]
MODELS = [("mbert", "mBERT"), ("xlmr", "XLM-R")]

# ---------------------------------------------------------------- parse the grid log
def parse_md(path):
    data = {}                                       # data[model][pair][cond] = {metric: (mean, ci)}
    cur = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"### (mbert|xlmr) (en-\w\w)", line)
        if m:
            cur = (m.group(1), m.group(2))
            data.setdefault(cur[0], {})[cur[1]] = {}
            continue
        if cur and line.startswith("|") and "±" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            cond = re.sub(r"\s*\(.*\)$", "", cells[0])
            vals = [tuple(float(v) for v in c.split("±")) for c in cells[1:6]]
            data[cur[0]][cur[1]][cond] = dict(zip(["p1", "antihub", "maxhub", "never", "skew"], vals))
    return data

D = parse_md(MD)
for mk, _ in MODELS:
    assert set(D[mk]) == set(PAIRS), (mk, D[mk].keys())
    for p in PAIRS:
        assert len(D[mk][p]) == 10, (mk, p, D[mk][p].keys())

def series(mk, p, metric):
    d = D[mk][p]
    return np.array([d["baseline"][metric][0]] +
                    [d[f"amplify x{a:g}"][metric][0] for a in ALPHAS[1:]])

# ---------------------------------------------------------------- cross-check against tab:amplify
tex = open(TEX, encoding="utf-8").read()
tab = tex[tex.index(r"\label{tab:amplify}") - 2500: tex.index(r"\label{tab:amplify}")]
rows = re.findall(r"&\s*en--(\w\w)\s*&\s*\$([\d.]+)\$\s*&\s*\$([\d.]+)\$\s*&\s*\$\\mathbf\{([\d.]+)\}\$"
                  r"\s*&\s*\$([\d.]+)\$\s*&\s*\$(\d+)\\!\\to\\!(\d+)\$", tab)
assert len(rows) == 10, len(rows)
bad = []
for i, (tl, abtt, base, amp8, rnd4, mh_b, mh_a) in enumerate(rows):
    mk = "mbert" if i < 5 else "xlmr"
    d = D[mk][f"en-{tl}"]
    got = (d["ABTT-k10"]["p1"][0], d["baseline"]["p1"][0], d["amplify x8"]["p1"][0],
           d["rand-amplify x4"]["p1"][0], round(d["baseline"]["maxhub"][0]), round(d["amplify x8"]["maxhub"][0]))
    exp = (float(abtt), float(base), float(amp8), float(rnd4), int(mh_b), int(mh_a))
    for g, e in zip(got, exp):
        if abs(g - e) > 1e-6 + (1.0 if isinstance(e, int) else 0.0005):
            bad.append((mk, tl, got, exp))
            break
assert not bad, bad
print("grid log matches tab:amplify on all 10 cells")

# ---------------------------------------------------------------- draw
W = 0.8 * 6.2992                      # 0.8\textwidth in the ACL two-column layout (16 cm text width)
fig, axes = plt.subplots(2, 2, figsize=(W, 3.9), sharex="col")
fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.115, wspace=0.28, hspace=0.12)

BLUE, FAINT, VERM = "#0072B2", "#9ecae1", "#D55E00"
X_ABL = 0.56                            # where the "ablate" column sits on the log-2 x axis
FS_TICK, FS_LAB, FS_TTL, FS_LEG = 7.5, 8.5, 9, 7.5

for j, (mk, mname) in enumerate(MODELS):
    for i, (metric, ylab) in enumerate([("p1", "P@1"), ("maxhub", "max in-degree")]):
        ax = axes[i, j]
        per = np.array([series(mk, p, metric) for p in PAIRS])       # 5 x 7
        mean = per.mean(0)
        base_mean = mean[0]
        abl = np.array([D[mk][p]["ABTT-k10"][metric][0] for p in PAIRS])
        rnd = np.array([D[mk][p]["rand-amplify x4"][metric][0] for p in PAIRS])

        ax.axvline(1.0, color="#bbbbbb", lw=0.7, zorder=1)
        ax.axhline(base_mean, color="#555555", lw=0.8, ls="--", zorder=1)
        for row in per:
            ax.plot(ALPHAS, row, color=FAINT, lw=0.8, zorder=2)
        ax.plot(ALPHAS, mean, color=BLUE, lw=1.8, marker="o", ms=3.6, zorder=4,
                label="hub-amplify (mean)")
        ax.scatter(np.full(5, X_ABL), abl, marker="v", s=12, color=FAINT, zorder=3)
        ax.scatter([X_ABL], [abl.mean()], marker="v", s=55, color=BLUE, zorder=5,
                   label="hub-ablate (ABTT)")
        ax.scatter(np.full(5, 4.0), rnd, s=10, facecolors="none", edgecolors=VERM,
                   linewidths=0.7, alpha=0.6, zorder=3)
        ax.scatter([4.0], [rnd.mean()], s=70, facecolors="none", edgecolors=VERM,
                   linewidths=1.6, zorder=5, label=r"random-amplify $\times$4")

        ax.set_xscale("log", base=2)
        ax.set_xlim(0.44, 9.2)
        ax.set_xticks([X_ABL, 1, 2, 4, 8])
        ax.set_xticklabels(["ablate", "1", "2", "4", "8"], fontsize=FS_TICK)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(axis="both", length=2.5, pad=2, labelsize=FS_TICK)
        ax.set_ylabel(ylab, fontsize=FS_LAB)
        ax.yaxis.grid(True, color="#e5e5e5", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_title(mname, fontsize=FS_TTL, pad=4)
        if i == 1:
            ax.set_xlabel(r"hub-subspace scale $\alpha$  (1 = baseline)", fontsize=FS_LAB)
        if metric == "p1":
            lo = 0.0 if mk == "mbert" else 0.0
            ax.set_ylim(lo, max(0.66, per.max() + 0.04))
        else:
            ax.set_ylim(0, per.max() * 1.08)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=FS_LEG, frameon=True, edgecolor="#cccccc",
           bbox_to_anchor=(0.53, 0.985), handletextpad=0.5, columnspacing=1.4)

# print the plotted means so they can be eyeballed against the paper
for mk, mname in MODELS:
    for metric in ("p1", "maxhub"):
        per = np.array([series(mk, p, metric) for p in PAIRS])
        abl = np.mean([D[mk][p]["ABTT-k10"][metric][0] for p in PAIRS])
        rnd = np.mean([D[mk][p]["rand-amplify x4"][metric][0] for p in PAIRS])
        print(f"{mname:6s} {metric:7s} ABTT={abl:.3f} " +
              " ".join(f"a{a:g}={v:.3f}" for a, v in zip(ALPHAS, per.mean(0))) + f" rnd4={rnd:.3f}")

os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_amplify.{ext}"))
print("wrote", os.path.join(OUT, "fig_amplify.{pdf,png}"))
