# Figure generation

Plotting scripts for the paper's regenerable figures. The original figure scripts were not kept in
this repository, so these reconstruct the three figures that needed redrawing for the EACL 2027
camera-ready, at readable font sizes (serif/Times, Type-42 embedded fonts, >= 7 pt at print width).

Each script writes `<name>.pdf` and `.png` to the directory in the `OUT` environment variable
(default: this `figures/` dir). To refresh the paper figures, point `OUT` at the camera-ready figure
dir, e.g. `OUT=../hubness_eacl2027_camera_ready/figures/hubness python3 figures/make_fig_amplify.py`.

| Script | Figure | Data source |
|---|---|---|
| `make_fig_lineage.py` | Fig. 7, `fig_lineage.pdf` (retrieval-fix lineage: CSLS / inverted-softmax flatten in-degree skew) | self-contained; means from `tab:lineage`, 95% CIs recovered from the published PDF (see the script's provenance docstring) |
| `make_fig_downstream.py` | Fig. 8, `fig4_downstream_v5.pdf` (frozen-decoder mid-forward ablation, first-token metrics) | `results/downstream_patch.json` (N=150, 8-seed run) |
| `make_fig_amplify.py` | Fig. 10, `fig_amplify.pdf` (bidirectional dial: ablate / amplify P@1 and max in-degree) | `paper_acl/EXPERIMENT_amplify_RESULTS.md` (20-seed grid), cross-checked against `tab:amplify` |

Notes:
- Each script re-derives its numbers from the recorded data and (where applicable) asserts agreement
  with the paper's tables before drawing, so the figures cannot silently drift from the reported
  values.
- Two of the data sources live under paths that are not tracked in this repository (`results/`,
  `paper_acl/`); the scripts run on a working copy that has them.
- Requires `matplotlib`, `numpy`, `scipy`.
