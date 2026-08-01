"""E2: effective-rank quantification of the hub subspace (host numpy on cached L1 activations).

Per encoder x pair (10 cells), on the concatenated per-language-centered, unit-normalized bilingual
activation matrix (the exact space the ABTT PCs are taken from), M=2000, 5 seeds averaged:
  - eigenspectrum lambda_i (= squared singular values of the centered matrix)
  - participation ratio  PR   = (sum lambda)^2 / sum lambda^2
  - effective rank       erank = exp(spectral entropy)  (Roy & Vetterli)
  - top-k variance share for k in {1,3,5,10}
  - concentration ratio  top-k share / (k/d)   (isotropic expectation = k/d)
Then correlate per-k cumulative variance share with the per-k anti-hub recovery from the k-sweep,
so the k-sweep becomes the quantitative bridge (recovery tracks captured hub variance).

Out: results/effective_rank.json (run from the repo root, like the other experiment scripts).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.muse import load_muse_dictionary
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor

PAIRS = ["en-es", "en-zh", "en-de", "en-fr", "en-it"]
M, N_SEEDS, LAYER = 2000, 5, 1
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

# published per-k anti-hub recovery from the appendix k-sweep table (k = 1,3,5,10)
KSWEEP = {
 "mbert": {"en-es": [.169,.175,.177,.187], "en-zh": [.134,.143,.142,.135], "en-de": [.126,.133,.137,.147],
           "en-fr": [.151,.158,.163,.174], "en-it": [.105,.124,.130,.137]},
 "xlmr":  {"en-es": [.127,.189,.196,.220], "en-zh": [.141,.174,.213,.222], "en-de": [.120,.158,.175,.210],
           "en-fr": [.119,.169,.187,.205], "en-it": [.092,.131,.151,.182]},
}
KS = [1, 3, 5, 10]

def spectrum(Zc):
    s = np.linalg.svd(Zc, compute_uv=False); lam = s**2
    p = lam/lam.sum()
    pr = (lam.sum()**2)/(lam**2).sum()
    erank = float(np.exp(-(p*np.log(p+1e-12)).sum()))
    cum = np.cumsum(lam)/lam.sum()
    return lam, float(pr), erank, cum

out = {"model": {}, "d": None}
cumvar_pts, rec_pts = [], []
for model, extname in [("mbert", "mbert"), ("xlmr", "xlmr_base")]:
    ext = build_extractor(extname)
    out["model"][model] = {}
    for pair in PAIRS:
        sl_, tl_ = pair.split("-"); allp = load_muse_dictionary(pair)
        ss = set(); tt = set(); rows = []
        for s, t in allp:
            if s in ss or t in tt: continue
            ss.add(s); tt.add(t); rows.append((s, t))
            if len(rows) >= 4000: break
        SRC = [s for s, _ in rows]; TGT = [t for _, t in rows]
        E = ext.extract(SRC, sl_, layers=[LAYER], context_pool=ctx, subword_agg="mean")[LAYER]
        F = ext.extract(TGT, tl_, layers=[LAYER], context_pool=ctx, subword_agg="mean")[LAYER]
        E = E - E.mean(0); F = F - F.mean(0); d = E.shape[1]; out["d"] = int(d)
        prs, eranks, shares = [], [], {k: [] for k in KS}
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed); idx = rng.choice(len(rows), min(M, len(rows)), replace=False)
            Z = np.vstack([unit(E[idx]), unit(F[idx])]).astype(np.float64); Zc = Z - Z.mean(0)
            lam, pr, erank, cum = spectrum(Zc)
            prs.append(pr); eranks.append(erank)
            for k in KS: shares[k].append(float(cum[k-1]))
        rec = {"pr": float(np.mean(prs)), "erank": float(np.mean(eranks)),
               "topk_share": {str(k): float(np.mean(shares[k])) for k in KS},
               "conc_ratio": {str(k): float(np.mean(shares[k])/(k/d)) for k in KS}}
        out["model"][model][pair] = rec
        for i, k in enumerate(KS):
            cumvar_pts.append(np.mean(shares[k])); rec_pts.append(KSWEEP[model][pair][i])
        print(f"[{model} {pair}] PR {rec['pr']:.1f}  erank {rec['erank']:.1f}  "
              f"top1 {rec['topk_share']['1']*100:.1f}%  top10 {rec['topk_share']['10']*100:.1f}%  "
              f"(x{rec['conc_ratio']['10']:.0f} isotropic)", flush=True)

r = float(np.corrcoef(cumvar_pts, rec_pts)[0, 1])
out["cumvar_recovery_pearson_r"] = r
# per-encoder means of erank/PR
for model in ["mbert", "xlmr"]:
    ers = [out["model"][model][p]["erank"] for p in PAIRS]; prs = [out["model"][model][p]["pr"] for p in PAIRS]
    out["model"][model]["_mean_erank"] = float(np.mean(ers)); out["model"][model]["_mean_pr"] = float(np.mean(prs))
    print(f"== {model}: mean effective rank {np.mean(ers):.1f}  mean PR {np.mean(prs):.1f} (d={out['d']})", flush=True)
print(f"== Pearson r(cumulative variance share, anti-hub recovery) over 40 (cell x k) points = {r:.3f}", flush=True)
os.makedirs("results", exist_ok=True)
with open("results/effective_rank.json", "w") as f:
    json.dump(out, f, indent=2)
print("E2_EFFECTIVE_RANK DONE", flush=True)
