"""Dose-response grid: amplify vs ablate the hub subspace.

Sweeps a single dial over the top-k subspace and reads every condition off the same cached
activations per seed, which keeps the comparison deterministic and reproducible. The extractor,
MUSE pairs, centering, retrieval, and confidence intervals match hub_ablation.py, so the
conditions line up with the ablation results there.

The dial runs from random-k{K} (down-side control) through ABTT-k{K} (ablate) and baseline
up to amplify by {1.5, 2, 3, 4, 6, 8}, with rand-amplify-k{K}-a4 as the up-side specificity
control. The amplify map is a -> a + (alpha-1)(a Q^T) Q, with Q an orthonormal basis of the
top-k subspace, followed by renormalization. alpha=1 is the baseline; alpha=0 reduces to
projecting the subspace out (ABTT).

Env: MODEL=mbert|xlmr, PAIRS=en-es,..., M=2000, N_SEEDS=20, LAYER=1, K=10
Out: results/amplification_grid_{MODEL}.json, plus results/grid_{MODEL}_{pair}.json per pair

Run: python experiments/amplification_grid.py
"""
from __future__ import annotations
import os, json, time
import numpy as np
from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from scipy.stats import t as _student_t

MODEL = os.environ.get("MODEL", "mbert")
EXTNAME = {"mbert": "mbert", "xlmr": "xlmr_base"}[MODEL]
PAIRS = os.environ.get("PAIRS", "en-es,en-zh,en-de,en-fr,en-it").split(",")
M = int(os.environ.get("M", "2000"))
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
LAYER = int(os.environ.get("LAYER", "1"))
K = int(os.environ.get("K", "10"))
ALPHAS = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext = build_extractor(EXTNAME)
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

def retrieve(X, Y, gold, src, gal):
    S = cosine_similarity(X, Y)
    pred = S.argmax(1)
    corr = np.array([1 if gal[pred[j]] in gold.get(src[j], set()) else 0 for j in range(len(src))])
    N1 = np.bincount(pred, minlength=len(gal))
    return float(corr.mean()), pred, N1, corr

def ortho(dirs):
    return np.linalg.qr(dirs.T)[0].T

def project_out(A, B, dirs):
    Q = ortho(dirs)
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q

def amplify(A, B, dirs, alpha):
    Q = ortho(dirs)
    return A+(alpha-1.0)*(A@Q.T)@Q, B+(alpha-1.0)*(B@Q.T)@Q

def topk_pca(X, Y, k):
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k]

CONDS = ([f"random-k{K}", f"ABTT-k{K}", "baseline"]
         + [f"amplify-k{K}-a{a:g}" for a in ALPHAS]
         + [f"rand-amplify-k{K}-a4"])
out = {"model": MODEL, "layer": LAYER, "K": K, "alphas": ALPHAS, "n_seeds": N_SEEDS, "M": M, "results": {}}
for pair in PAIRS:
    sl_, tl_ = pair.split("-")
    allp = load_muse_dictionary(pair)
    gold = build_gold_map(allp)
    # Keep one entry per source and per target word, capped at 4000.
    ss = set()
    tt = set()
    rows = []
    for s, t in allp:
        if s in ss or t in tt:
            continue
        ss.add(s)
        tt.add(t)
        rows.append((s, t))
        if len(rows) >= 4000:
            break
    SRC = [s for s, _ in rows]
    TGT = [t for _, t in rows]
    E = ext.extract(SRC, sl_, layers=[LAYER], context_pool=ctx, subword_agg="mean")[LAYER]
    F = ext.extract(TGT, tl_, layers=[LAYER], context_pool=ctx, subword_agg="mean")[LAYER]
    E = E - E.mean(0)
    F = F - F.mean(0)
    hb(f"[{pair}] extracted {len(rows)}")
    agg = {k: {"p1": [], "maxhub": [], "skew": [], "never": [], "antihub_rec": []} for k in CONDS}
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), min(M, len(rows)), replace=False)
        src = [SRC[i] for i in idx]
        gal = [TGT[i] for i in idx]
        X = unit(E[idx]).astype(np.float64)
        Y = unit(F[idx]).astype(np.float64)
        _, _, N1b, _ = retrieve(X, Y, gold, src, gal)
        gpos = {w: j for j, w in enumerate(gal)}
        # anti_q: queries whose gold target is an anti-hub at baseline (never retrieved by anyone).
        anti_q = [q for q in range(len(src)) if any(gpos.get(t, -1) >= 0 and N1b[gpos[t]] == 0 for t in gold.get(src[q], set()))]
        def metrics(Xa, Ya):
            p1, pred, N1, corr = retrieve(unit(Xa), unit(Ya), gold, src, gal)
            rec = float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
            return p1, int(N1.max()), float(skew(N1)), float((N1 == 0).mean()), rec
        def run(name, Xa, Ya):
            for key, val in zip(["p1", "maxhub", "skew", "never", "antihub_rec"], metrics(Xa, Ya)):
                agg[name][key].append(val)
        pcs = topk_pca(X, Y, K)
        Xa, Ya = project_out(X, Y, rng.randn(K, X.shape[1]))
        run(f"random-k{K}", Xa, Ya)
        Xa, Ya = project_out(X, Y, pcs)
        run(f"ABTT-k{K}", Xa, Ya)
        run("baseline", X, Y)
        for a in ALPHAS:
            Xa, Ya = amplify(X, Y, pcs, a)
            run(f"amplify-k{K}-a{a:g}", Xa, Ya)
        Xa, Ya = amplify(X, Y, rng.randn(K, X.shape[1]), 4.0)
        run(f"rand-amplify-k{K}-a4", Xa, Ya)
        if seed % 5 == 0:
            hb(f"  seed {seed} (n_antihub_q={len(anti_q)})")
    def ci(v):
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]
    out["results"][pair] = {cd: {k: ci(agg[cd][k]) for k in agg[cd]} for cd in CONDS}
    os.makedirs("results", exist_ok=True)
    with open(f"results/grid_{MODEL}_{pair}.json", "w") as f:
        json.dump({"model": MODEL, "layer": LAYER, "K": K, "alphas": ALPHAS,
                   "n_seeds": N_SEEDS, "M": M, "pair": pair,
                   "result": out["results"][pair]}, f, indent=2)
    r = out["results"][pair]
    print(f"\n[{pair}] {MODEL} L{LAYER} K{K} ({N_SEEDS} seeds): P@1 / anti-hub rec / max-in-deg", flush=True)
    for cd in CONDS:
        print(f"  {cd:22s} p1={r[cd]['p1'][0]:.3f}  rec={r[cd]['antihub_rec'][0]:.3f}  "
              f"maxhub={r[cd]['maxhub'][0]:.1f}  never={r[cd]['never'][0]:.3f}", flush=True)

os.makedirs("results", exist_ok=True)
with open(f"results/amplification_grid_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nAMPLIFY_GRID_{MODEL.upper()} DONE", flush=True)
