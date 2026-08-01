"""De-hubbing ablation across language pairs and encoders.

Per (model, pair): identity cosine retrieval in centered layer-1 space, comparing baseline,
CSLS, ABTT-k{1,3,5,10}, and random-k{3,10} controls. Metrics are P@1, max in-degree, skew of
the in-degree distribution, fraction of words never retrieved, and anti-hub recovery, each with
multi-seed 95% confidence intervals.

ABTT (all-but-the-top) projects out the top-k principal components; the random-k conditions
project out k random directions as a control for that ablation. CSLS is the retrieval-time
correction of Lample et al. (2018).

Env: MODEL=mbert|xlmr, PAIRS=en-es,en-zh,en-de,en-fr,en-it, M=2000, N_SEEDS=5, LAYER=1
Out: results/hub_ablation_{MODEL}.json

Run: python experiments/hub_ablation.py
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
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
LAYER = int(os.environ.get("LAYER", "1"))
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext = build_extractor(EXTNAME)
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

def retrieve(X, Y, gold, src, gal, csls=False):
    S = cosine_similarity(X, Y)
    if csls:
        k = 10
        rT = np.partition(S, -k, axis=1)[:, -k:].mean(1, keepdims=True)
        rS = np.partition(S, -k, axis=0)[-k:, :].mean(0, keepdims=True)
        S = 2*S - rT - rS
    pred = S.argmax(1)
    corr = np.array([1 if gal[pred[j]] in gold.get(src[j], set()) else 0 for j in range(len(src))])
    N1 = np.bincount(pred, minlength=len(gal))
    return float(corr.mean()), pred, N1, corr

def project_out(A, B, dirs):
    if dirs is None or len(dirs) == 0:
        return A, B
    Q = np.linalg.qr(dirs.T)[0].T
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q

def topk_pca(X, Y, k):
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k]

out = {"model": MODEL, "layer": LAYER, "results": {}}
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
    conds = ["baseline", "CSLS", "ABTT-k1", "ABTT-k3", "ABTT-k5", "ABTT-k10", "random-k3", "random-k10"]
    agg = {k: {"p1": [], "maxhub": [], "skew": [], "never": [], "antihub_rec": []} for k in conds}
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
        def metrics(Xa, Ya, csls=False):
            p1, pred, N1, corr = retrieve(unit(Xa), unit(Ya), gold, src, gal, csls=csls)
            rec = float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
            return p1, int(N1.max()), float(skew(N1)), float((N1 == 0).mean()), rec
        def run(name, Xa, Ya, csls=False):
            for key, val in zip(["p1", "maxhub", "skew", "never", "antihub_rec"], metrics(Xa, Ya, csls=csls)):
                agg[name][key].append(val)
        run("baseline", X, Y)
        run("CSLS", X, Y, csls=True)
        for k in [1, 3, 5, 10]:
            Xa, Ya = project_out(X, Y, topk_pca(X, Y, k))
            run(f"ABTT-k{k}", Xa, Ya)
        for k in [3, 10]:
            Xa, Ya = project_out(X, Y, rng.randn(k, X.shape[1]))
            run(f"random-k{k}", Xa, Ya)
        hb(f"  seed {seed} done (n_antihub_q={len(anti_q)})")
    def ci(v):
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]
    out["results"][pair] = {cd: {k: ci(agg[cd][k]) for k in agg[cd]} for cd in conds}
    r = out["results"][pair]
    print(f"[{pair}] {MODEL} L{LAYER}: base antihub {r['baseline']['antihub_rec'][0]:.3f} -> "
          f"ABTT10 {r['ABTT-k10']['antihub_rec'][0]:.3f} (rand {r['random-k10']['antihub_rec'][0]:.3f}); "
          f"CSLS {r['CSLS']['antihub_rec'][0]:.3f}", flush=True)

os.makedirs("results", exist_ok=True)
with open(f"results/hub_ablation_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"ABLATION_{MODEL.upper()} DONE", flush=True)
