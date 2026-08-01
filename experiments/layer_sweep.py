"""Layer-wise sweep of hubness and bilingual retrieval accuracy.

For each (model, pair) it extracts every layer (word isolation, subword-mean pooling),
mean-centers per layer per language, then at each layer measures word-level BLI P@1
(identity cosine nearest neighbor) and hubness (in-degree skew, max in-degree, fraction
of words never retrieved). Confidence intervals come from multi-seed subsampling. Accuracy
and the inverse of hubness tend to track each other across depth and peak at the same
early layers.

Env: MODEL=mbert|xlmr, PAIRS=en-es,en-zh, M=1500, N_SEEDS=3
Out: results/layer_sweep_{MODEL}.json

Run: python experiments/layer_sweep.py
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
PAIRS = os.environ.get("PAIRS", "en-es,en-zh").split(",")
M = int(os.environ.get("M", "1500"))
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext = build_extractor(EXTNAME)
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

def retrieve(X, Y, gold, src, gal):
    S = cosine_similarity(X, Y)
    pred = S.argmax(1)
    corr = np.array([1 if gal[pred[j]] in gold.get(src[j], set()) else 0 for j in range(len(src))])
    N1 = np.bincount(pred, minlength=len(gal))  # in-degree: how often each target is retrieved
    return float(corr.mean()), N1

out = {"model": MODEL, "pairs": {}}
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
    Eall = ext.extract(SRC, sl_, layers="all", context_pool=ctx, subword_agg="mean")
    Fall = ext.extract(TGT, tl_, layers="all", context_pool=ctx, subword_agg="mean")
    layers = sorted(Eall.keys())
    hb(f"[{pair}] extracted {len(rows)} words, {len(layers)} layers")
    per_layer = {}
    for L in layers:
        E = Eall[L].astype(np.float64)
        F = Fall[L].astype(np.float64)
        E = E - E.mean(0)
        F = F - F.mean(0)
        p1s = []
        sk = []
        mh = []
        nv = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(rows), min(M, len(rows)), replace=False)
            src = [SRC[i] for i in idx]
            gal = [TGT[i] for i in idx]
            X = unit(E[idx])
            Y = unit(F[idx])
            p1, N1 = retrieve(X, Y, gold, src, gal)
            p1s.append(p1)
            sk.append(float(skew(N1)))
            mh.append(int(N1.max()))
            nv.append(float((N1 == 0).mean()))
        def ci(v):
            v = np.array(v, float)
            return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]
        per_layer[str(L)] = {"p1": ci(p1s), "skew": ci(sk), "maxhub": ci(mh), "never": ci(nv)}
    out["pairs"][pair] = {"layers": layers, "per_layer": per_layer}
    best = max(layers, key=lambda L: per_layer[str(L)]["p1"][0])
    hb(f"[{pair}] best layer for P@1 = {best} ({per_layer[str(best)]['p1'][0]:.3f}); "
       f"skew there {per_layer[str(best)]['skew'][0]:.2f}")

os.makedirs("results", exist_ok=True)
with open(f"results/layer_sweep_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"LAYERSWEEP_{MODEL.upper()} DONE", flush=True)
