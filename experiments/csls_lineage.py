"""Retrieval-time hubness corrections vs the low-rank subspace they touch.

Compares the family of retrieval-time hubness corrections, CSLS (Lample et al. 2018),
inverted softmax (Smith et al. 2017), and globally-corrected / GC (Dinu et al. 2015),
against the subspace ablation used elsewhere. It shows two things: (a) all of these
corrections flatten the same in-degree distribution and lift full-test BLI P@1, and (b) the
hub mass they correct lives in the same low-rank residual subspace that ABTT ablates. The
targets CSLS de-hubs are the ones with large projection onto the top-k principal components,
and once that subspace is ablated CSLS adds almost nothing on top.

Env: MODEL=mbert|xlmr, PAIRS=en-es,en-zh,en-de, M=2000, N_SEEDS=5, K=10, BETA=30
Out: results/csls_lineage_{MODEL}.json

Run: python experiments/csls_lineage.py
"""
from __future__ import annotations
import os, json, time
import numpy as np
from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew, pearsonr
from scipy.stats import t as _student_t

MODEL = os.environ.get("MODEL", "mbert")
EXTNAME = {"mbert": "mbert", "xlmr": "xlmr_base"}[MODEL]
PAIRS = os.environ.get("PAIRS", "en-es,en-zh,en-de").split(",")
M = int(os.environ.get("M", "2000"))
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
K = int(os.environ.get("K", "10"))
BETA = float(os.environ.get("BETA", "30"))
LAYER = int(os.environ.get("LAYER", "1"))
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext = build_extractor(EXTNAME)
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

def base_sim(X, Y): return cosine_similarity(X, Y)
def csls_sim(S, k=10):
    rT = np.partition(S, -k, axis=1)[:, -k:].mean(1, keepdims=True)
    rS = np.partition(S, -k, axis=0)[-k:, :].mean(0, keepdims=True)
    return 2*S - rT - rS
def invsoftmax_sim(S, beta):           # Smith et al. 2017: softmax over sources per target
    Z = S - S.max(0, keepdims=True)
    P = np.exp(beta*Z)
    P = P/(P.sum(0, keepdims=True)+1e-12)
    return P                            # prediction is argmax_j P[i, j]
def gc_sim(S):                          # Dinu et al. 2015: subtract per-target mean similarity
    return S - S.mean(0, keepdims=True)

def eval_pred(Smat, gold, src, gal):
    pred = Smat.argmax(1)
    corr = np.array([1 if gal[pred[j]] in gold.get(src[j], set()) else 0 for j in range(len(src))])
    N1 = np.bincount(pred, minlength=len(gal))
    return float(corr.mean()), N1, corr
def topk_pca(X, Y, k):
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k]
def project_out(A, B, dirs):
    Q = np.linalg.qr(dirs.T)[0].T
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q

out = {"model": MODEL, "layer": LAYER, "K": K, "beta": BETA, "results": {}}
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
    methods = ["cosine", "CSLS", "inv-softmax", "GC"]
    agg = {m: {"p1": [], "skew": [], "maxhub": [], "never": []} for m in methods}
    redun = {"base": [], "CSLS": [], "ABTT": [], "ABTT+CSLS": []}          # in-degree skew under each
    redun_p1 = {"base": [], "CSLS": [], "ABTT": [], "ABTT+CSLS": []}
    corr_n1_proj = []
    corr_reduce_proj = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), min(M, len(rows)), replace=False)
        src = [SRC[i] for i in idx]
        gal = [TGT[i] for i in idx]
        X = unit(E[idx]).astype(np.float64)
        Y = unit(F[idx]).astype(np.float64)
        S = base_sim(X, Y)
        sims = {"cosine": S, "CSLS": csls_sim(S, 10), "inv-softmax": invsoftmax_sim(S, BETA), "GC": gc_sim(S)}
        N1 = {}
        for m in methods:
            p1, n1, _ = eval_pred(sims[m], gold, src, gal)
            agg[m]["p1"].append(p1)
            agg[m]["skew"].append(float(skew(n1)))
            agg[m]["maxhub"].append(int(n1.max()))
            agg[m]["never"].append(float((n1 == 0).mean()))
            N1[m] = n1
        # Do the targets CSLS de-hubs live in the top-K PC subspace?
        dirs = topk_pca(X, Y, K)
        Q = np.linalg.qr(dirs.T)[0].T
        # Fraction of each target's norm that lies in the subspace.
        proj = np.linalg.norm(Y @ Q.T, axis=1) / (np.linalg.norm(Y, axis=1)+1e-9)
        reduce_csls = N1["cosine"].astype(float) - N1["CSLS"].astype(float)
        m_hub = N1["cosine"] > 0
        corr_n1_proj.append(float(pearsonr(N1["cosine"][m_hub], proj[m_hub])[0]))
        corr_reduce_proj.append(float(pearsonr(reduce_csls[m_hub], proj[m_hub])[0]))
        # Redundancy check: after ABTT, CSLS adds little.
        Xa, Ya = project_out(X, Y, dirs)
        Sa = base_sim(unit(Xa), unit(Ya))
        for nm, Smat in [("base", S), ("CSLS", sims["CSLS"]), ("ABTT", Sa), ("ABTT+CSLS", csls_sim(Sa, 10))]:
            p1, n1, _ = eval_pred(Smat, gold, src, gal)
            redun[nm].append(float(skew(n1)))
            redun_p1[nm].append(p1)
        hb(f"  seed {seed} done")
    def ci(v):
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]
    out["results"][pair] = {
        "methods": {m: {k: ci(agg[m][k]) for k in agg[m]} for m in methods},
        "subspace_tie": {"corr_N1_proj": ci(corr_n1_proj), "corr_CSLSreduction_proj": ci(corr_reduce_proj)},
        "redundancy_skew": {k: ci(redun[k]) for k in redun},
        "redundancy_p1": {k: ci(redun_p1[k]) for k in redun_p1},
    }
    r = out["results"][pair]
    print(f"[{pair}] {MODEL}: cosine skew {r['methods']['cosine']['skew'][0]:.2f} -> "
          f"CSLS {r['methods']['CSLS']['skew'][0]:.2f} / invsm {r['methods']['inv-softmax']['skew'][0]:.2f} / "
          f"GC {r['methods']['GC']['skew'][0]:.2f}", flush=True)
    print(f"        corr(N1,subspace-proj)={r['subspace_tie']['corr_N1_proj'][0]:.2f}  "
          f"corr(CSLS-reduction,proj)={r['subspace_tie']['corr_CSLSreduction_proj'][0]:.2f}", flush=True)
    print(f"        skew base {r['redundancy_skew']['base'][0]:.2f} | CSLS {r['redundancy_skew']['CSLS'][0]:.2f} | "
          f"ABTT {r['redundancy_skew']['ABTT'][0]:.2f} | ABTT+CSLS {r['redundancy_skew']['ABTT+CSLS'][0]:.2f}", flush=True)

os.makedirs("results", exist_ok=True)
with open(f"results/csls_lineage_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"LINEAGE_{MODEL.upper()} DONE", flush=True)
