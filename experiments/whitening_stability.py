"""Subspace stability and ZCA whitening vs ABTT.

Two analyses, both running on the cached layer-1 activations with the same retrieval and
metric code as hub_ablation.py. No model forward pass beyond reloading the cached extraction.

Stability: top-k=10 principal components of the concatenated bilingual space, recomputed per
  seed from a resampled M draw, then principal angles between every pair of the seed subspaces.
  Reports mean and max principal angle (degrees) and mean canonical correlation (subspace
  overlap). A sample-size sweep over M in {500, 1000, 2000, 4000} on en-es shows the angle
  shrink as M grows.

Whitening: ZCA whitening (fit on the concatenated centered bilingual space) vs ABTT-k10, using
  the same metrics as the ablation table (P@1, anti-hub recovery, in-degree skew, fraction never
  retrieved), plus the fraction of total variance the top-10 subspace carries (what ABTT discards
  and whitening only rescales).

Env: MODEL=mbert|xlmr, PAIRS=..., M=2000, N_SEEDS=5, LAYER=1
Out: results/whitening_stability_{MODEL}.json

Run: python experiments/whitening_stability.py
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
KSUB = 10
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext = build_extractor(EXTNAME)
ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

def topk_pca(X, Y, k):
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, Sv, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k], Sv

def principal_angles(Q1, Q2):
    # Rows of Q1, Q2 are orthonormal bases; singular values of Q1 Q2^T are cos(theta_i).
    s = np.linalg.svd(Q1 @ Q2.T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))  # angles in ascending order

def project_out(A, B, dirs):
    Q = np.linalg.qr(dirs.T)[0].T
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q

def zca_whiten_pair(X, Y, eps=1e-3):
    # Fit ZCA on the concatenated, centered bilingual space, matching the concat-PC convention.
    Z = np.vstack([X, Y])
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / (Zc.shape[0]-1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0/np.sqrt(S+eps)) @ U.T
    return (X-mu) @ W, (Y-mu) @ W

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

def ci(v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]

out = {"model": MODEL, "layer": LAYER, "ksub": KSUB, "stability": {}, "whitening": {}, "msweep": {}}
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

    # Stability: per-seed top-k subspaces and principal angles across seeds.
    seed_bases = []
    vexp = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), min(M, len(rows)), replace=False)
        X = unit(E[idx]).astype(np.float64)
        Y = unit(F[idx]).astype(np.float64)
        Vt, Sv = topk_pca(X, Y, KSUB)
        seed_bases.append(Vt)
        vexp.append(float((Sv[:KSUB]**2).sum() / (Sv**2).sum()))
    mean_ang = []
    max_ang = []
    mean_cos = []
    for a in range(N_SEEDS):
        for b in range(a+1, N_SEEDS):
            ang = principal_angles(seed_bases[a], seed_bases[b])
            mean_ang.append(float(ang.mean()))
            max_ang.append(float(ang.max()))
            mean_cos.append(float(np.cos(np.radians(ang)).mean()))
    out["stability"][pair] = {
        "mean_principal_angle_deg": ci(mean_ang),
        "max_principal_angle_deg": ci(max_ang),
        "mean_canonical_corr": ci(mean_cos),
        "topk_var_explained": ci(vexp),
        "n_seed_pairs": len(mean_ang),
    }
    print(f"[{pair}] {MODEL} stability: mean angle {ci(mean_ang)[0]:.1f}deg max {ci(max_ang)[0]:.1f}deg "
          f"cancorr {ci(mean_cos)[0]:.3f} topk-var {ci(vexp)[0]:.3f}", flush=True)

    # Whitening: ZCA vs ABTT, same metrics as the ablation table.
    conds = ["baseline", "CSLS", "ABTT-k10", "ZCA"]
    agg = {c: {"p1": [], "skew": [], "never": [], "antihub_rec": []} for c in conds}
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
            return p1, float(skew(N1)), float((N1 == 0).mean()), rec
        def run(name, Xa, Ya, csls=False):
            for key, val in zip(["p1", "skew", "never", "antihub_rec"], metrics(Xa, Ya, csls=csls)):
                agg[name][key].append(val)
        run("baseline", X, Y)
        run("CSLS", X, Y, csls=True)
        Xa, Ya = project_out(X, Y, topk_pca(X, Y, KSUB)[0])
        run("ABTT-k10", Xa, Ya)
        Xw, Yw = zca_whiten_pair(X, Y)
        run("ZCA", Xw, Yw)
    out["whitening"][pair] = {c: {k: ci(agg[c][k]) for k in agg[c]} for c in conds}
    r = out["whitening"][pair]
    print(f"[{pair}] {MODEL} whitening: base p1 {r['baseline']['p1'][0]:.3f} skew {r['baseline']['skew'][0]:.2f} "
          f"antihub {r['baseline']['antihub_rec'][0]:.3f} | ABTT p1 {r['ABTT-k10']['p1'][0]:.3f} skew {r['ABTT-k10']['skew'][0]:.2f} "
          f"antihub {r['ABTT-k10']['antihub_rec'][0]:.3f} | ZCA p1 {r['ZCA']['p1'][0]:.3f} skew {r['ZCA']['skew'][0]:.2f} "
          f"antihub {r['ZCA']['antihub_rec'][0]:.3f}", flush=True)

# Sample-size sensitivity of the stability angle, en-es only.
if "en-es" in PAIRS:
    sl_, tl_ = "en", "es"
    allp = load_muse_dictionary("en-es")
    gold = build_gold_map(allp)
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
    for Ms in [500, 1000, 2000, 4000]:
        bases = []
        for seed in range(N_SEEDS):
            # Offset the seed by 100 so this sweep does not reuse the main loop's draws.
            rng = np.random.RandomState(100+seed)
            idx = rng.choice(len(rows), min(Ms, len(rows)), replace=(Ms > len(rows)))
            X = unit(E[idx]).astype(np.float64)
            Y = unit(F[idx]).astype(np.float64)
            bases.append(topk_pca(X, Y, KSUB)[0])
        ang = [float(principal_angles(bases[a], bases[b]).mean()) for a in range(N_SEEDS) for b in range(a+1, N_SEEDS)]
        out["msweep"][str(Ms)] = ci(ang)
        print(f"[en-es msweep] M={Ms}: mean principal angle {ci(ang)[0]:.1f}deg", flush=True)

os.makedirs("results", exist_ok=True)
with open(f"results/whitening_stability_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"STABILITY_WHITENING_{MODEL.upper()} DONE", flush=True)
