"""E3: per-word bootstrapped indirect effect (ambiguity -> hub geometry -> retrieval failure).

On the per-word logistic-regression dataset (mBERT layer-1, identity retrieval, en-es, N=4000):
  treatment X = log polysemy (log1p MUSE translation count)
  mediator  M = projection of the word onto the top-10 hub subspace (how much it lives in the hub
                directions), the geometric quantity ABTT removes; standardized
  outcome   Y = retrieval success (gold target retrieved at rank 1)
Product-of-coefficients indirect effect: a = slope of M ~ X; b = logistic coef of M in Y ~ M + X;
indirect = a*b. 2000 bootstrap resamples over words, BCa 95% CI. Gate: if the CI includes 0, the paper
falls back to covariation wording (r=-0.72 relabeled), not "mediates".

Also reports centrality (mean cosine to gallery) as an alternative mediator for robustness.

Out: results/perword_mediation.json (run from the repo root, like the other experiment scripts).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

ext = build_extractor("mbert"); ctx = IsolationSpec()
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)
allp = load_muse_dictionary("en-es"); gold = build_gold_map(allp)
ss = set(); tt = set(); rows = []
for s, t in allp:
    if s in ss or t in tt: continue
    ss.add(s); tt.add(t); rows.append((s, t))
    if len(rows) >= 4000: break
SRC = [s for s, _ in rows]; TGT = [t for _, t in rows]
E = ext.extract(SRC, "en", layers=[1], context_pool=ctx, subword_agg="mean")[1]
F = ext.extract(TGT, "es", layers=[1], context_pool=ctx, subword_agg="mean")[1]
E = E - E.mean(0); F = F - F.mean(0)
X = unit(E); Y = unit(F)
S = cosine_similarity(X, Y); pred = S.argmax(1)
success = np.array([1.0 if TGT[pred[j]] in gold.get(SRC[j], set()) else 0.0 for j in range(len(SRC))])
# hub subspace = top-10 PCs of concatenated centered unit space (same as ABTT)
Z = np.vstack([X, Y]); Zc = Z - Z.mean(0); _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
Q = Vt[:10]
proj = np.linalg.norm(X @ Q.T, axis=1)                       # mediator 1: hub-subspace projection (en side)
centr = cosine_similarity(X, Y).mean(1)                       # mediator 2: mean cosine to gallery (centrality)
poly = np.log1p(np.array([len(gold.get(s, set())) for s in SRC], float))   # treatment

def z(v): return (v - v.mean())/(v.std()+1e-12)
Xt = z(poly)

def indirect(xt, med, y, idx):
    xs, ms, ys = xt[idx], z(med[idx]), y[idx]
    a = np.polyfit(xs, ms, 1)[0]                              # M ~ X slope
    clf = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([ms, xs]), ys)
    b = clf.coef_.ravel()[0]                                  # coef of M in Y ~ M + X
    return a*b, a, b

def bca(point, boot, jack):
    boot = np.array(boot); z0 = _norm_ppf((boot < point).mean())
    jbar = jack.mean(); num = ((jbar - jack)**3).sum(); den = 6*(((jbar - jack)**2).sum()**1.5)+1e-12
    acc = num/den
    def adj(al):
        zal = _norm_ppf(al); p = _norm_cdf(z0 + (z0+zal)/(1-acc*(z0+zal)))
        return float(np.percentile(boot, 100*p))
    return adj(0.025), adj(0.975)

from math import erf, sqrt
def _norm_cdf(x): return 0.5*(1+erf(x/sqrt(2)))
def _norm_ppf(p):
    p = min(max(p, 1e-6), 1-1e-6)                             # Acklam rational approx
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    dd=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=sqrt(-2*np.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p<=1-pl:
        q=p-0.5; r=q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=sqrt(-2*np.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)

n = len(success); rng = np.random.RandomState(0); B = 2000
out = {"n": n, "overall_p1": float(success.mean()), "mediators": {}}
for mname, med in [("hub_projection", proj), ("centrality_meancos", centr)]:
    pt, a0, b0 = indirect(Xt, med, success, np.arange(n))
    boot = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if success[idx].sum() in (0, n): continue
        boot.append(indirect(Xt, med, success, idx)[0])
    jack = np.array([indirect(Xt, med, success, np.delete(np.arange(n), i))[0] for i in rng.choice(n, 200, replace=False)])
    lo, hi = bca(pt, boot, jack)
    excl0 = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    out["mediators"][mname] = {"indirect": float(pt), "a_XtoM": float(a0), "b_MtoY": float(b0),
                               "ci_bca": [float(lo), float(hi)], "excludes_zero": bool(excl0), "n_boot": len(boot)}
    print(f"[{mname}] indirect a*b = {pt:+.4f}  (a={a0:+.3f}, b={b0:+.3f})  BCa95% [{lo:+.4f},{hi:+.4f}]  "
          f"{'EXCLUDES 0' if excl0 else 'includes 0'}", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/perword_mediation.json", "w") as f:
    json.dump(out, f, indent=2)
print("E3_PERWORD_MEDIATION DONE", flush=True)
