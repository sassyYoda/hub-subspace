"""Multivariate logistic regression of en-es per-word retrieval success.

Setup is mBERT layer 1, identity cosine retrieval against the full gallery. For each English source
word we record success (gold target retrieved at rank 1) and five features:
  polysemy   = number of gold translations (MUSE), log1p
  function   = 1 if in the closed-class function-word list
  fertility  = number of mBERT subword tokens, log1p
  length     = source word length in characters, log1p
  logfreq    = log1p(MUSE frequency rank); higher = rarer
Predictors are standardised. Logistic regression is near-unregularised (MLE), with 95% CIs from a
bootstrap over words (B=2000). A factor "survives jointly" if its standardised-coefficient CI
excludes 0.

Run: python experiments/perword_logreg.py
Output: results/perword_logreg.json
"""
from __future__ import annotations
import os, json
import numpy as np
from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

FUNC = set("the a an of to in on at by for with from as and or but if then than that this these those "
 "i you he she it we they me him her us them my your his its our their is are was were be been being "
 "have has had do does did will would can could should may might must not yes so very more most some "
 "any all each every none who what which when where why how about into over under after before".split())

ext = build_extractor("mbert")
ctx = IsolationSpec()
tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
def unit(A): return A/(np.linalg.norm(A, axis=1, keepdims=True)+1e-9)

allp = load_muse_dictionary("en-es")
gold = build_gold_map(allp)
ss = set()
tt = set()
rows = []
for s, t in allp:
    if s in ss or t in tt: continue
    ss.add(s); tt.add(t); rows.append((s, t))
    if len(rows) >= 4000: break
SRC = [s for s, _ in rows]
TGT = [t for _, t in rows]
E = ext.extract(SRC, "en", layers=[1], context_pool=ctx, subword_agg="mean")[1]
F = ext.extract(TGT, "es", layers=[1], context_pool=ctx, subword_agg="mean")[1]
E = E - E.mean(0)
F = F - F.mean(0)
X = unit(E)
Y = unit(F)
S = cosine_similarity(X, Y)
pred = S.argmax(1)
correct = np.array([1 if TGT[pred[j]] in gold.get(SRC[j], set()) else 0 for j in range(len(SRC))], float)

poly = np.array([len(gold.get(s, set())) for s in SRC], float)
func = np.array([1.0 if s in FUNC else 0.0 for s in SRC])
fert = np.array([len(tok.tokenize(s)) for s in SRC], float)
length = np.array([len(s) for s in SRC], float)
freqrank = np.arange(len(SRC), dtype=float)  # MUSE dict is frequency-ordered
feat = np.column_stack([np.log1p(poly), func, np.log1p(fert), np.log1p(length), np.log1p(freqrank)])
names = ["log polysemy", "function word", "log fertility", "log length", "log freq-rank"]
mu = feat.mean(0)
sd = feat.std(0) + 1e-9
Z = (feat - mu) / sd

def fit(Zb, yb):
    clf = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    clf.fit(Zb, yb)
    return np.concatenate([clf.coef_.ravel(), clf.intercept_])

base = fit(Z, correct)
B = 2000
rng = np.random.RandomState(0)
n = len(correct)
boot = np.zeros((B, Z.shape[1]+1))
for b in range(B):
    ii = rng.randint(0, n, n)
    # logistic regression is undefined if the resample has only one class
    if correct[ii].sum() in (0, len(ii)):
        boot[b] = np.nan
        continue
    boot[b] = fit(Z[ii], correct[ii])
boot = boot[~np.isnan(boot).any(1)]

out = {"pair": "en-es", "model": "mbert", "layer": 1, "n_words": int(n),
       "overall_p1": float(correct.mean()), "n_bootstrap": int(boot.shape[0]), "coefficients": {}}
print(f"en-es mBERT L1 identity: N={n} overall P@1={correct.mean():.3f}  (function words n={int(func.sum())})", flush=True)
print("standardised logistic-regression coefficients (95% bootstrap CI), success ~ all five jointly:", flush=True)
for i, nm in enumerate(names):
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    sig = "*" if (lo > 0 or hi < 0) else " "
    out["coefficients"][nm] = {"coef": float(base[i]), "ci": [float(lo), float(hi)],
                               "odds_ratio": float(np.exp(base[i])), "survives": bool(lo > 0 or hi < 0)}
    print(f"  {sig} {nm:16s} beta={base[i]:+.3f}  OR={np.exp(base[i]):.3f}  CI[{lo:+.3f},{hi:+.3f}]", flush=True)

# univariate coefficients, for contrast with the joint fit
out["univariate"] = {}
for i, nm in enumerate(names):
    cu = LogisticRegression(C=1e6, max_iter=2000).fit(Z[:, [i]], correct)
    out["univariate"][nm] = float(cu.coef_.ravel()[0])

os.makedirs("results", exist_ok=True)
with open("results/perword_logreg.json", "w") as f:
    json.dump(out, f, indent=2)
print("PERWORD_LOGREG DONE", flush=True)
