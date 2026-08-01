"""XNLI head training, layer selection, and correlational + causal analyses.

Works entirely off the cached features written by xnli_extract.py and xnli_hubness.py (no GPU).
For each encoder:
  1. Layer selection: train a linear head on English MNLI-train features at each candidate layer,
     pick the layer with best English MNLI-val accuracy.
  2. Zero-shot XNLI: train head on English (5 seeds, each a random 25k subsample), evaluate every
     XNLI language. Reports en_acc, per-language acc (mean +/- 95% CI), transfer gap = en - lang.
  3. Correlational: Pearson and Spearman of transfer gap vs per-language single-token hubness.
  4. Causal: fit the top-K PC "hub subspace" on pooled multilingual features, project it out of both
     train and test features, retrain the head, and compare mean target-language accuracy against
     no ablation and an equal-rank random-direction control. 5 seeds, 95% CIs.

Run: python experiments/xnli_analyze.py
Env: LAYERS=6,9,12, K=10, N_SEEDS=5, SUB=25000
Output: results/xnli_results.json
"""
from __future__ import annotations
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as _student_t

FEATROOT = "results/xnli_feats"
LAYERS = [int(x) for x in os.environ.get("LAYERS", "6,9,12").split(",")]
LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
TGT = [l for l in LANGS if l != "en"]
K = int(os.environ.get("K", "10"))
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
SUB = int(os.environ.get("SUB", "25000"))
hub = json.load(open("results/xnli_hubness.json"))

def ci(v):
    v = np.array(v, float)
    return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]

def standardize(Xtr, *others):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-6
    return [(A-mu)/sd for A in (Xtr,)+others]

def fit_eval(Xtr, ytr, tests, seed):
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
    clf.fit(Xtr[idx], ytr[idx])
    return {lang: float((clf.predict(Xte) == yte).mean()) for lang, (Xte, yte) in tests.items()}

def project_out(mats, dirs):
    Q = np.linalg.qr(dirs.T)[0].T
    return [A - (A @ Q.T) @ Q for A in mats]

out = {}
for model in ["mbert", "xlmr"]:
    d = f"{FEATROOT}/{model}"
    if not os.path.exists(f"{d}/train.npz"):
        print(f"skip {model} (no features)")
        continue
    tr = np.load(f"{d}/train.npz")
    dv = np.load(f"{d}/dev.npz")
    test = {l: np.load(f"{d}/test_{l}.npz") for l in LANGS}
    print(f"\n===== {model} =====", flush=True)

    # 1. layer selection on English MNLI-val
    layer_dev = {}
    for L in LAYERS:
        Xtr, Xdv = standardize(tr[f"L{L}"], dv[f"L{L}"])
        clf = LogisticRegression(max_iter=200, n_jobs=-1).fit(Xtr[:SUB], tr["y"][:SUB])
        layer_dev[L] = float((clf.predict(Xdv) == dv["y"]).mean())
    bestL = max(layer_dev, key=layer_dev.get)
    print(f"layer dev acc: {layer_dev}  -> select L{bestL}", flush=True)

    # 2. zero-shot transfer at selected layer (raw features), 5 seeds
    Xtr_raw = tr[f"L{bestL}"]
    ytr = tr["y"]
    mu = Xtr_raw.mean(0)
    sd = Xtr_raw.std(0)+1e-6
    Xtr = (Xtr_raw-mu)/sd
    tests = {l: ((test[l][f"L{bestL}"]-mu)/sd, test[l]["y"]) for l in LANGS}
    per_seed = [fit_eval(Xtr, ytr, tests, s) for s in range(N_SEEDS)]
    acc = {l: ci([ps[l] for ps in per_seed]) for l in LANGS}
    en_acc = acc["en"][0]
    gap = {l: round(en_acc - acc[l][0], 4) for l in TGT}
    print(f"en acc {en_acc:.3f}; mean target acc {np.mean([acc[l][0] for l in TGT]):.3f}", flush=True)

    # 3. correlational: gap vs single-token hubness
    g = np.array([gap[l] for l in TGT])
    h = np.array([hub[l]["skew_k1"] for l in TGT])
    hp = np.array([hub[l]["p1_k1"] for l in TGT])
    pr_skew = pearsonr(h, g)
    sp_skew = spearmanr(h, g)
    pr_p1 = pearsonr(hp, g)
    print(f"corr(gap, hub-skew): Pearson r={pr_skew[0]:.2f} p={pr_skew[1]:.3f}; "
          f"Spearman rho={sp_skew[0]:.2f} p={sp_skew[1]:.3f}", flush=True)
    print(f"corr(gap, single-token P@1): Pearson r={pr_p1[0]:.2f} p={pr_p1[1]:.3f}", flush=True)

    # 4. causal: project out top-K PC hub subspace (fit on pooled multilingual), vs random control
    pool = np.vstack([Xtr_raw[:4000]] + [test[l][f"L{bestL}"][:1500] for l in LANGS])
    poolc = pool - pool.mean(0)
    _, _, Vt = np.linalg.svd(poolc, full_matrices=False)
    hubdirs = Vt[:K]
    def run_condition(dirs):
        accs = []
        for s in range(N_SEEDS):
            if dirs is None:
                Xtr_a = Xtr_raw
                tests_a = {l: test[l][f"L{bestL}"] for l in LANGS}
            else:
                proj = project_out([Xtr_raw] + [test[l][f"L{bestL}"] for l in LANGS], dirs)
                Xtr_a = proj[0]
                tests_a = {l: proj[i+1] for i, l in enumerate(LANGS)}
            m2 = Xtr_a.mean(0)
            s2 = Xtr_a.std(0)+1e-6
            Xt = (Xtr_a-m2)/s2
            tt = {l: ((tests_a[l]-m2)/s2, test[l]["y"]) for l in LANGS}
            r = fit_eval(Xt, ytr, tt, s)
            accs.append(r)
        return accs
    base = run_condition(None)
    hubc = run_condition(hubdirs)
    rng = np.random.RandomState(0)
    randc = run_condition(rng.randn(K, Xtr_raw.shape[1]))
    def meantgt(runs): return [np.mean([r[l] for l in TGT]) for r in runs]
    def meanen(runs): return [r["en"] for r in runs]
    causal = {
        "baseline_tgt": ci(meantgt(base)), "hub_tgt": ci(meantgt(hubc)), "random_tgt": ci(meantgt(randc)),
        "baseline_en": ci(meanen(base)), "hub_en": ci(meanen(hubc)), "random_en": ci(meanen(randc)),
    }
    # paired hub-minus-random per seed (mean target acc)
    dlt = np.array(meantgt(hubc)) - np.array(meantgt(randc))
    causal["hub_minus_random_tgt"] = ci(dlt.tolist())
    causal["hub_minus_random_signs"] = int((dlt > 0).sum())
    print(f"CAUSAL (mean target acc): base {causal['baseline_tgt'][0]:.3f} | "
          f"hub {causal['hub_tgt'][0]:.3f} | random {causal['random_tgt'][0]:.3f} | "
          f"hub-rand {causal['hub_minus_random_tgt'][0]:+.3f}+/-{causal['hub_minus_random_tgt'][1]:.3f} "
          f"({causal['hub_minus_random_signs']}/{N_SEEDS} seeds)", flush=True)

    out[model] = {"layer_dev": layer_dev, "selected_layer": bestL, "K": K,
                  "acc": acc, "gap": gap, "en_acc": en_acc,
                  "corr": {"gap_vs_hubskew_pearson": [pr_skew[0], pr_skew[1]],
                           "gap_vs_hubskew_spearman": [sp_skew[0], sp_skew[1]],
                           "gap_vs_singletoken_p1_pearson": [pr_p1[0], pr_p1[1]]},
                  "causal": causal}

os.makedirs("results", exist_ok=True)
with open("results/xnli_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nXNLI_ANALYZE DONE", flush=True)
