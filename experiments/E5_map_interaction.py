"""E5: does hub-subspace ablation help on top of a LEARNED cross-lingual map?

The causal dial (S5) is scoped to map-free identity retrieval by design. A BLI reviewer will
ask the obvious next question: once you fit a supervised map, does de-hubbing the representation
still help, or does the map already absorb the hubness gain? This script answers it on the same
cells as the main dial result, with the matched random-direction control, so it slots into the
existing tables.

Learned map: orthogonal Procrustes (the field-standard supervised BLI map, argmin_W ||XW - Y||
with W orthogonal; smith2017offline), fit on a TRAIN dictionary half and evaluated on a disjoint
TEST half. Everything estimated on train, evaluated on test:
  - per-language centering means (train only),
  - the top-K hub subspace (stacked SVD on the de-duplicated train words), and
  - the Procrustes map W (train pairs only).
No evaluated word in either language touches the estimation, so the comparison is leakage-free
(same discipline as E4's cross-split).

Conditions (all evaluated on the held-out TEST half; full-test P@1 is the headline, anti-hub
recovery reported alongside):
  identity        centered/unit, NN retrieval, no map (reference)
  map             Procrustes, NN retrieval
  abtt_map        project out top-K hub subspace from both sides, then fit+apply Procrustes
  random_map      project out K random directions (matched rank), then fit+apply Procrustes  [control]
  map_csls        Procrustes, then CSLS retrieval (retrieval-time hubness fix on top of the map)
  abtt_map_csls   de-hub + Procrustes + CSLS

The paired contrast of interest is abtt_map - random_map: if positive and > 0 across seeds,
de-hubbing helps on top of the learned map (a practical lever); if it sits at zero, the map
absorbs the hubness gain (an honest scope statement). CSLS variants situate the representation-space
intervention against the standard retrieval-time correction.

Protocol matches experiments/hub_ablation.py / E4_subspace_transfer.py: cached layer-1 activations
(cache hits, no new forward passes), MUSE dedup capped at 4000 rows, IsolationSpec, subword-mean,
unit norm, M=2000 per half, 5 seeds, K=10. Deterministic RandomState seeds: 400+ (splits),
410+ (random-direction control). Out: results/E5_map_interaction.json.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Forcing HF_HUB_OFFLINE here would break a first run on a machine with an empty
# model cache; export it yourself if you want to pin an already-downloaded copy.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew as skewness
from scipy.stats import t as _student_t

PAIRS = ["en-es", "en-de", "en-fr", "en-it", "en-zh"]
M, N_SEEDS, LAYER, K = 2000, 5, 1, 10
ctx = IsolationSpec()
_T0 = time.time()


def hb(m):
    print(f"  [E5 {time.time()-_T0:.1f}s] {m}", flush=True)


def unit(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def project_out(A, B, dirs):
    """Project directions `dirs` out of both sides (same as E4)."""
    if dirs is None or len(dirs) == 0:
        return A, B
    Q = np.linalg.qr(dirs.T)[0].T
    return A - (A @ Q.T) @ Q, B - (B @ Q.T) @ Q


def stacked_svd(X, Y):
    """Right singular vectors of the stacked, mean-centered bilingual matrix (same as E4)."""
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    return s, Vt


def procrustes(Xtr, Ytr):
    """Orthogonal map W (argmin ||Xtr W - Ytr||_F, W orthogonal). W = U V^T of Ytr^T Xtr."""
    U, _, Vt = np.linalg.svd(Ytr.T @ Xtr)
    return (U @ Vt).T  # so that Xtr @ W approximates Ytr


def retrieve(X, Y, gold, src, gal, csls=False):
    """Identity/mapped cosine NN retrieval; optional CSLS. Same math as E4."""
    S = cosine_similarity(X, Y)
    if csls:
        k = 10
        rT = np.partition(S, -k, axis=1)[:, -k:].mean(1, keepdims=True)
        rS = np.partition(S, -k, axis=0)[-k:, :].mean(0, keepdims=True)
        S = 2 * S - rT - rS
    pred = S.argmax(1)
    corr = np.array([1 if gal[pred[j]] in gold.get(src[j], set()) else 0 for j in range(len(src))])
    N1 = np.bincount(pred, minlength=len(gal))
    return float(corr.mean()), pred, N1, corr


def metrics(X, Y, gold, src, gal, anti_q, csls=False):
    p1, pred, N1, corr = retrieve(unit(X), unit(Y), gold, src, gal, csls=csls)
    rec = float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
    return {"p1": p1, "skew": float(skewness(N1)), "never": float((N1 == 0).mean()),
            "antihub_rec": rec}


def ci(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0:
        return [float("nan"), 0.0]
    return [float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0]


def agg(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {k: ci([d[k] for d in list_of_dicts]) for k in keys}


def load_pair(ext, pair, layer):
    sl, tl = pair.split("-")
    allp = load_muse_dictionary(pair)
    gold = build_gold_map(allp)
    ss, tt, rows = set(), set(), []
    for s, t in allp:
        if s in ss or t in tt:
            continue
        ss.add(s); tt.add(t); rows.append((s, t))
        if len(rows) >= 4000:
            break
    SRC = [s for s, _ in rows]
    TGT = [t for _, t in rows]
    E = np.asarray(ext.extract(SRC, sl, layers=[layer], context_pool=ctx, subword_agg="mean")[layer], np.float64)
    F = np.asarray(ext.extract(TGT, tl, layers=[layer], context_pool=ctx, subword_agg="mean")[layer], np.float64)
    return SRC, TGT, gold, E, F


CONDS = ["identity", "map", "abtt_map", "random_map", "map_csls", "abtt_map_csls"]

out = {"config": {"pairs": PAIRS, "M": M, "n_seeds": N_SEEDS, "layer": LAYER, "k": K,
                  "map": "orthogonal Procrustes (supervised), fit on disjoint TRAIN half",
                  "protocol": "leakage-free train/test split; centering, hub subspace, and map all "
                              "estimated on TRAIN only; cached layer-1 activations"},
       "map_interaction": {}}

for model, extname in [("mbert", "mbert"), ("xlmr", "xlmr_base")]:
    ext = build_extractor(extname)
    data = {p: load_pair(ext, p, LAYER) for p in PAIRS}
    d = data[PAIRS[0]][3].shape[1]
    hb(f"{model}: loaded {len(PAIRS)} pairs (d={d})")

    res = {p: {c: [] for c in CONDS} for p in PAIRS}
    # per-seed paired deltas for the honest "n/N seeds positive" statistic
    paired = {p: {"abtt_minus_random_p1": [], "abtt_minus_random_anti": [],
                  "abtt_minus_map_p1": [], "csls_gain_over_map_p1": [],
                  "abtt_csls_minus_map_csls_p1": []} for p in PAIRS}

    for seed in range(N_SEEDS):
        for pair in PAIRS:
            SRC_f, TGT_f, gold_f, E_raw, F_raw = data[pair]
            rng_s = np.random.RandomState(400 + seed)
            perm = rng_s.permutation(len(SRC_f))
            nh = min(M, len(SRC_f) // 2)
            train, test = perm[:nh], perm[nh:2 * nh]

            # train-only per-language centering
            muE, muF = E_raw[train].mean(0), F_raw[train].mean(0)
            Xtr, Ytr = unit(E_raw[train] - muE), unit(F_raw[train] - muF)
            Xte, Yte = unit(E_raw[test] - muE), unit(F_raw[test] - muF)

            src_te = [SRC_f[i] for i in test]
            gal_te = [TGT_f[i] for i in test]

            # anti-hub queries: gold target is an anti-hub under identity NN retrieval on TEST
            _, _, N1b, _ = retrieve(Xte, Yte, gold_f, src_te, gal_te)
            gpos = {w: j for j, w in enumerate(gal_te)}
            anti_q = [q for q in range(len(src_te))
                      if any(gpos.get(t, -1) >= 0 and N1b[gpos[t]] == 0
                             for t in gold_f.get(src_te[q], set()))]

            # top-K hub subspace and K random directions, estimated on TRAIN only
            _, Vt_tr = stacked_svd(Xtr, Ytr)
            Q_hub = Vt_tr[:K]
            rng_r = np.random.RandomState(410 + seed)
            Q_rand = rng_r.randn(K, d)

            # ---- identity (no map) ----
            res[pair]["identity"].append(metrics(Xte, Yte, gold_f, src_te, gal_te, anti_q))

            # ---- map: Procrustes fit on train, applied to test source ----
            W = procrustes(Xtr, Ytr)
            res[pair]["map"].append(metrics(Xte @ W, Yte, gold_f, src_te, gal_te, anti_q))
            res[pair]["map_csls"].append(metrics(Xte @ W, Yte, gold_f, src_te, gal_te, anti_q, csls=True))

            # ---- abtt + map: de-hub both sides (train AND test) with the train subspace, refit map ----
            Xtr_a, Ytr_a = project_out(Xtr, Ytr, Q_hub)
            Xte_a, Yte_a = project_out(Xte, Yte, Q_hub)
            Wa = procrustes(unit(Xtr_a), unit(Ytr_a))
            res[pair]["abtt_map"].append(metrics(unit(Xte_a) @ Wa, unit(Yte_a), gold_f, src_te, gal_te, anti_q))
            res[pair]["abtt_map_csls"].append(metrics(unit(Xte_a) @ Wa, unit(Yte_a), gold_f, src_te, gal_te, anti_q, csls=True))

            # ---- random + map: matched-rank random-direction control ----
            Xtr_r, Ytr_r = project_out(Xtr, Ytr, Q_rand)
            Xte_r, Yte_r = project_out(Xte, Yte, Q_rand)
            Wr = procrustes(unit(Xtr_r), unit(Ytr_r))
            res[pair]["random_map"].append(metrics(unit(Xte_r) @ Wr, unit(Yte_r), gold_f, src_te, gal_te, anti_q))

            m = {c: res[pair][c][-1] for c in CONDS}
            paired[pair]["abtt_minus_random_p1"].append(m["abtt_map"]["p1"] - m["random_map"]["p1"])
            paired[pair]["abtt_minus_random_anti"].append(m["abtt_map"]["antihub_rec"] - m["random_map"]["antihub_rec"])
            paired[pair]["abtt_minus_map_p1"].append(m["abtt_map"]["p1"] - m["map"]["p1"])
            paired[pair]["csls_gain_over_map_p1"].append(m["map_csls"]["p1"] - m["map"]["p1"])
            paired[pair]["abtt_csls_minus_map_csls_p1"].append(m["abtt_map_csls"]["p1"] - m["map_csls"]["p1"])
        hb(f"{model} seed {seed} done")

    def npos(vals):
        return int(sum(1 for x in vals if x > 0))

    out["map_interaction"][model] = {
        p: {**{c: agg(res[p][c]) for c in CONDS},
            "paired": {k: ci(v) for k, v in paired[p].items()},
            "seeds_positive": {k: [npos(v), len(v)] for k, v in paired[p].items()}}
        for p in PAIRS}

os.makedirs("results", exist_ok=True)
with open("results/E5_map_interaction.json", "w") as f:
    json.dump(out, f, indent=2)
hb("wrote results/E5_map_interaction.json")

# compact console summary
for model in ["mbert", "xlmr"]:
    print(f"\n=== {model} : full-test P@1 (5 seeds) ===")
    print(f"{'pair':7s} {'ident':>6s} {'map':>6s} {'abtt+map':>9s} {'rand+map':>9s} "
          f"{'Δ(abtt-rand)':>13s} {'seeds+':>7s} {'map+csls':>9s} {'abtt+map+csls':>13s}")
    for p in PAIRS:
        r = out["map_interaction"][model][p]
        dpos = r["seeds_positive"]["abtt_minus_random_p1"]
        print(f"{p:7s} {r['identity']['p1'][0]:6.3f} {r['map']['p1'][0]:6.3f} "
              f"{r['abtt_map']['p1'][0]:9.3f} {r['random_map']['p1'][0]:9.3f} "
              f"{r['paired']['abtt_minus_random_p1'][0]:+13.3f} {dpos[0]}/{dpos[1]:>5} "
              f"{r['map_csls']['p1'][0]:9.3f} {r['abtt_map_csls']['p1'][0]:13.3f}")
