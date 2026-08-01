"""E4: cross-pair and cross-split transfer of the hub subspace.

Four parts, all on the cached layer-1 activations (cache hits; no new forward passes):

A. CROSS-PAIR TRANSFER. Estimate the top-10 hub subspace on the reference pair (en-es), transfer
   the projection unchanged to en-{de,fr,it,zh}, and compare anti-hub recovery, P@1, and in-degree
   skew against (i) the pair-native ABTT-k10 subspace, (ii) the equal-rank random-direction
   control, and (iii) CSLS. Also reports the principal angles between the reference and native
   subspaces, the en-side vocabulary overlap between the reference estimation draw and each
   evaluation draw, and a STRICT recovery number restricted to anti-hub queries whose English word
   never entered the reference estimation sample (zero shared words between estimation and
   evaluation).

B. CROSS-SPLIT TRANSFER. Within each pair, split the deduplicated rows into disjoint TRAIN/TEST
   halves. Per-language centering means AND the top-10 subspace are estimated on TRAIN only and
   applied to TEST only, so no test word in either language touches the estimation. Compared
   against the in-sample subspace (estimated on TEST) and the random control. This answers the
   leakage question directly.

C. LANGUAGE-ID CONTROL (styled like the fertility/length negative control). Per-language centering
   removes the class means, but a language-ID direction can survive in the residual structure. We
   fit a logistic en-vs-target classifier on the stacked centered space, report its held-out
   accuracy, its principal angle to the hub subspace, and the effect of ablating it alone,
   alongside ABTT-k1 and a random-k1 control. Expected: high classifier accuracy, near-orthogonal
   angle, ablation ~= random.

D. INTER-PAIR PRINCIPAL ANGLES between the five pair-native subspaces per encoder, with a
   random-subspace reference angle, plus the leading marginal eigenvalue shares (for the
   k-selection rule: k* = max{k : lambda_k / sum lambda > tau/d}).

Protocol matches experiments/hub_ablation.py exactly: MUSE dedup capped at 4000 rows, layer 1,
IsolationSpec, subword-mean, per-language centering, unit norm, M=2000, 5 seeds, identity cosine
retrieval; anti-hub recovery is P@1 on queries whose gold target had baseline in-degree 0.

Optional: L4=1 additionally runs the ABTT spot-check at mBERT layer 4 (the mBERT word-BLI peak,
en-es). This is the one part that can trigger NEW forward passes (layer 4 is not in the cache);
it is off by default and everything else runs without it.

Deterministic: RandomState seeds 0..4 (draws), 100+ (splits), 200+ (language-ID), 300+ (random
subspace reference). Out: results/E4_transfer.json. Run from anywhere; the script
chdirs to the repo root so the extractor's results/cache path resolves.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # extractor cache dir is results/cache, relative to the repo root

from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from scipy.stats import skew as skewness
from scipy.stats import t as _student_t

PAIRS = ["en-es", "en-de", "en-fr", "en-it", "en-zh"]
REF_PAIR = "en-es"
M, N_SEEDS, LAYER, K = 2000, 5, 1, 10
RUN_L4 = os.environ.get("L4", "0") == "1"
ctx = IsolationSpec()
_T0 = time.time()


def hb(m):
    print(f"  [E4 {time.time()-_T0:.1f}s] {m}", flush=True)


def unit(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def retrieve(X, Y, gold, src, gal, csls=False):
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


def project_out(A, B, dirs):
    if dirs is None or len(dirs) == 0:
        return A, B
    Q = np.linalg.qr(dirs.T)[0].T
    return A - (A @ Q.T) @ Q, B - (B @ Q.T) @ Q


def stacked_svd(X, Y):
    """Singular values and right singular vectors of the stacked, mean-centered bilingual matrix."""
    Z = np.vstack([X, Y])
    Zc = Z - Z.mean(0)
    _, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    return s, Vt


def principal_angles_deg(Qa, Qb):
    s = np.clip(np.linalg.svd(Qa @ Qb.T, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def metrics(X, Y, gold, src, gal, anti_q, csls=False, corr_mask=None):
    p1, pred, N1, corr = retrieve(unit(X), unit(Y), gold, src, gal, csls=csls)
    rec = float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
    out = {"p1": p1, "maxhub": int(N1.max()), "skew": float(skewness(N1)),
           "never": float((N1 == 0).mean()), "antihub_rec": rec}
    if corr_mask is not None:
        sq = [q for q in anti_q if corr_mask[q]]
        out["antihub_rec_strict"] = float(np.mean([corr[q] for q in sq])) if sq else float("nan")
        out["n_antihub_strict"] = len(sq)
    return out


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
        ss.add(s)
        tt.add(t)
        rows.append((s, t))
        if len(rows) >= 4000:
            break
    SRC = [s for s, _ in rows]
    TGT = [t for _, t in rows]
    E = np.asarray(ext.extract(SRC, sl, layers=[layer], context_pool=ctx, subword_agg="mean")[layer], np.float64)
    F = np.asarray(ext.extract(TGT, tl, layers=[layer], context_pool=ctx, subword_agg="mean")[layer], np.float64)
    return SRC, TGT, gold, E, F


out = {"config": {"pairs": PAIRS, "ref_pair": REF_PAIR, "M": M, "n_seeds": N_SEEDS, "layer": LAYER,
                  "k": K, "protocol": "hub_ablation.py-identical (dedup 4000, IsolationSpec, "
                  "subword-mean, per-language centering, unit norm, identity cosine retrieval)"},
       "cross_pair": {}, "cross_split": {}, "langid": {}, "interpair_angles": {}, "spectrum": {}}

for model, extname in [("mbert", "mbert"), ("xlmr", "xlmr_base")]:
    ext = build_extractor(extname)
    data = {p: load_pair(ext, p, LAYER) for p in PAIRS}
    cent = {p: (E - E.mean(0), F - F.mean(0)) for p, (_, _, _, E, F) in data.items()}
    d = cent[REF_PAIR][0].shape[1]
    hb(f"{model}: loaded {len(PAIRS)} pairs (d={d})")

    cp = {p: {c: [] for c in ["baseline", "csls", "abtt_native", "abtt_transfer", "random_k10"]} for p in PAIRS}
    cp_extra = {p: {"angle_ref_native_mean": [], "angle_ref_native_max": [], "en_overlap_frac": [],
                    "transfer_gain_frac": [], "n_antihub_q": []} for p in PAIRS}
    cs = {p: {c: [] for c in ["baseline", "csls", "abtt_train", "abtt_insample", "random_k10"]} for p in PAIRS}
    cs_extra = {p: {"angle_train_test_mean": [], "angle_train_test_max": []} for p in PAIRS}
    li = {p: {"heldout_acc": [], "angle_to_hub": [], "langid_rec": [], "langid_skew": [],
              "abtt_k1_rec": [], "random_k1_rec": [], "baseline_rec": []} for p in PAIRS}
    ip_angles = {f"{a}|{b}": [] for i, a in enumerate(PAIRS) for b in PAIRS[i + 1:]}
    ip_random = []
    spec_shares = {p: [] for p in PAIRS}

    for seed in range(N_SEEDS):
        # --- reference subspace (en-es), same draw protocol as hub_ablation.py ---
        SRC_r = data[REF_PAIR][0]
        Ec_r, Fc_r = cent[REF_PAIR]
        rng_r = np.random.RandomState(seed)
        idx_r = rng_r.choice(len(SRC_r), min(M, len(SRC_r)), replace=False)
        Xr, Yr = unit(Ec_r[idx_r]), unit(Fc_r[idx_r])
        _, Vt_r = stacked_svd(Xr, Yr)
        Q_ref = Vt_r[:K]
        ref_words = {SRC_r[i] for i in idx_r}

        Q_nat_by_pair = {}
        for pair in PAIRS:
            SRC, TGT, gold, _, _ = data[pair]
            Ec, Fc = cent[pair]
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(SRC), min(M, len(SRC)), replace=False)
            src = [SRC[i] for i in idx]
            gal = [TGT[i] for i in idx]
            X, Y = unit(Ec[idx]), unit(Fc[idx])

            _, _, N1b, _ = retrieve(X, Y, gold, src, gal)
            gpos = {w: j for j, w in enumerate(gal)}
            anti_q = [q for q in range(len(src))
                      if any(gpos.get(t, -1) >= 0 and N1b[gpos[t]] == 0 for t in gold.get(src[q], set()))]
            strict = np.array([w not in ref_words for w in src])

            s_nat, Vt_nat = stacked_svd(X, Y)
            Q_nat = Vt_nat[:K]
            Q_nat_by_pair[pair] = Q_nat
            lam = s_nat ** 2
            spec_shares[pair].append((lam / lam.sum())[:20])

            cp[pair]["baseline"].append(metrics(X, Y, gold, src, gal, anti_q))
            cp[pair]["csls"].append(metrics(X, Y, gold, src, gal, anti_q, csls=True))
            Xa, Ya = project_out(X, Y, Q_nat)
            cp[pair]["abtt_native"].append(metrics(Xa, Ya, gold, src, gal, anti_q))
            Xt_, Yt_ = project_out(X, Y, Q_ref)
            cp[pair]["abtt_transfer"].append(metrics(Xt_, Yt_, gold, src, gal, anti_q, corr_mask=strict))
            Xn, Yn = project_out(X, Y, rng.randn(K, d))
            cp[pair]["random_k10"].append(metrics(Xn, Yn, gold, src, gal, anti_q))

            ang = principal_angles_deg(Q_ref, Q_nat)
            cp_extra[pair]["angle_ref_native_mean"].append(float(ang.mean()))
            cp_extra[pair]["angle_ref_native_max"].append(float(ang.max()))
            cp_extra[pair]["en_overlap_frac"].append(float(np.mean([w in ref_words for w in src])))
            cp_extra[pair]["n_antihub_q"].append(len(anti_q))
            rb = cp[pair]["baseline"][-1]["antihub_rec"]
            rn = cp[pair]["abtt_native"][-1]["antihub_rec"]
            rt = cp[pair]["abtt_transfer"][-1]["antihub_rec"]
            cp_extra[pair]["transfer_gain_frac"].append(
                float((rt - rb) / (rn - rb)) if abs(rn - rb) > 1e-9 else float("nan"))

            # --- language-ID control on the same draw ---
            rng_l = np.random.RandomState(200 + seed)
            Z = np.vstack([X, Y])
            Zc = Z - Z.mean(0)
            yl = np.array([0] * len(X) + [1] * len(Y))
            perm = rng_l.permutation(len(Zc))
            ntr = int(0.75 * len(Zc))
            tr, te = perm[:ntr], perm[ntr:]
            clf = LogisticRegression(max_iter=1000).fit(Zc[tr], yl[tr])
            li[pair]["heldout_acc"].append(float(clf.score(Zc[te], yl[te])))
            w = clf.coef_.ravel()
            w = w / (np.linalg.norm(w) + 1e-12)
            li[pair]["angle_to_hub"].append(float(np.degrees(np.arccos(np.clip(np.linalg.norm(Q_nat @ w), 0, 1)))))
            Xl, Yl = project_out(X, Y, w[None, :])
            ml = metrics(Xl, Yl, gold, src, gal, anti_q)
            li[pair]["langid_rec"].append(ml["antihub_rec"])
            li[pair]["langid_skew"].append(ml["skew"])
            X1, Y1 = project_out(X, Y, Q_nat[:1])
            li[pair]["abtt_k1_rec"].append(metrics(X1, Y1, gold, src, gal, anti_q)["antihub_rec"])
            Xr1, Yr1 = project_out(X, Y, rng_l.randn(1, d))
            li[pair]["random_k1_rec"].append(metrics(Xr1, Yr1, gold, src, gal, anti_q)["antihub_rec"])
            li[pair]["baseline_rec"].append(rb)

            # --- cross-split (train/test disjoint words; train-only centering AND subspace) ---
            SRC_f, TGT_f, gold_f, E_raw, F_raw = data[pair]
            rng_s = np.random.RandomState(100 + seed)
            perm_s = rng_s.permutation(len(SRC_f))
            nh = min(M, len(SRC_f) // 2)
            train, test = perm_s[:nh], perm_s[nh:2 * nh]
            muE, muF = E_raw[train].mean(0), F_raw[train].mean(0)
            Xtr, Ytr = unit(E_raw[train] - muE), unit(F_raw[train] - muF)
            Xte, Yte = unit(E_raw[test] - muE), unit(F_raw[test] - muF)
            _, Vt_tr = stacked_svd(Xtr, Ytr)
            Q_train = Vt_tr[:K]
            _, Vt_te = stacked_svd(Xte, Yte)
            Q_insample = Vt_te[:K]
            src_te = [SRC_f[i] for i in test]
            gal_te = [TGT_f[i] for i in test]
            _, _, N1b_te, _ = retrieve(Xte, Yte, gold_f, src_te, gal_te)
            gpos_te = {w_: j for j, w_ in enumerate(gal_te)}
            anti_te = [q for q in range(len(src_te))
                       if any(gpos_te.get(t, -1) >= 0 and N1b_te[gpos_te[t]] == 0
                              for t in gold_f.get(src_te[q], set()))]
            cs[pair]["baseline"].append(metrics(Xte, Yte, gold_f, src_te, gal_te, anti_te))
            cs[pair]["csls"].append(metrics(Xte, Yte, gold_f, src_te, gal_te, anti_te, csls=True))
            Xa2, Ya2 = project_out(Xte, Yte, Q_train)
            cs[pair]["abtt_train"].append(metrics(Xa2, Ya2, gold_f, src_te, gal_te, anti_te))
            Xa3, Ya3 = project_out(Xte, Yte, Q_insample)
            cs[pair]["abtt_insample"].append(metrics(Xa3, Ya3, gold_f, src_te, gal_te, anti_te))
            Xn2, Yn2 = project_out(Xte, Yte, rng_s.randn(K, d))
            cs[pair]["random_k10"].append(metrics(Xn2, Yn2, gold_f, src_te, gal_te, anti_te))
            ang_tt = principal_angles_deg(Q_train, Q_insample)
            cs_extra[pair]["angle_train_test_mean"].append(float(ang_tt.mean()))
            cs_extra[pair]["angle_train_test_max"].append(float(ang_tt.max()))

        # --- inter-pair principal angles + random-subspace reference ---
        for i, a in enumerate(PAIRS):
            for b in PAIRS[i + 1:]:
                ip_angles[f"{a}|{b}"].append(float(principal_angles_deg(Q_nat_by_pair[a], Q_nat_by_pair[b]).mean()))
        rng_d = np.random.RandomState(300 + seed)
        Qr1 = np.linalg.qr(rng_d.randn(d, K))[0].T
        Qr2 = np.linalg.qr(rng_d.randn(d, K))[0].T
        ip_random.append(float(principal_angles_deg(Qr1, Qr2).mean()))
        hb(f"{model} seed {seed} done")

    out["cross_pair"][model] = {
        p: {**{c: agg(cp[p][c]) for c in cp[p]},
            **{k: ci(v) for k, v in cp_extra[p].items()}} for p in PAIRS}
    out["cross_split"][model] = {
        p: {**{c: agg(cs[p][c]) for c in cs[p]},
            **{k: ci(v) for k, v in cs_extra[p].items()}} for p in PAIRS}
    out["langid"][model] = {p: {k: ci(v) for k, v in li[p].items()} for p in PAIRS}
    out["interpair_angles"][model] = {
        "per_combo_mean_deg": {k: ci(v) for k, v in ip_angles.items()},
        "overall_mean_deg": ci([x for v in ip_angles.values() for x in v]),
        "random_subspace_ref_deg": ci(ip_random)}
    out["spectrum"][model] = {}
    for p in PAIRS:
        shares = np.mean(np.stack(spec_shares[p]), axis=0)
        kstar = {str(tau): int(max([k + 1 for k in range(20) if shares[k] > tau / d] or [0]))
                 for tau in [3, 5, 8, 10, 15, 20]}
        out["spectrum"][model][p] = {"marginal_share_top20": [float(x) for x in shares],
                                     "kstar_at_tau": kstar, "d": int(d)}

    for p in PAIRS:
        r = out["cross_pair"][model][p]
        print(f"[{model} {p}] antihub base {r['baseline']['antihub_rec'][0]:.3f} | native "
              f"{r['abtt_native']['antihub_rec'][0]:.3f} | TRANSFER(en-es) "
              f"{r['abtt_transfer']['antihub_rec'][0]:.3f} (strict "
              f"{r['abtt_transfer']['antihub_rec_strict'][0]:.3f}, en-overlap "
              f"{r['en_overlap_frac'][0]:.2f}) | rand {r['random_k10']['antihub_rec'][0]:.3f} | CSLS "
              f"{r['csls']['antihub_rec'][0]:.3f} | angle(ref,nat) {r['angle_ref_native_mean'][0]:.1f} deg | "
              f"gain-frac {r['transfer_gain_frac'][0]:.2f}", flush=True)
        rs = out["cross_split"][model][p]
        print(f"          split: base {rs['baseline']['antihub_rec'][0]:.3f} | TRAIN-fit "
              f"{rs['abtt_train']['antihub_rec'][0]:.3f} | in-sample "
              f"{rs['abtt_insample']['antihub_rec'][0]:.3f} | rand {rs['random_k10']['antihub_rec'][0]:.3f} | "
              f"angle(tr,te) {rs['angle_train_test_mean'][0]:.1f} deg", flush=True)
        rl = out["langid"][model][p]
        print(f"          langid: acc {rl['heldout_acc'][0]:.3f} | angle-to-hub {rl['angle_to_hub'][0]:.1f} deg | "
              f"ablate {rl['langid_rec'][0]:.3f} vs rand-k1 {rl['random_k1_rec'][0]:.3f} vs ABTT-k1 "
              f"{rl['abtt_k1_rec'][0]:.3f} (base {rl['baseline_rec'][0]:.3f})", flush=True)
    print(f"== {model}: inter-pair subspace angle {out['interpair_angles'][model]['overall_mean_deg'][0]:.1f} deg "
          f"vs random-subspace {out['interpair_angles'][model]['random_subspace_ref_deg'][0]:.1f} deg", flush=True)

# --- optional L4 spot-check (mBERT en-es; may trigger new forward passes) ---
if RUN_L4:
    hb("L4 spot-check (mBERT en-es, layer 4) ...")
    ext = build_extractor("mbert")
    SRC, TGT, gold, E4_, F4_ = load_pair(ext, "en-es", 4)
    Ec, Fc = E4_ - E4_.mean(0), F4_ - F4_.mean(0)
    d = Ec.shape[1]
    conds = {c: [] for c in ["baseline", "csls", "abtt_native", "random_k10"]}
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(SRC), min(M, len(SRC)), replace=False)
        src = [SRC[i] for i in idx]
        gal = [TGT[i] for i in idx]
        X, Y = unit(Ec[idx]), unit(Fc[idx])
        _, _, N1b, _ = retrieve(X, Y, gold, src, gal)
        gpos = {w: j for j, w in enumerate(gal)}
        anti_q = [q for q in range(len(src))
                  if any(gpos.get(t, -1) >= 0 and N1b[gpos[t]] == 0 for t in gold.get(src[q], set()))]
        _, Vt = stacked_svd(X, Y)
        conds["baseline"].append(metrics(X, Y, gold, src, gal, anti_q))
        conds["csls"].append(metrics(X, Y, gold, src, gal, anti_q, csls=True))
        Xa, Ya = project_out(X, Y, Vt[:K])
        conds["abtt_native"].append(metrics(Xa, Ya, gold, src, gal, anti_q))
        Xn, Yn = project_out(X, Y, rng.randn(K, d))
        conds["random_k10"].append(metrics(Xn, Yn, gold, src, gal, anti_q))
    out["L4_spotcheck"] = {"model": "mbert", "pair": "en-es", "layer": 4,
                           **{c: agg(v) for c, v in conds.items()}}
    r = out["L4_spotcheck"]
    print(f"[L4 mbert en-es] antihub base {r['baseline']['antihub_rec'][0]:.3f} -> ABTT "
          f"{r['abtt_native']['antihub_rec'][0]:.3f} (rand {r['random_k10']['antihub_rec'][0]:.3f}, "
          f"CSLS {r['csls']['antihub_rec'][0]:.3f}); P@1 {r['baseline']['p1'][0]:.3f} -> "
          f"{r['abtt_native']['p1'][0]:.3f}", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/E4_transfer.json", "w") as f:
    json.dump(out, f, indent=2)
print("E4_SUBSPACE_TRANSFER DONE", flush=True)
