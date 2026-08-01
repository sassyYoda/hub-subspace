"""Does the de-hubbing ablation replicate on a decoder model (Llama-3-8B-4bit)?

Runs on cached Llama layer-16 last-token activations to test that the hub-localization mechanism is
not specific to encoders / mBERT. The activation cache is produced by experiments/decoder_extract.py.
Includes a random-direction control and reports 5-seed CIs.

Run: python experiments/decoder_ablation.py
Env: PAIRS (default en-es,en-de), M (default 1500), N_SEEDS (default 5)
Output: results/decoder_ablation.json
"""
from __future__ import annotations
import os, json, time
import numpy as np
from data.muse import load_muse_dictionary, build_gold_map
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from scipy.stats import t as _student_t
PAIRS=os.environ.get("PAIRS","en-es,en-de").split(",")
M=int(os.environ.get("M","1500"))
N_SEEDS=int(os.environ.get("N_SEEDS","5"))
_T0=time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
def unit(A): return A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
def load(lang):
    d=np.load(f"results/cache/llama3_8b_4bit/{lang}_L16_last.npz",allow_pickle=True)
    return {w:v for w,v in zip(d['words'],d['vecs'])}
def project_out(A,B,dirs):
    Q=np.linalg.qr(dirs.T)[0].T
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q
def topk_pca(X,Y,k):
    Z=np.vstack([X,Y])
    Zc=Z-Z.mean(0)
    _,_,Vt=np.linalg.svd(Zc,full_matrices=False)
    return Vt[:k]
def retrieve(X,Y,gold,src,gal):
    S=cosine_similarity(unit(X),unit(Y))
    pred=S.argmax(1)
    corr=np.array([1 if gal[pred[j]] in gold.get(src[j],set()) else 0 for j in range(len(src))])
    return float(corr.mean()),pred,np.bincount(pred,minlength=len(gal)),corr
out={"results":{}}
for pair in PAIRS:
    sl_,tl_=pair.split("-")
    allp=load_muse_dictionary(pair)
    gold=build_gold_map(allp)
    en=load(sl_)
    xx=load(tl_)
    ss=set()
    tt=set()
    rows=[]
    for s,t in allp:
        if s in ss or t in tt or s not in en or t not in xx: continue
        ss.add(s);tt.add(t);rows.append((s,t))
    EN=np.stack([en[s] for s,_ in rows])
    XX=np.stack([xx[t] for _,t in rows])
    EN=EN-EN.mean(0)
    XX=XX-XX.mean(0)
    hb(f"[{pair}] n={len(rows)} dim={EN.shape[1]}")
    conds=["centered(base)","ABTT-k3","ABTT-k5","ABTT-k10","random-k10"]
    agg={k:{"p1":[],"maxhub":[],"skew":[],"antihub":[]} for k in conds}
    for seed in range(N_SEEDS):
        rng=np.random.RandomState(seed)
        idx=rng.choice(len(rows),min(M,len(rows)),replace=False)
        src=[rows[i][0] for i in idx]
        gal=[rows[i][1] for i in idx]
        X=EN[idx].astype(np.float64)
        Y=XX[idx].astype(np.float64)
        # anti-hub queries: gold target never retrieved at rank 1 in the base run
        p1b,_,N1b,_=retrieve(X,Y,gold,src,gal)
        gpos={w:j for j,w in enumerate(gal)}
        anti_q=[q for q in range(len(src)) if any(gpos.get(t,-1)>=0 and N1b[gpos[t]]==0 for t in gold.get(src[q],set()))]
        def rec(Xa,Ya):
            p1,_,N1,corr=retrieve(Xa,Ya,gold,src,gal)
            r=float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
            return p1,int(N1.max()),float(skew(N1)),r
        for name,dirs in [("centered(base)",None),("ABTT-k3",topk_pca(X,Y,3)),("ABTT-k5",topk_pca(X,Y,5)),("ABTT-k10",topk_pca(X,Y,10)),("random-k10",rng.randn(10,X.shape[1]))]:
            Xa,Ya=(X,Y) if dirs is None else project_out(X,Y,dirs)
            for kk,vv in zip(["p1","maxhub","skew","antihub"],rec(Xa,Ya)): agg[name][kk].append(vv)
        hb(f"  seed {seed} (n_antihub_q={len(anti_q)})")
    def ci(v):
        v=np.array(v,float)
        v=v[~np.isnan(v)]
        return (float(v.mean()),float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0)
    print(f"\n[{pair}] Llama-3-8B L16 (decoder) raw cross-lingual NN, {N_SEEDS} seeds (mean±95%CI)", flush=True)
    print(f"  {'condition':<16}{'P@1':>13}{'max-hub':>9}{'skew':>8}{'antihub-rec':>14}", flush=True)
    for cd in conds:
        r={k:ci(agg[cd][k]) for k in agg[cd]}
        print(f"  {cd:<16}{r['p1'][0]:>7.3f}±{r['p1'][1]:.3f}{r['maxhub'][0]:>9.0f}{r['skew'][0]:>8.2f}{r['antihub'][0]:>9.3f}±{r['antihub'][1]:.3f}", flush=True)
    out["results"][pair]={cd:{k:ci(agg[cd][k]) for k in agg[cd]} for cd in conds}
os.makedirs("results",exist_ok=True)
with open("results/decoder_ablation.json","w") as f:
    json.dump(out,f,indent=2)
print("\nDECODER_ABLATION DONE", flush=True)
