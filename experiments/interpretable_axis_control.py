"""Are the ablated hub directions the interpretable tokenization-artifact axes
(subword fertility, character length)?

Three parts:
(a) R^2 of predicting fertility/length from top-PC projections.
(b) Geometric overlap: cosine(PC_i, fertility/length axis), plus principal angles between
    span{top-2 PCs} (the hub subspace we ablate) and span{fertility-axis, length-axis}.
(c) Causal check: ablate the explicitly computed {fertility, length} axes and test whether this
    reproduces the de-hubbing and anti-hub recovery of top-2-PC ablation, against a random-2 control.
mBERT layer 1, 5 seeds.

Run: python experiments/interpretable_axis_control.py
Env: PAIRS (default en-es,en-zh), M (default 2000), N_SEEDS (default 5)
Output: results/interpretable_axis.json
"""
from __future__ import annotations
import os, json, time
import numpy as np
from core.centering import mean_center
from data.muse import load_muse_dictionary, build_gold_map
from extract.context_pool import IsolationSpec
from extract.registry import build_extractor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from scipy.stats import t as _student_t
PAIRS=os.environ.get("PAIRS","en-es,en-zh").split(",")
M=int(os.environ.get("M","2000"))
N_SEEDS=int(os.environ.get("N_SEEDS","5"))
_T0=time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ext=build_extractor("mbert")
ctx=IsolationSpec()
tok=ext.tokenizer
def unit(A): return A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
def ridge_axis(X,y,lam=1.0):
    Xc=X-X.mean(0)
    yc=y-y.mean()
    w=np.linalg.solve(Xc.T@Xc+lam*np.eye(Xc.shape[1]), Xc.T@yc)
    pred=Xc@w
    r2=1-((yc-pred)**2).sum()/((yc-yc.mean())**2).sum()
    return w/(np.linalg.norm(w)+1e-9), float(r2)
def princ_angles(B1,B2):
    Q1=np.linalg.qr(B1.T)[0]
    Q2=np.linalg.qr(B2.T)[0]
    s=np.clip(np.linalg.svd(Q1.T@Q2,compute_uv=False),-1,1)
    return np.degrees(np.arccos(s))
def project_out(A,B,dirs):
    Q=np.linalg.qr(dirs.T)[0].T
    return A-(A@Q.T)@Q, B-(B@Q.T)@Q
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
    ss=set()
    tt=set()
    rows=[]
    for s,t in allp:
        if s in ss or t in tt: continue
        ss.add(s);tt.add(t);rows.append((s,t))
        if len(rows)>=4000: break
    SRC=[s for s,_ in rows]
    TGT=[t for _,t in rows]
    E=ext.extract(SRC,sl_,layers=[1],context_pool=ctx,subword_agg="mean")[1]
    F=ext.extract(TGT,tl_,layers=[1],context_pool=ctx,subword_agg="mean")[1]
    c=mean_center({sl_:E,tl_:F},mode="per_layer_per_lang")
    E,F=c[sl_],c[tl_]
    fert=np.array([len(tok.tokenize(w)) for w in SRC]+[len(tok.tokenize(w)) for w in TGT],float)
    clen=np.array([len(w) for w in SRC]+[len(w) for w in TGT],float)
    hb(f"[{pair}] extracted {len(rows)}")
    A=dict(r2f_pc0=[],r2l_pc1=[],cos_pc0_fert=[],cos_pc1_len=[],ang_subspace=[])
    abl={k:{"p1":[],"maxhub":[],"skew":[],"antihub":[]} for k in ["base","ablate-top2PC","ablate-fert+len","random-2"]}
    for seed in range(N_SEEDS):
        rng=np.random.RandomState(seed)
        idx=rng.choice(len(rows),min(M,len(rows)),replace=False)
        src=[SRC[i] for i in idx]
        gal=[TGT[i] for i in idx]
        X=E[idx].astype(np.float64)
        Y=F[idx].astype(np.float64)
        Z=np.vstack([X,Y])
        fz=np.concatenate([fert[idx],fert[np.array(idx)+len(rows)]])
        lz=np.concatenate([clen[idx],clen[np.array(idx)+len(rows)]])
        Zc=Z-Z.mean(0)
        _,_,Vt=np.linalg.svd(Zc,full_matrices=False)
        pc0,pc1=Vt[0],Vt[1]
        p0=Zc@pc0
        p1=Zc@pc1
        r2f=1-((fz-fz.mean()-(np.polyfit(p0,fz,1)[0]*p0))**2).sum()/((fz-fz.mean())**2).sum()
        r2l=1-((lz-lz.mean()-(np.polyfit(p1,lz,1)[0]*p1))**2).sum()/((lz-lz.mean())**2).sum()
        fax,_=ridge_axis(Z,fz)
        lax,_=ridge_axis(Z,lz)
        A["r2f_pc0"].append(r2f)
        A["r2l_pc1"].append(r2l)
        A["cos_pc0_fert"].append(abs(float(pc0@fax)))
        A["cos_pc1_len"].append(abs(float(pc1@lax)))
        A["ang_subspace"].append(float(princ_angles(np.vstack([pc0,pc1]),np.vstack([fax,lax])).mean()))
        # anti-hub queries: source words whose gold target is never retrieved at rank 1 in the base run
        p1b,_,N1b,_=retrieve(X,Y,gold,src,gal)
        gpos={w:j for j,w in enumerate(gal)}
        anti_q=[q for q in range(len(src)) if any(gpos.get(t,-1)>=0 and N1b[gpos[t]]==0 for t in gold.get(src[q],set()))]
        def rec(Xa,Ya):
            p1,_,N1,corr=retrieve(Xa,Ya,gold,src,gal)
            r=float(np.mean([corr[q] for q in anti_q])) if anti_q else float("nan")
            return p1,int(N1.max()),float(skew(N1)),r
        for name,dirs in [("base",None),("ablate-top2PC",np.vstack([pc0,pc1])),("ablate-fert+len",np.vstack([fax,lax])),("random-2",rng.randn(2,X.shape[1]))]:
            Xa,Ya=(X,Y) if dirs is None else project_out(X,Y,dirs)
            m=rec(Xa,Ya)
            for kk,vv in zip(["p1","maxhub","skew","antihub"],m): abl[name][kk].append(vv)
        hb(f"  seed {seed} done")
    def ci(v):
        v=np.array(v,float)
        v=v[~np.isnan(v)]
        return (float(v.mean()),float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0)
    print(f"\n[{pair}] INTERPRETABILITY of hub directions (mean±95%CI, {N_SEEDS} seeds)", flush=True)
    print(f"  R^2(fertility|PC0)={ci(A['r2f_pc0'])[0]:.2f}±{ci(A['r2f_pc0'])[1]:.2f}  R^2(length|PC1)={ci(A['r2l_pc1'])[0]:.2f}±{ci(A['r2l_pc1'])[1]:.2f}", flush=True)
    print(f"  |cos(PC0,fertility-axis)|={ci(A['cos_pc0_fert'])[0]:.2f}  |cos(PC1,length-axis)|={ci(A['cos_pc1_len'])[0]:.2f}  mean principal angle(span top2PC, span fert+len)={ci(A['ang_subspace'])[0]:.1f}deg", flush=True)
    print(f"  Causal: does ablating the interpretable axes reproduce hub ablation?", flush=True)
    print(f"  {'condition':<18}{'P@1':>13}{'max-hub':>9}{'skew':>8}{'antihub-rec':>14}", flush=True)
    for name in ["base","ablate-top2PC","ablate-fert+len","random-2"]:
        r={k:ci(abl[name][k]) for k in abl[name]}
        print(f"  {name:<18}{r['p1'][0]:>7.3f}±{r['p1'][1]:.3f}{r['maxhub'][0]:>9.0f}{r['skew'][0]:>8.2f}{r['antihub'][0]:>9.3f}±{r['antihub'][1]:.3f}", flush=True)
    out["results"][pair]={"interp":{k:ci(A[k]) for k in A},"ablation":{n:{k:ci(abl[n][k]) for k in abl[n]} for n in abl}}
os.makedirs("results",exist_ok=True)
with open("results/interpretable_axis.json","w") as f:
    json.dump(out,f,indent=2)
print("\nINTERPRETABLE_AXIS DONE", flush=True)
