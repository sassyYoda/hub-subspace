"""Does ablating the hub subspace change a real cross-lingual behavior?

Few-shot word translation via Llama-3-8B generation. We patch out the hub directions at layer 16
during the forward pass and measure translation accuracy against a random-direction control and an
unablated baseline, over multiple seeds. The test is whether hub-subspace ablation changes accuracy
more than an equal-rank random ablation, which would mean the behavior is causally sensitive to the
hub subspace. The hub subspace comes from the cached layer-16 activations produced by
experiments/decoder_extract.py.

Run: python experiments/downstream_patch.py
Env: N (default 150), KDIR (default 10), NSHOT (default 5), N_SEEDS (default 3),
     SMOKE=1 for a quick dry run that skips writing output
Output: results/downstream_patch.json
"""
from __future__ import annotations
import os, json, time
import numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from data.muse import load_muse_dictionary
from scipy.stats import t as _student_t
SMOKE=os.environ.get("SMOKE","0")=="1"
N=int(os.environ.get("N","150"))
KDIR=int(os.environ.get("KDIR","10"))
NSHOT=int(os.environ.get("NSHOT","5"))
N_SEEDS=int(os.environ.get("N_SEEDS","3"))
_T0=time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
hb("loading Llama")
model,tok=load("mlx-community/Meta-Llama-3-8B-4bit")
layers=model.model.layers
L=16
tie=getattr(model.args,"tie_word_embeddings",False)
hb(f"loaded ({len(layers)} layers, tie={tie})")

# hub directions from the cached layer-16 last-token word activations
def loadc(lang):
    d=np.load(f"results/cache/llama3_8b_4bit/{lang}_L16_last.npz",allow_pickle=True)
    return {w:v for w,v in zip(d['words'],d['vecs'])}
en_c=loadc("en")
es_c=loadc("es")
allp=load_muse_dictionary("en-es")
pairs=[(s,t) for s,t in allp if s in en_c and t in es_c]
ss=set()
tt=set()
P=[]
for s,t in pairs:
    if s in ss or t in tt: continue
    ss.add(s);tt.add(t);P.append((s,t))
A=np.stack([en_c[s] for s,_ in P]+[es_c[t] for _,t in P])
# Unit-normalize before PCA, as the encoder analyses do. On raw Llama activations the top PCs are the
# few massive-norm outlier dimensions; projecting those out collapses generation. Set NORM=0 to see that.
NORM=os.environ.get("NORM","1")=="1"
if NORM:
    A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
A=A-A.mean(0)
_,_,Vt=np.linalg.svd(A,full_matrices=False)
HUB=mx.array(Vt[:KDIR].astype(np.float32))      # (k, D) hub directions
hb(f"hub dirs from {len(P)} pairs, k={KDIR}")

def forward_logits(ids, ablate=None):
    h=model.model.embed_tokens(ids[None])
    mask=create_attention_mask(h, cache=None)
    for i,layer in enumerate(layers):
        h=layer(h, mask=mask, cache=None)
        if i==L and ablate is not None:
            Q=ablate                                   # (k,D) orthonormal rows
            h=h-(h@Q.T)@Q
    h=model.model.norm(h)
    logits=model.model.embed_tokens.as_linear(h) if tie else model.lm_head(h)
    return logits[0,-1]

def forward_all(ids, ablate=None):
    """Logits at every position. The per-position logits are computed anyway; the
    next-token path just discards them, so scoring a whole target costs one pass."""
    h=model.model.embed_tokens(ids[None])
    mask=create_attention_mask(h, cache=None)
    for i,layer in enumerate(layers):
        h=layer(h, mask=mask, cache=None)
        if i==L and ablate is not None:
            Q=ablate
            h=h-(h@Q.T)@Q
    h=model.model.norm(h)
    return (model.model.embed_tokens.as_linear(h) if tie else model.lm_head(h))[0]

def target_ids(word):
    t=tok.encode(" "+word)
    return t[1:] if len(t)>1 else t

def score_target(prompt_ids, tgt_ids, ablate=None):
    """Teacher-forced mean log-prob per target token, and whether every target
    token is the argmax given its gold prefix (full-word exact match)."""
    full=mx.array([int(x) for x in np.array(prompt_ids)]+[int(x) for x in tgt_ids])
    lg=forward_all(full, ablate=ablate)
    n_p=int(np.array(prompt_ids).size)
    lps, allmax = [], True
    for j,tid in enumerate(tgt_ids):
        row=lg[n_p-1+j]
        row=row-mx.logsumexp(row)
        lps.append(float(row[tid]))
        if int(np.argmax(np.array(row)))!=tid: allmax=False
    return float(np.mean(lps)), (1.0 if allmax else 0.0)

def build_prompt(shots, query):
    s="".join([f"English: {a}\nSpanish: {b}\n" for a,b in shots])
    return s+f"English: {query}\nSpanish:"

def first_tok(word):
    t=tok.encode(" "+word)
    return t[1] if len(t)>1 else (t[0] if t else -1)   # skip BOS

def orth(M):
    Q=mx.array(np.linalg.qr(np.array(M).T)[0].T.astype(np.float32))
    return Q
HUBo=orth(HUB)

def logsoftmax(lg):
    lg=np.array(lg,dtype=np.float64)
    m=lg.max()
    return lg-m-np.log(np.exp(lg-m).sum())
test_pool=P[:max(N*3,500)]
# per condition: mean logprob(gold tok), mean reciprocal rank, top-1 acc, top-5 acc, median gold rank
metr={c:{"lp":[],"mrr":[],"acc":[],"acc5":[],"rank_median":[],"seq_lp":[],"exact":[]} for c in ["baseline","hub","random"]}
for seed in range(1 if SMOKE else N_SEEDS):
    rs=np.random.RandomState(seed)
    test=[test_pool[i] for i in rs.choice(len(test_pool), min(N,len(test_pool)), replace=False)]
    randdirs=orth(mx.array(rs.randn(KDIR, HUB.shape[1]).astype(np.float32)))
    acc={c:[] for c in metr}
    acc5={c:[] for c in metr}
    ranks={c:[] for c in metr}
    lp={c:[] for c in metr}
    mrr={c:[] for c in metr}
    seqlp={c:[] for c in metr}
    exact={c:[] for c in metr}
    for qi,(qs,qt) in enumerate(test):
        shots=[test_pool[i] for i in rs.choice(len(test_pool),NSHOT,replace=False) if test_pool[i][0]!=qs][:NSHOT]
        ids=mx.array(tok.encode(build_prompt(shots,qs)))
        gt=first_tok(qt)
        if gt<0: continue
        tids=target_ids(qt)
        for cond,ab in [("baseline",None),("hub",HUBo),("random",randdirs)]:
            lg=forward_logits(ids, ablate=ab)
            ls=logsoftmax(lg)
            pred=int(np.argmax(ls))
            rank=int((ls>ls[gt]).sum())+1
            lp[cond].append(float(ls[gt])); mrr[cond].append(1.0/rank); acc[cond].append(1.0 if pred==gt else 0.0)
            acc5[cond].append(1.0 if rank<=5 else 0.0); ranks[cond].append(rank)
            slp,ex=score_target(ids, tids, ablate=ab)
            seqlp[cond].append(slp); exact[cond].append(ex)
        if SMOKE and qi>=9: break
    for c in metr:
        metr[c]["lp"].append(float(np.mean(lp[c]))); metr[c]["mrr"].append(float(np.mean(mrr[c]))); metr[c]["acc"].append(float(np.mean(acc[c])))
        metr[c]["acc5"].append(float(np.mean(acc5[c]))); metr[c]["rank_median"].append(float(np.median(ranks[c])))
        metr[c]["seq_lp"].append(float(np.mean(seqlp[c]))); metr[c]["exact"].append(float(np.mean(exact[c])))
    hb(f"seed {seed}: lp base={metr['baseline']['lp'][-1]:.2f} hub={metr['hub']['lp'][-1]:.2f} rand={metr['random']['lp'][-1]:.2f} | "
       f"acc1 {metr['baseline']['acc'][-1]:.3f}/{metr['hub']['acc'][-1]:.3f}/{metr['random']['acc'][-1]:.3f} | "
       f"acc5 {metr['baseline']['acc5'][-1]:.3f}/{metr['hub']['acc5'][-1]:.3f}/{metr['random']['acc5'][-1]:.3f}")
    if SMOKE: break
def ci(v):
    v=np.array(v,float)
    return (float(v.mean()), float(_student_t.ppf(0.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.0)
print(f"\n[Llama few-shot en->es translation; L16 hub-subspace ablation k={KDIR}; {len(metr['baseline']['lp'])} seeds, N={N}]", flush=True)
print(f"  {'condition':<10}{'logprob(gold)':>16}{'MRR':>14}{'top1-acc':>14}{'top5-acc':>14}{'med-rank':>12}", flush=True)
for c in ["baseline","hub","random"]:
    lpm=ci(metr[c]['lp']); mr=ci(metr[c]['mrr']); ac=ci(metr[c]['acc']); ac5=ci(metr[c]['acc5']); rkm=ci(metr[c]['rank_median'])
    print(f"  {c:<10}{lpm[0]:>10.3f}±{lpm[1]:.3f}{mr[0]:>9.3f}±{mr[1]:.3f}{ac[0]:>9.3f}±{ac[1]:.3f}{ac5[0]:>9.3f}±{ac5[1]:.3f}{rkm[0]:>8.1f}±{rkm[1]:.1f}", flush=True)
# Full-word metrics. top1-acc above scores only the FIRST target token; these score
# every token of the gold target, so a word whose first token is right but whose
# remainder is wrong no longer counts as correct.
print(f"  {'condition':<10}{'seq-logprob/tok':>18}{'full-word-exact':>18}", flush=True)
for c in ["baseline","hub","random"]:
    sl=ci(metr[c]['seq_lp']); ex=ci(metr[c]['exact'])
    print(f"  {c:<10}{sl[0]:>12.3f}±{sl[1]:.3f}{ex[0]:>12.3f}±{ex[1]:.3f}", flush=True)
print("  Expectation: hub-ablation should change logprob/MRR/accuracy more than random-ablation (causal sensitivity)", flush=True)
# paired (within-seed) hub-vs-random differences control for seed and word-sample variance
for name in ["lp","mrr","acc","acc5","seq_lp","exact"]:
    d=np.array(metr['hub'][name])-np.array(metr['random'][name])
    ci_=_student_t.ppf(0.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else 0.0
    print(f"  PAIRED hub-random Δ{name}={d.mean():+.4f}±{ci_:.4f} (sign {int((d>0).sum())}/{len(d)})", flush=True)
# median gold-token rank moves down (better) under hub vs random
drk=np.array(metr['hub']['rank_median'])-np.array(metr['random']['rank_median'])
print(f"  PAIRED hub-random Δmedian-rank={drk.mean():+.2f} (sign neg {int((drk<0).sum())}/{len(drk)}; negative = hub improves rank)", flush=True)
if not SMOKE:
    os.makedirs("results",exist_ok=True)
    with open("results/downstream_patch.json","w") as f:
        json.dump(metr,f,indent=2)
print("DOWNSTREAM_PATCH DONE", flush=True)
