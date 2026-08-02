"""Distinctiveness control for the length-vs-accuracy curve.

Tests whether the rise in P@1 with input length comes from coherent context resolving
ambiguity, or simply from longer text being more unique (distinctiveness). For k in
{1,3,5,full} it compares bitext P@1 across three constructions at matched length:
  coherent  : first-k words of the parallel sentence (real context)
  shuffled  : the same k words in scrambled order (removes syntax, keeps content and length)
  randombag : k words each from a different random sentence (removes coherence; pure length)
It also reports within-language distinctiveness (mean max off-diagonal cosine; lower means
more unique) to quantify the confound directly. If randombag rises about as fast as coherent,
the gain is aggregation/distinctiveness rather than comprehension; if coherent stays well above
randombag, coherent context carries real signal. mBERT, masked-mean pooling at layer 8.

Run: python experiments/length_confound_control.py
"""
from __future__ import annotations
import os, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
_T0=time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
ROOT="data/flores200/flores200_dataset/devtest"
CODE={"en":"eng_Latn","es":"spa_Latn","zh":"zho_Hans"}
def load_sents(lang,n=1012):
    with open(f"{ROOT}/{CODE[lang]}.devtest") as f:
        return [l.strip() for l in f][:n]
dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model=AutoModel.from_pretrained("bert-base-multilingual-cased").to(dev).eval()
hb(f"model {dev}")
@torch.no_grad()
def embed(sents):
    out=[]
    for i in range(0,len(sents),32):
        enc=tok(sents[i:i+32],padding=True,truncation=True,max_length=64,return_tensors="pt").to(dev)
        hs=model(**enc).hidden_states[8] if model.config.output_hidden_states else model(**enc,output_hidden_states=True).hidden_states[8]
        m=enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out,0)

# FLORES Chinese is unsegmented, so s.split() leaves the target side untruncated.
# Segment per language; SEG=whitespace restores the old behaviour for comparison.
SEG = os.environ.get("SEG", "aware")
NOSPACE = {"zh"}

def segment(lang, sent):
    if SEG == "whitespace" or lang not in NOSPACE:
        return sent.split()
    import jieba
    return [w for w in jieba.cut(sent) if w.strip()]

def joiner(lang):
    return "" if (SEG != "whitespace" and lang in NOSPACE) else " "

def p1_and_distinct(en_items,xx_items):
    X=embed(en_items)
    Y=embed(xx_items)
    X=X-X.mean(0)
    Y=Y-Y.mean(0)
    n=len(X)
    rng=np.random.RandomState(0)
    test=rng.permutation(n)[300:]
    cos=cosine_similarity(X[test],Y[test])
    p1=float((cos.argmax(1)==np.arange(len(test))).mean())
    # Within-English confusability: mean of the max off-diagonal cosine. Higher means more
    # confusable (less unique). Drop the diagonal first so self-similarity is excluded.
    Se=cosine_similarity(X[test])
    np.fill_diagonal(Se,-9)
    confus=float(Se.max(1).mean())
    return p1, confus

for tgt in ["es","zh"]:
    en=load_sents("en")
    xx=load_sents(tgt)
    en_w=[segment("en", s) for s in en]
    xx_w=[segment(tgt, s) for s in xx]
    N=len(en)
    rng=np.random.RandomState(0)
    print(f"\n[en-{tgt}] mBERT L8 bitext P@1 (test=712) / within-en confusability", flush=True)
    print(f"  {'k':>4} {'coherent':>18} {'shuffled':>18} {'randombag':>18}", flush=True)
    for k in [1,3,5,99]:
        def build(words_list, mode, lang="en"):
            items=[]
            for i in range(N):
                w=words_list[i]
                if mode=="coherent":
                    it=w[:k]
                elif mode=="shuffled":
                    it=list(w[:k]); rng.shuffle(it)
                else:
                    # randombag: first word of k random sentences. The sentence indices are
                    # shared across languages so the bags stay parallel.
                    js=randidx[i]
                    it=[words_list[j][0] for j in js if words_list[j]]
                items.append(joiner(lang).join(it) if it else (w[0] if w else ""))
            return items
        kk=min(k,99)
        randidx=[rng.choice(N,min(k,8),replace=False) for _ in range(N)]  # cap bag size at 8
        coh_p,coh_c=p1_and_distinct(build(en_w,"coherent","en"),build(xx_w,"coherent",tgt))
        shf_p,shf_c=p1_and_distinct(build(en_w,"shuffled","en"),build(xx_w,"shuffled",tgt))
        if k>=99:
            rb_p,rb_c=(float("nan"),float("nan"))
        else:
            rb_p,rb_c=p1_and_distinct(build(en_w,"randombag","en"),build(xx_w,"randombag",tgt))
        klab=k if k<99 else "full"
        print(f"  {str(klab):>4} {coh_p:>7.3f}(c{coh_c:.2f}) {shf_p:>7.3f}(c{shf_c:.2f}) {rb_p:>7.3f}(c{rb_c:.2f})", flush=True)
    hb(f"en-{tgt} done")
print("LENGTH_CONFOUND_CONTROL DONE", flush=True)
