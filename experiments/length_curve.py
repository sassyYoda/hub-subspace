"""Bitext retrieval accuracy as a function of input length.

Measures P@1 against input length on FLORES, using a coherent prefix of the parallel
sentence, masked-mean pooling at layer L, and raw cosine retrieval. Source is English;
targets span typologically diverse languages. English prefix embeddings are cached per
length and reused across pairs. Run once per model.

Env: MODEL=mbert|xlmr, L=8, N=1012. The test split is fixed and deterministic.
Out: results/length_{MODEL}.json

Run: python experiments/length_curve.py
"""
from __future__ import annotations
import os, json, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = os.environ.get("MODEL", "mbert")
HFID = {"mbert": "bert-base-multilingual-cased", "xlmr": "xlm-roberta-base"}[MODEL]
L = int(os.environ.get("L", "8"))
N = int(os.environ.get("N", "1012"))
ROOT = "data/flores200/flores200_dataset/devtest"
CODE = {"en": "eng_Latn", "es": "spa_Latn", "zh": "zho_Hans", "de": "deu_Latn", "fr": "fra_Latn",
        "it": "ita_Latn", "bg": "bul_Cyrl", "fi": "fin_Latn", "ko": "kor_Hang", "tr": "tur_Latn",
        "ru": "rus_Cyrl", "ar": "arb_Arab"}
TARGETS = os.environ.get("TARGETS", "es,fr,de,it,bg,ru,fi,tr,ko,zh,ar").split(",")
KS = [1, 2, 3, 5, 10, 99]
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(HFID)
model = AutoModel.from_pretrained(HFID).to(dev).eval()
hb(f"{MODEL} on {dev}, layer {L}")

def load_sents(lang, n=N):
    with open(f"{ROOT}/{CODE[lang]}.devtest") as f:
        return [l.strip() for l in f][:n]

@torch.no_grad()
def embed(sents):
    out = []
    for i in range(0, len(sents), 32):
        enc = tok(sents[i:i+32], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[L]
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out, 0)

def prefix(words, k):
    return [" ".join(w[:k]) if w[:k] else (w[0] if w else "") for w in words]

def p1(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)
    # Hold out a fixed test split; the first 300 indices act as the gallery seen during tuning.
    test = np.random.RandomState(0).permutation(n)[300:]
    pred = cosine_similarity(X[test], Y[test]).argmax(1)
    return float((pred == np.arange(len(test))).mean())

en_words = [s.split() for s in load_sents("en")]
en_emb = {k: embed(prefix(en_words, k)) for k in KS}
hb("cached English prefix embeddings")

curves = {}
for tgt in TARGETS:
    if tgt not in CODE:
        hb(f"skip {tgt} (no FLORES code)")
        continue
    xx_words = [s.split() for s in load_sents(tgt)]
    row = []
    for k in KS:
        row.append(round(p1(en_emb[k], embed(prefix(xx_words, k))), 4))
    curves[f"en-{tgt}"] = dict(zip([str(k) for k in KS], row))
    hb(f"en-{tgt}: {row}")

out = {"model": MODEL, "layer": L, "n": N, "ks": KS, "curves": curves}
os.makedirs("results", exist_ok=True)
with open(f"results/length_{MODEL}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"LENGTH_{MODEL.upper()} DONE", flush=True)
