"""Bitext retrieval accuracy as a function of input length.

Measures P@1 against input length on FLORES, using a coherent prefix of the parallel
sentence, masked-mean pooling at layer L, and raw cosine retrieval. Source is English;
targets span typologically diverse languages. English prefix embeddings are cached per
length and reused across pairs. Run once per model.

Env: MODEL=mbert|xlmr, L=8, N=1012, TARGETS=comma-separated language codes, SEG=aware|whitespace.
The test split is fixed and deterministic. A run with a restricted TARGETS writes a tagged filename
so it cannot overwrite the eleven-language sweep that Figure 2 is drawn from.
HUBSTATS=1 additionally records the in-degree distribution behind each point on the
curve: gallery size, peak in-degree, fraction never retrieved, skewness and the hub
string itself. This is where the paper's single-token hubness figures come from
(mBERT en--es at k=1: peak in-degree 134, 82% never retrieved, hub "El"; at full
sentence length: 2 and 0.4%).
Out: results/length_{MODEL}.json (restricted TARGETS -> results/length_{MODEL}_{targets}.json)

Run: python experiments/length_curve.py
     MODEL=mbert TARGETS=es HUBSTATS=1 python experiments/length_curve.py
"""
from __future__ import annotations
import os, json, time
import numpy as np, torch
from scipy.stats import skew
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
ALL_TARGETS = "es,fr,de,it,bg,ru,fi,tr,ko,zh,ar"
TARGETS = os.environ.get("TARGETS", ALL_TARGETS).split(",")
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

# Whitespace is not a word boundary in every script. FLORES Chinese is unsegmented,
# so s.split() returns the whole sentence as one "word" for ~69% of lines and the
# target side is never actually truncated. Segment per language instead; SEG=whitespace
# restores the old behaviour for comparison.
SEG = os.environ.get("SEG", "aware")
NOSPACE = {"zh"}          # scripts written without inter-word spaces

def segment(lang, sent):
    if SEG == "whitespace" or lang not in NOSPACE:
        return sent.split()
    import jieba
    return [w for w in jieba.cut(sent) if w.strip()]

def prefix(words, k, lang="en"):
    join = "" if (SEG != "whitespace" and lang in NOSPACE) else " "
    return [join.join(w[:k]) if w[:k] else (w[0] if w else "") for w in words]

HUBSTATS = os.environ.get("HUBSTATS", "") not in ("", "0")

def p1(X, Y, labels=None):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)
    # Hold out a fixed test split; the first 300 indices act as the gallery seen during tuning.
    test = np.random.RandomState(0).permutation(n)[300:]
    pred = cosine_similarity(X[test], Y[test]).argmax(1)
    acc = float((pred == np.arange(len(test))).mean())
    if not HUBSTATS:
        return acc, None
    # Queries and gallery are the same held-out set, so in-degree is over len(test) items.
    n1 = np.bincount(pred, minlength=len(test))
    top = int(n1.argmax())
    return acc, {"gallery": int(len(test)), "peak_indegree": int(n1.max()),
                 "never_retrieved": round(float((n1 == 0).mean()), 4),
                 "skew": round(float(skew(n1)), 3),
                 "hub": labels[test[top]] if labels else None}

en_words = [segment("en", s) for s in load_sents("en")]
en_emb = {k: embed(prefix(en_words, k, "en")) for k in KS}
hb("cached English prefix embeddings")

curves, hubness = {}, {}
for tgt in TARGETS:
    if tgt not in CODE:
        hb(f"skip {tgt} (no FLORES code)")
        continue
    xx_words = [segment(tgt, s) for s in load_sents(tgt)]
    row, stats = [], {}
    for k in KS:
        texts = prefix(xx_words, k, tgt)
        acc, st = p1(en_emb[k], embed(texts), texts)
        row.append(round(acc, 4))
        if st:
            stats[str(k)] = st
            hb(f"en-{tgt} k={k}: P@1={acc:.4f} peak={st['peak_indegree']} "
               f"never={st['never_retrieved']:.3f} hub={st['hub']!r}")
    curves[f"en-{tgt}"] = dict(zip([str(k) for k in KS], row))
    if stats:
        hubness[f"en-{tgt}"] = stats
    hb(f"en-{tgt}: {row}")

out = {"model": MODEL, "layer": L, "n": N, "ks": KS, "segmentation": SEG, "curves": curves}
if hubness:
    out["hubness"] = hubness
os.makedirs("results", exist_ok=True)
suffix = "" if SEG == "aware" else f"_{SEG}"
# A restricted run holds fewer languages than the full sweep, so it must not overwrite the
# eleven-language file the paper's Figure 2 is drawn from. Tag the filename instead.
if set(TARGETS) != set(ALL_TARGETS.split(",")):
    suffix += "_" + "-".join(TARGETS)
with open(f"results/length_{MODEL}{suffix}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"LENGTH_{MODEL.upper()} DONE", flush=True)
