"""Per-language single-token hubness for the 15 XNLI languages.

For the correlational analysis: does a language's single-token hubness predict its XNLI transfer gap?
Single-token hubness is measured on FLORES (first token of each sentence, mBERT masked-mean layer 8,
raw cross-lingual cosine nearest-neighbor, en -> lang): in-degree skewness, fraction never retrieved,
max in-degree, and single-token bitext P@1 (an inverse-hubness proxy).

Run: python experiments/xnli_hubness.py
Env: N (default 1012)
Output: results/xnli_hubness.json
"""
from __future__ import annotations
import os, json, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew

ROOT = "data/flores200/flores200_dataset/devtest"
CODE = {"ar": "arb_Arab", "bg": "bul_Cyrl", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn",
        "es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "ru": "rus_Cyrl", "sw": "swh_Latn",
        "th": "tha_Thai", "tr": "tur_Latn", "ur": "urd_Arab", "vi": "vie_Latn", "zh": "zho_Hans"}
LANGS = [l for l in CODE if l != "en"]
N = int(os.environ.get("N", "1012"))
L = 8
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(dev).eval()
hb(f"mbert on {dev}")

def first_words(lang):
    with open(f"{ROOT}/{CODE[lang]}.devtest") as f:
        sents = [s.strip() for s in f][:N]
    return [(s.split()[0] if s.split() else s) for s in sents]

@torch.no_grad()
def embed(words):
    out = []
    for i in range(0, len(words), 64):
        enc = tok(words[i:i+64], padding=True, truncation=True, max_length=8, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[L]
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out, 0)

en = embed(first_words("en"))
en = en - en.mean(0)
hb("english single tokens embedded")
res = {}
for lang in LANGS:
    xx = embed(first_words(lang))
    xx = xx - xx.mean(0)
    S = cosine_similarity(en, xx)
    pred = S.argmax(1)
    p1 = float((pred == np.arange(len(en))).mean())
    N1 = np.bincount(pred, minlength=len(xx))
    res[lang] = {"p1_k1": round(p1, 4), "skew_k1": round(float(skew(N1)), 3),
                 "never_k1": round(float((N1 == 0).mean()), 4), "maxhub_k1": int(N1.max())}
    hb(f"en-{lang}: P@1 {p1:.3f}  skew {res[lang]['skew_k1']:.2f}  never {res[lang]['never_k1']*100:.0f}%  max {res[lang]['maxhub_k1']}")

os.makedirs("results", exist_ok=True)
with open("results/xnli_hubness.json", "w") as f:
    json.dump(res, f, indent=2)
print("XNLI_HUBNESS DONE", flush=True)
