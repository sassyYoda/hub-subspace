"""Cross-encoder frozen-feature extraction for MNLI/XNLI.

Cross-encodes `premise </s> hypothesis` through a frozen encoder, masked-mean pools a few candidate
layers (so the layer can be model-selected on MNLI-val later), and caches per-target .npz files.
Any target already cached is skipped, so the job is safe to re-run or resume after an interrupt.

Targets: train (MNLI), dev (MNLI validation_matched), test_<lang> for the 15 XNLI languages.

Run: python experiments/xnli_extract.py
Env: MODEL=mbert|xlmr, TRAIN_N=30000, DEV_N=2000, LAYERS=6,9,12, MAXLEN=128, BATCH=64
Output: results/xnli_feats/{MODEL}/{target}.npz  (keys: L6, L9, L12, y)
"""
from __future__ import annotations
import os, time
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "mbert")
HFID = {"mbert": "bert-base-multilingual-cased", "xlmr": "xlm-roberta-base"}[MODEL]
TRAIN_N = int(os.environ.get("TRAIN_N", "30000"))
DEV_N = int(os.environ.get("DEV_N", "2000"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "6,9,12").split(",")]
MAXLEN = int(os.environ.get("MAXLEN", "128"))
BATCH = int(os.environ.get("BATCH", "64"))
XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
OUTDIR = f"results/xnli_feats/{MODEL}"
os.makedirs(OUTDIR, exist_ok=True)
_T0 = time.time()
def hb(m): print(f"  [hb {time.time()-_T0:.1f}s] {m}", flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(HFID)
model = AutoModel.from_pretrained(HFID).to(dev).eval()
hb(f"{MODEL} on {dev}; layers {LAYERS}")

@torch.no_grad()
def encode(prem, hyp):
    # returns {layer: [N, d]} masked-mean pooled cross-encoded features
    out = {L: [] for L in LAYERS}
    for i in range(0, len(prem), BATCH):
        enc = tok(prem[i:i+BATCH], hyp[i:i+BATCH], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for L in LAYERS:
            pooled = (hs[L]*m).sum(1)/m.sum(1).clamp(min=1)
            out[L].append(pooled.cpu().numpy().astype(np.float32))
        if (i//BATCH) % 20 == 0 and i > 0:
            hb(f"    {i}/{len(prem)}")
    return {L: np.concatenate(out[L], 0) for L in LAYERS}

def save_target(name, prem, hyp, labels):
    path = f"{OUTDIR}/{name}.npz"
    if os.path.exists(path):
        hb(f"skip {name} (cached)")
        return
    t = time.time()
    feats = encode(list(prem), list(hyp))
    np.savez(path, y=np.array(labels, dtype=np.int64), **{f"L{L}": feats[L] for L in LAYERS})
    hb(f"saved {name}: {len(labels)} pairs in {time.time()-t:.1f}s")

# MNLI English train + dev (labels: 0 entail, 1 neutral, 2 contra; matches XNLI)
mnli_tr = load_dataset("nyu-mll/glue", "mnli", split=f"train[:{TRAIN_N}]")
save_target("train", mnli_tr["premise"], mnli_tr["hypothesis"], mnli_tr["label"])
mnli_dev = load_dataset("nyu-mll/glue", "mnli", split=f"validation_matched[:{DEV_N}]")
save_target("dev", mnli_dev["premise"], mnli_dev["hypothesis"], mnli_dev["label"])

# XNLI test, per language
for lang in XNLI_LANGS:
    if os.path.exists(f"{OUTDIR}/test_{lang}.npz"):
        hb(f"skip test_{lang} (cached)")
        continue
    ds = load_dataset("xnli", lang, split="test")
    save_target(f"test_{lang}", ds["premise"], ds["hypothesis"], ds["label"])

print(f"XNLI_EXTRACT_{MODEL.upper()} DONE", flush=True)
