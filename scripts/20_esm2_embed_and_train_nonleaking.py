#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import esm
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent
AI = BASE / "ai_training"

DATA = AI / "gh5_training_data_with_features.csv"
GROUPS = AI / "groups_map_cdhit70.csv"
OUT = AI / "esm2_150M_nonleaking"
OUT.mkdir(parents=True, exist_ok=True)

def clean_seq(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())

print("Loading data...", flush=True)
df = pd.read_csv(DATA)
g = pd.read_csv(GROUPS)

df["_acc"] = df["Accession"].astype(str).str.split(".").str[0]
g["_acc"]  = g["Accession"].astype(str).str.split(".").str[0]

df = df.merge(g[["_acc","ClusterID"]], on="_acc", how="inner")
df["Sequence"] = df["Sequence"].map(clean_seq)

y = df["Is_Thermophilic_Guess"].astype(int).to_numpy()
groups = df["ClusterID"].to_numpy()
seqs = df["Sequence"].tolist()

print("Samples:", len(seqs), flush=True)

print("Loading ESM-2 150M...", flush=True)
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model.eval()
batch_converter = alphabet.get_batch_converter()

@torch.no_grad()
def embed(seq):
    _, _, toks = batch_converter([("x", seq)])
    rep = model(toks, repr_layers=[30])["representations"][30]
    return rep[:,1:-1,:].mean(dim=1).squeeze(0).cpu().numpy()

rows = []
cv = GroupKFold(n_splits=5)

for fold, (tr, te) in enumerate(cv.split(seqs, y, groups), 1):
    print(f"[Fold {fold}]", flush=True)

    Xtr = np.vstack([embed(seqs[i]) for i in tr])
    Xte = np.vstack([embed(seqs[i]) for i in te])
    ytr = y[tr]
    yte = y[te]

    if len(np.unique(ytr)) < 2:
        pred = np.full(len(yte), ytr[0])
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    rows.append({"fold": fold, "accuracy": acc})
    print("  ACC =", acc, flush=True)

pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
print("DONE", flush=True)
