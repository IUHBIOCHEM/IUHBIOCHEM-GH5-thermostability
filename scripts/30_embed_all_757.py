"""Embed toàn bộ tập 757 GH5 đã xác minh bằng ESM2-650M (mean-pool), cache theo accession.
Output: ai_training/embeddings_all/esm2_650M_all.npy  + accessions.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import numpy as np
import pandas as pd
import torch
import esm

DATA = "ai_training/gh5_all_verified.csv"
OUT = "ai_training/embeddings_all"
MAX_LEN = 1022


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA).reset_index(drop=True)
    seqs = df["Sequence"].map(clean).tolist()
    dev = device()
    print(f"Nạp ESM2-650M trên {dev}, {len(seqs)} seq...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    bc = alphabet.get_batch_converter()
    embs = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            _, _, toks = bc([("x", s[:MAX_LEN])])
            rep = model(toks.to(dev), repr_layers=[33])["representations"][33]
            embs.append(rep[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy())
            if i % 50 == 0:
                print(f"  {i}/{len(seqs)}", flush=True)
    X = np.vstack(embs).astype(np.float32)
    np.save(f"{OUT}/esm2_650M_all.npy", X)
    df[["Accession", "Organism"]].to_csv(f"{OUT}/accessions.csv", index=False)
    print(f"Lưu {X.shape} -> {OUT}/esm2_650M_all.npy")


if __name__ == "__main__":
    main()
