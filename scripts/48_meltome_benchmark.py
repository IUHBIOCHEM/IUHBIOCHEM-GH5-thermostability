"""
48_meltome_benchmark.py — External experimental validation trên Meltome Atlas (FLIP mixed split).

Chứng minh khung ESM2 dự đoán được NHIỆT ĐỘ NÓNG CHẢY THỰC NGHIỆM (Tm), không chỉ OGT proxy.
Subsample (compute), embed ESM2-150M, RidgeCV train -> test (split chuẩn của FLIP), báo Spearman/RMSE/R².

Output: manuscript/figures/Fig19_meltome.png , ai_training/analysis/meltome_benchmark.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import numpy as np
import pandas as pd
import torch, esm
from Bio import SeqIO
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CACHE = "ai_training/meltome"; os.makedirs(CACHE, exist_ok=True)
N_TRAIN, N_TEST, SEED = 4000, 1500, 0


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def load():
    rows = []
    for rec in SeqIO.parse("data_meltome/mixed_split.fasta", "fasta"):
        t = re.search(r"TARGET=([0-9.]+)", rec.description); s = re.search(r"SET=(\w+)", rec.description)
        if t and s:
            rows.append((s.group(1), float(t.group(1)), clean(rec.seq)))
    d = pd.DataFrame(rows, columns=["set", "Tm", "seq"])
    d = d[d["seq"].str.len().between(40, 1022)]
    rng = np.random.default_rng(SEED)
    tr = d[d["set"] == "train"].sample(min(N_TRAIN, (d["set"] == "train").sum()), random_state=SEED)
    te = d[d["set"] == "test"].sample(min(N_TEST, (d["set"] == "test").sum()), random_state=SEED)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def embed(tag, seqs):
    cp = f"{CACHE}/{tag}.npy"
    if os.path.exists(cp):
        return np.load(cp)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, alph = esm.pretrained.esm2_t30_150M_UR50D(); model = model.to(dev).eval(); bc = alph.get_batch_converter()
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            _, _, toks = bc([("x", s[:1022])])
            rep = model(toks.to(dev), repr_layers=[30])["representations"][30]
            out.append(rep[:, 1:-1, :].mean(1).squeeze(0).cpu().numpy())
            if i % 200 == 0:
                print(f"  {tag} {i}/{len(seqs)}", flush=True)
    X = np.vstack(out).astype(np.float32); np.save(cp, X); return X


def main():
    tr, te = load()
    print(f"train={len(tr)}, test={len(te)}", flush=True)
    Xtr = embed("train", tr["seq"].tolist()); Xte = embed("test", te["seq"].tolist())
    sc = StandardScaler().fit(Xtr)
    reg = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xtr), tr["Tm"].values)
    pred = reg.predict(sc.transform(Xte))
    rho = spearmanr(te["Tm"], pred).correlation
    rmse = np.sqrt(mean_squared_error(te["Tm"], pred)); r2 = r2_score(te["Tm"], pred)
    print(f"\nMeltome test: Spearman={rho:.3f}  RMSE={rmse:.2f}C  R2={r2:.3f}")
    pd.DataFrame([{"n_train": len(tr), "n_test": len(te), "Spearman": rho, "RMSE": rmse, "R2": r2}]
                 ).to_csv("ai_training/analysis/meltome_benchmark.csv", index=False)

    plt.figure(figsize=(5.2, 5))
    plt.scatter(te["Tm"], pred, s=8, alpha=0.35, edgecolor="none", color="#2166ac")
    lims = [te["Tm"].min() - 3, te["Tm"].max() + 3]; plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Experimental Tm (°C, Meltome Atlas)"); plt.ylabel("Predicted Tm (°C)")
    plt.title(f"External experimental validation (Meltome)\nSpearman={rho:.2f}, RMSE={rmse:.1f}°C (n={len(te)} test)")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig19_meltome.png", dpi=200)
    print("→ Fig19_meltome.png")


if __name__ == "__main__":
    main()
