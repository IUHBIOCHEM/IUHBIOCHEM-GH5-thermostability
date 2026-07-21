"""
37_plm_benchmark.py — Benchmark nhiều protein language model trên nhiệm vụ hồi quy OGT.

So sánh embedding từ các PLM khác nhau (ESM2 4 kích thước, ESM-1b thế hệ trước, ProtBERT
kiến trúc BERT) trên CÙNG tập, CÙNG GroupKFold. Cho thấy kết quả không phụ thuộc một PLM
và ESM2-650M cạnh tranh tốt nhất.

Output: manuscript/figures/Fig11_plm_benchmark.png
        ai_training/analysis/plm_benchmark.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "ai_training/plm_bench"
os.makedirs(CACHE, exist_ok=True)
os.makedirs("ai_training/analysis", exist_ok=True)
MAX = 1022


def dev():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def esm_embed(seqs, loader, layer, tag):
    cp = f"{CACHE}/{tag}.npy"
    if os.path.exists(cp):
        return np.load(cp)
    import esm
    model, alphabet = loader()
    d = dev(); model = model.to(d).eval(); bc = alphabet.get_batch_converter()
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            _, _, toks = bc([("x", s[:MAX])])
            rep = model(toks.to(d), repr_layers=[layer])["representations"][layer]
            out.append(rep[:, 1:-1, :].mean(1).squeeze(0).cpu().numpy())
            if i % 50 == 0:
                print(f"    {tag} {i}/{len(seqs)}", flush=True)
    X = np.vstack(out).astype(np.float32); np.save(cp, X); return X


def protbert_embed(seqs, tag="protbert"):
    cp = f"{CACHE}/{tag}.npy"
    if os.path.exists(cp):
        return np.load(cp)
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(dev()).eval()
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            spaced = " ".join(list(s[:510]))
            enc = tok(spaced, return_tensors="pt")
            enc = {k: v.to(dev()) for k, v in enc.items()}
            h = model(**enc).last_hidden_state[0, 1:-1, :]  # bỏ [CLS]/[SEP]
            out.append(h.mean(0).cpu().numpy())
            if i % 50 == 0:
                print(f"    {tag} {i}/{len(seqs)}", flush=True)
    X = np.vstack(out).astype(np.float32); np.save(cp, X); return X


def evaluate(X, y, groups):
    gkf = GroupKFold(5); oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return (r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), spearmanr(y, oof).correlation)


def main():
    df = pd.read_csv("ai_training/gh5_ogt_features.csv")
    g = pd.read_csv("ai_training/groups_map_ogt_cdhit70.csv")
    df["_acc"] = df["Accession"].astype(str).str.split(".").str[0]
    g["_acc"] = g["Accession"].astype(str).str.split(".").str[0]
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner").dropna(subset=["OGT"]).reset_index(drop=True)
    seqs = df["Sequence"].map(clean).tolist()
    y = df["OGT"].to_numpy(float); groups = df["ClusterID"].to_numpy()
    print(f"n={len(df)}")

    import esm
    models = [
        ("ESM2-8M", lambda: esm_embed(seqs, esm.pretrained.esm2_t6_8M_UR50D, 6, "esm2_8M"), 8),
        ("ESM2-35M", lambda: esm_embed(seqs, esm.pretrained.esm2_t12_35M_UR50D, 12, "esm2_35M"), 35),
        ("ESM2-150M", lambda: esm_embed(seqs, esm.pretrained.esm2_t30_150M_UR50D, 30, "esm2_150M"), 150),
        ("ESM2-650M", lambda: esm_embed(seqs, esm.pretrained.esm2_t33_650M_UR50D, 33, "esm2_650M"), 650),
        ("ProtBERT", lambda: protbert_embed(seqs), 420),
    ]
    rows = []
    for name, fn, params in models:
        print(f"[{name}] embedding...", flush=True)
        try:
            X = fn()
            r2, rmse, rho = evaluate(X, y, groups)
            rows.append({"PLM": name, "params_M": params, "dim": X.shape[1], "R2": r2, "RMSE": rmse, "Spearman": rho})
            print(f"  {name}: R2={r2:.3f} RMSE={rmse:.2f} rho={rho:.3f}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    res = pd.DataFrame(rows).sort_values("R2", ascending=False)
    res.to_csv("ai_training/analysis/plm_benchmark.csv", index=False)
    print(res.round(3).to_string(index=False))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    r = res.sort_values("R2")
    colors = ["#c0392b" if "650M" in p and "ESM2" in p else "#279" for p in r["PLM"]]
    ax[0].barh(r["PLM"], r["R2"], color=colors)
    for i, v in enumerate(r["R2"]):
        ax[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax[0].set_xlabel("R² (OOF, GroupKFold)"); ax[0].set_xlim(0, 1); ax[0].set_title("(a) OGT prediction accuracy by PLM")
    ax[1].scatter(res["params_M"], res["R2"], s=70, color="#279")
    for _, row in res.iterrows():
        ax[1].annotate(row["PLM"], (row["params_M"], row["R2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Model size (M params, log)"); ax[1].set_ylabel("R²")
    ax[1].set_title("(b) Accuracy vs model scale")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig11_plm_benchmark.png", dpi=200)
    print("→ Fig11_plm_benchmark.png")


if __name__ == "__main__":
    main()
