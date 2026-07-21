"""
26_esm2_ogt_regression.py

Nâng cấp hồi quy OGT bằng embedding ESM2 (thay/kết hợp 5 đặc trưng lý hóa).
Embed 1 lần rồi cache (.npy). Đánh giá không rò rỉ bằng GroupKFold theo cụm cd-hit.

Input:
  ai_training/gh5_ogt_features.csv        (Accession, Sequence, OGT, + 5 feature)
  ai_training/groups_map_ogt_cdhit70.csv
Output:
  ai_training/ogt_esm2/  (embeddings cache, metrics, OOF, scatter, model)
"""
import os
# Tránh xung đột OpenMP giữa torch (pip) và libomp (conda) — phải đặt TRƯỚC import torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import argparse
import numpy as np
import pandas as pd
import torch
import esm
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "ai_training/gh5_ogt_features.csv"
GROUPS = "ai_training/groups_map_ogt_cdhit70.csv"
PHYS = ["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]
TARGET = "OGT"
MAX_LEN = 1022  # giới hạn ngữ cảnh ESM2

# Cấu hình model ESM2: tên -> (loader, layer đại diện, tag)
ESM_MODELS = {
    "150M": (lambda: esm.pretrained.esm2_t30_150M_UR50D(), 30, "150M"),
    "650M": (lambda: esm.pretrained.esm2_t33_650M_UR50D(), 33, "650M"),
}


def clean_seq(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def norm_acc(x):
    return str(x).split(".")[0]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_all(seqs, cache_path, model_key):
    if os.path.exists(cache_path):
        print(f"Dùng cache embedding: {cache_path}")
        return np.load(cache_path)

    loader, layer, tag = ESM_MODELS[model_key]
    device = get_device()
    print(f"Nạp ESM2-{tag} (layer {layer}) trên {device}...", flush=True)
    model, alphabet = loader()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()

    embs = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            s = s[:MAX_LEN]
            _, _, toks = bc([("x", s)])
            toks = toks.to(device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            embs.append(rep[:, 1:-1, :].mean(dim=1).squeeze(0).cpu().numpy())
            if i % 25 == 0:
                print(f"  embedded {i}/{len(seqs)}", flush=True)
    X = np.vstack(embs).astype(np.float32)
    np.save(cache_path, X)
    print(f"Lưu embeddings: {cache_path} shape={X.shape}")
    return X


def eval_cv(X, y, groups, name):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        reg = RidgeCV(alphas=np.logspace(-2, 4, 25))
        reg.fit(sc.transform(X[tr]), y[tr])
        oof[te] = reg.predict(sc.transform(X[te]))
    rmse = np.sqrt(mean_squared_error(y, oof))
    mae = mean_absolute_error(y, oof)
    r2 = r2_score(y, oof)
    rho = spearmanr(y, oof).correlation
    print(f"[{name:16}] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  Spearman={rho:.3f}")
    return oof, dict(model=name, RMSE=rmse, MAE=mae, R2=r2, Spearman=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(ESM_MODELS), default="650M")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--groups", default=GROUPS)
    ap.add_argument("--target", default=TARGET, help="Cột nhãn hồi quy (OGT hoặc Topt)")
    ap.add_argument("--out", default=None, help="Thư mục output (mặc định theo target+model)")
    args = ap.parse_args()
    tag = ESM_MODELS[args.model][2]
    target = args.target
    OUT = args.out or f"ai_training/{target.lower()}_esm2_{tag}"

    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(args.data)
    g = pd.read_csv(args.groups)
    df["_acc"] = df["Accession"].map(norm_acc)
    g["_acc"] = g["Accession"].map(norm_acc)
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df["_seq"] = df["Sequence"].map(clean_seq)

    y = df[target].to_numpy(float)
    groups = df["ClusterID"].to_numpy()
    print(f"target={target} | n={len(df)} | groups={len(np.unique(groups))} | {target} {y.min():.0f}-{y.max():.0f}")

    X_esm = embed_all(df["_seq"].tolist(), f"{OUT}/esm2_{tag}_emb.npy", args.model)
    X_phys = StandardScaler().fit_transform(df[PHYS].to_numpy(float))
    X_comb = np.hstack([X_esm, X_phys])

    print("\n=== So sánh (OOF, GroupKFold không rò rỉ) ===")
    baseline_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    print(f"[baseline (mean) ] RMSE={baseline_rmse:.2f}")
    _, m_phys = eval_cv(X_phys, y, groups, "5 lý hóa")
    oof_esm, m_esm = eval_cv(X_esm, y, groups, f"ESM2-{tag}")
    oof_comb, m_comb = eval_cv(X_comb, y, groups, f"ESM2-{tag} + lý hóa")

    metrics = pd.DataFrame([m_phys, m_esm, m_comb])
    metrics.to_csv(f"{OUT}/metrics_comparison.csv", index=False)

    # chọn mô hình tốt nhất theo RMSE để lưu OOF + scatter + model cuối
    best = min([(f"ESM2-{tag}", oof_esm, X_esm), (f"ESM2-{tag} + lý hóa", oof_comb, X_comb)],
               key=lambda t: np.sqrt(mean_squared_error(y, t[1])))
    best_name, best_oof, best_X = best
    r2 = r2_score(y, best_oof); rmse = np.sqrt(mean_squared_error(y, best_oof))

    out = df[["Accession", "Organism", target]].copy()
    out["OGT_pred"] = best_oof
    out.to_csv(f"{OUT}/oof_predictions.csv", index=False)

    plt.figure(figsize=(5, 5))
    plt.scatter(y, best_oof, s=18, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [y.min() - 3, y.max() + 3]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("Measured OGT (°C)"); plt.ylabel("Predicted OGT (°C)")
    plt.title(f"GH5 OGT — {best_name}\nR²={r2:.2f}, RMSE={rmse:.1f}°C, n={len(y)}")
    plt.tight_layout(); plt.savefig(f"{OUT}/scatter_true_vs_pred.png", dpi=150)

    sc = StandardScaler().fit(best_X)
    final = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(best_X), y)
    joblib.dump({"scaler": sc, "model": final, "features": best_name}, f"{OUT}/ogt_esm2_model.pkl")

    print(f"\nMô hình tốt nhất: {best_name}")
    print(f"→ Kết quả tại {OUT}/")


if __name__ == "__main__":
    main()
