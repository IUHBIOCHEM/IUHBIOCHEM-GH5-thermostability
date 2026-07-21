"""
40_ablation.py — Ablation study cho pipeline dự đoán OGT.

(A) Leakage control: random 5-fold vs GroupKFold theo cụm cd-hit (chứng minh random split thổi phồng).
(B) Regressor head trên ESM2-650M: Ridge, RandomForest, SVR, kNN.
(C) Feature-set: physicochemical / composition / ESM2 / ESM2+physico.

Output: manuscript/figures/Fig13_ablation.png
        ai_training/analysis/ablation.csv
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AA = "ACDEFGHIKLMNPQRSTVWY"


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def load():
    df = pd.read_csv("ai_training/gh5_ogt_features.csv")
    g = pd.read_csv("ai_training/groups_map_ogt_cdhit70.csv")
    df["_acc"] = df["Accession"].astype(str).str.split(".").str[0]
    g["_acc"] = g["Accession"].astype(str).str.split(".").str[0]
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner").dropna(subset=["OGT"]).reset_index(drop=True)
    y = df["OGT"].to_numpy(float); groups = df["ClusterID"].to_numpy()
    Xphys = df[["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]].to_numpy(float)
    seqs = [clean(s) for s in df["Sequence"]]
    Xaac = np.vstack([[s.count(a) / len(s) for a in AA] for s in seqs])
    Xesm = np.load("ai_training/plm_bench/esm2_650M.npy")
    return y, groups, Xphys, Xaac, Xesm


def run(X, y, splits, head="ridge"):
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        if head == "ridge":
            sc = StandardScaler().fit(X[tr]); m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr]); oof[te] = m.predict(sc.transform(X[te]))
        elif head == "rf":
            m = RandomForestRegressor(500, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        elif head == "svr":
            sc = StandardScaler().fit(X[tr]); m = SVR(C=10, gamma="scale").fit(sc.transform(X[tr]), y[tr]); oof[te] = m.predict(sc.transform(X[te]))
        elif head == "knn":
            sc = StandardScaler().fit(X[tr]); m = KNeighborsRegressor(n_neighbors=10).fit(sc.transform(X[tr]), y[tr]); oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), spearmanr(y, oof).correlation


def main():
    os.makedirs("ai_training/analysis", exist_ok=True)
    y, groups, Xphys, Xaac, Xesm = load()
    Xcomb = np.hstack([Xesm, StandardScaler().fit_transform(Xphys)])
    group_splits = list(GroupKFold(5).split(Xesm, y, groups))
    random_splits = list(KFold(5, shuffle=True, random_state=42).split(Xesm))

    rows = []
    # (A) leakage: random vs grouped (ESM2)
    r2g, rg, _ = run(Xesm, y, group_splits, "ridge")
    r2r, rr, _ = run(Xesm, y, random_splits, "ridge")
    rows.append({"ablation": "CV scheme", "setting": "GroupKFold (leakage-controlled)", "R2": r2g, "RMSE": rg})
    rows.append({"ablation": "CV scheme", "setting": "Random KFold (leaky)", "R2": r2r, "RMSE": rr})
    print(f"(A) Leakage: GroupKFold R2={r2g:.3f} | Random KFold R2={r2r:.3f} (inflation +{r2r-r2g:.3f})")

    # (B) regressor head on ESM2 (grouped)
    for head, nm in [("ridge", "Ridge"), ("rf", "RandomForest"), ("svr", "SVR"), ("knn", "kNN")]:
        r2, rm, _ = run(Xesm, y, group_splits, head)
        rows.append({"ablation": "Regressor head (ESM2)", "setting": nm, "R2": r2, "RMSE": rm})
        print(f"(B) head {nm}: R2={r2:.3f}")

    # (C) feature set (grouped, ridge)
    for X, nm in [(Xphys, "Physicochemical(5)"), (Xaac, "AA composition(20)"), (Xesm, "ESM2-650M"), (Xcomb, "ESM2 + physico")]:
        r2, rm, _ = run(X, y, group_splits, "ridge")
        rows.append({"ablation": "Feature set", "setting": nm, "R2": r2, "RMSE": rm})
        print(f"(C) features {nm}: R2={r2:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv("ai_training/analysis/ablation.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, (abl, title) in enumerate([("CV scheme", "(a) Leakage control"),
                                      ("Regressor head (ESM2)", "(b) Regressor head"),
                                      ("Feature set", "(c) Representation")]):
        d = res[res["ablation"] == abl]
        colors = ["#c0392b" if ("Group" in s or s == "Ridge" or s == "ESM2-650M") else "#888" for s in d["setting"]]
        ax[k].barh(d["setting"], d["R2"], color=colors)
        for i, v in enumerate(d["R2"]):
            ax[k].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        ax[k].set_xlim(0, 1); ax[k].set_xlabel("R²"); ax[k].set_title(title)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig13_ablation.png", dpi=200)
    print("→ Fig13_ablation.png")


if __name__ == "__main__":
    main()
