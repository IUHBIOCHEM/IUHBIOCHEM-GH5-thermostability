"""
33_umap_ranking_error.py

(1) UMAP của embedding ESM2-650M (tô màu theo nhiệt độ dự đoán / nhãn / taxonomy)
(2) Dự đoán OGT + xác suất bền nhiệt cho toàn bộ 757 -> bảng xếp hạng ứng viên
(3) Error analysis cho mô hình hồi quy OGT (residual theo nhiều chiều + calibration)

Output: manuscript/figures/Fig6_UMAP.png
        manuscript/figures/Fig7_error_analysis.png
        ai_training/analysis/top_candidates.csv
        ai_training/analysis/all757_predictions.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
import umap

os.makedirs("ai_training/analysis", exist_ok=True)


def nacc(x):
    return str(x).split(".")[0]


# ---- load embeddings + metadata ----
X = np.load("ai_training/embeddings_all/esm2_650M_all.npy")
meta = pd.read_csv("ai_training/embeddings_all/accessions.csv")
meta["_acc"] = meta["Accession"].map(nacc)
idx = {a: i for i, a in enumerate(meta["_acc"])}

ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]].dropna()
ogt["_acc"] = ogt["Accession"].map(nacc)
topt = pd.read_csv("ai_training/gh5_topt_labeled.csv")[["Accession", "Topt"]].dropna()
topt["_acc"] = topt["Accession"].map(nacc)

ogt_map = dict(zip(ogt["_acc"], ogt["OGT"]))
topt_map = dict(zip(topt["_acc"], topt["Topt"]))
meta["OGT"] = meta["_acc"].map(ogt_map)
meta["Topt"] = meta["_acc"].map(topt_map)

# ---- train final models on labeled subsets, predict all 757 ----
def fit_predict_reg():
    m = ogt.dropna(subset=["OGT"])
    rows = [idx[a] for a in m["_acc"] if a in idx]
    Xtr = X[rows]; ytr = m.set_index("_acc").loc[[meta.loc[r, "_acc"] for r in rows], "OGT"].values
    sc = StandardScaler().fit(Xtr)
    reg = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xtr), ytr)
    return sc, reg

def fit_predict_clf():
    m = topt.dropna(subset=["Topt"])
    rows = [idx[a] for a in m["_acc"] if a in idx]
    Xtr = X[rows]; ytr = (m.set_index("_acc").loc[[meta.loc[r, "_acc"] for r in rows], "Topt"].values >= 60).astype(int)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return sc, clf

scr, reg = fit_predict_reg()
scc, clf = fit_predict_clf()
meta["OGT_pred"] = reg.predict(scr.transform(X))
meta["P_thermostable"] = clf.predict_proba(scc.transform(X))[:, 1]
meta.drop(columns=["_acc"]).to_csv("ai_training/analysis/all757_predictions.csv", index=False)

# ---- ranked candidate table: sequences WITHOUT experimental thermostable label ----
novel = meta[meta["Topt"].isna() & meta["OGT"].isna()].copy()
novel = novel.sort_values("P_thermostable", ascending=False)
top = novel.head(20)[["Accession", "Organism", "OGT_pred", "P_thermostable"]]
top.columns = ["Accession", "Organism", "Predicted_OGT_C", "P_thermostable"]
top.round({"Predicted_OGT_C": 1, "P_thermostable": 3}).to_csv("ai_training/analysis/top_candidates.csv", index=False)
print("Top candidates (novel, no experimental label):")
print(top.round(2).head(12).to_string(index=False))

# ---- UMAP ----
Xs = StandardScaler().fit_transform(X)
emb = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=42).fit_transform(Xs)
meta["u1"], meta["u2"] = emb[:, 0], emb[:, 1]

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
# (a) predicted OGT
sc0 = ax[0].scatter(meta["u1"], meta["u2"], c=meta["OGT_pred"], cmap="coolwarm", s=14, alpha=0.85)
plt.colorbar(sc0, ax=ax[0], label="Predicted OGT (°C)"); ax[0].set_title("(a) ESM2 embedding — predicted OGT")
# (b) label class
lab = np.where(meta["Topt"] >= 60, "thermo (exp.)",
      np.where(meta["Topt"] < 60, "meso (exp.)",
      np.where(meta["OGT"] >= 55, "thermo (OGT)",
      np.where(meta["OGT"] < 55, "meso (OGT)", "unlabeled"))))
for name, col in [("unlabeled", "#cccccc"), ("meso (OGT)", "#6699cc"), ("thermo (OGT)", "#cc5544"),
                  ("meso (exp.)", "#2b6cb0"), ("thermo (exp.)", "#b03030")]:
    mk = lab == name
    ax[1].scatter(meta["u1"][mk], meta["u2"][mk], s=14, alpha=0.8, label=f"{name} ({mk.sum()})", color=col)
ax[1].legend(fontsize=8, markerscale=1.5); ax[1].set_title("(b) Thermostability label")
# (c) genus
meta["genus"] = meta["Organism"].str.replace("[", "", regex=False).str.split().str[0]
topg = meta["genus"].value_counts().head(8).index
import matplotlib.cm as cm
colors = cm.tab10(np.linspace(0, 1, len(topg)))
ax[2].scatter(meta["u1"], meta["u2"], s=10, color="#dddddd")
for gi, gname in enumerate(topg):
    mk = meta["genus"] == gname
    ax[2].scatter(meta["u1"][mk], meta["u2"][mk], s=14, alpha=0.85, label=f"{gname} ({mk.sum()})", color=colors[gi])
ax[2].legend(fontsize=7, markerscale=1.5); ax[2].set_title("(c) Source genus (top 8)")
for a in ax: a.set_xlabel("UMAP-1"); a.set_ylabel("UMAP-2"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig("manuscript/figures/Fig6_UMAP.png", dpi=200); plt.close()
print("→ Fig6_UMAP.png")

# ---- Error analysis (OOF residuals of OGT ESM2-650M model) ----
oof = pd.read_csv("ai_training/ogt_esm2_650M/oof_predictions.csv")
feat = pd.read_csv("ai_training/gh5_ogt_features.csv")[["Accession", "Length"]]
lblsrc = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT_source"]]
oof = oof.merge(feat, on="Accession", how="left").merge(lblsrc, on="Accession", how="left")
oof["resid"] = oof["OGT_pred"] - oof["OGT"]

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
# residual vs predicted
ax[0, 0].scatter(oof["OGT_pred"], oof["resid"], s=16, alpha=0.6, edgecolor="k", linewidth=0.3)
ax[0, 0].axhline(0, color="r", ls="--"); ax[0, 0].set_xlabel("Predicted OGT (°C)"); ax[0, 0].set_ylabel("Residual (pred − true)")
ax[0, 0].set_title(f"(a) Residuals vs prediction (RMSE={np.sqrt((oof['resid']**2).mean()):.1f}°C)")
# residual vs length
ax[0, 1].scatter(oof["Length"], oof["resid"], s=16, alpha=0.6, edgecolor="k", linewidth=0.3, color="#279")
ax[0, 1].axhline(0, color="r", ls="--"); ax[0, 1].set_xlabel("Sequence length (aa)"); ax[0, 1].set_ylabel("Residual")
ax[0, 1].set_title("(b) Residuals vs sequence length")
# |residual| by label source
for i, (src, c) in enumerate([("species", "#cc5544"), ("genus", "#6699cc")]):
    d = oof[oof["OGT_source"] == src]["resid"].abs().dropna()
    ax[1, 0].boxplot(d, positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=c, alpha=0.7), showfliers=False)
ax[1, 0].set_xticks([0, 1]); ax[1, 0].set_xticklabels(["species-level", "genus-level"])
ax[1, 0].set_ylabel("|Residual| (°C)"); ax[1, 0].set_title("(c) Absolute error by label source")
# calibration: true vs predicted with y=x
ax[1, 1].scatter(oof["OGT"], oof["OGT_pred"], s=16, alpha=0.6, edgecolor="k", linewidth=0.3, color="#933")
lims = [oof["OGT"].min() - 3, oof["OGT"].max() + 3]
ax[1, 1].plot(lims, lims, "k--"); ax[1, 1].set_xlabel("True OGT (°C)"); ax[1, 1].set_ylabel("Predicted OGT (°C)")
ax[1, 1].set_title("(d) Calibration")
plt.tight_layout(); plt.savefig("manuscript/figures/Fig7_error_analysis.png", dpi=200); plt.close()

# error summary
print("\nError by label source (|residual| °C):")
print(oof.groupby("OGT_source")["resid"].apply(lambda s: f"MAE={s.abs().mean():.2f}, RMSE={np.sqrt((s**2).mean()):.2f}").to_string())
print("→ Fig7_error_analysis.png")
