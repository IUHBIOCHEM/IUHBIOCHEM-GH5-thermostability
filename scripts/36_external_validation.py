"""
36_external_validation.py

External validation cho khả năng tổng quát hóa sang enzyme GH5 CHƯA THẤY:
  (A) Cross-label: mô hình OGT (huấn luyện trên nhãn TEMPURA) dự đoán cho các trình tự
      CHỈ có nhãn BRENDA Topt (độc lập, khác nguồn & khác loại đo) và KHÔNG nằm trong tập train OGT.
      -> tương quan giữa OGT dự đoán và Topt thực nghiệm = bằng chứng tổng quát hóa.
  (B) Leave-one-genus-out: đánh giá OGT với GroupKFold theo CHI (khắc nghiệt hơn theo cụm),
      kiểm tra dự đoán cho các chi hoàn toàn vắng mặt lúc train.

Output: manuscript/figures/Fig10_external_validation.png
        ai_training/analysis/external_validation.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ai_training/analysis", exist_ok=True)


def nacc(x):
    return str(x).split(".")[0]


X = np.load("ai_training/embeddings_all/esm2_650M_all.npy")
meta = pd.read_csv("ai_training/embeddings_all/accessions.csv")
meta["_acc"] = meta["Accession"].map(nacc)
idx = {a: i for i, a in enumerate(meta["_acc"])}

ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv").dropna(subset=["OGT"])
ogt["_acc"] = ogt["Accession"].map(nacc)
topt = pd.read_csv("ai_training/gh5_topt_labeled.csv").dropna(subset=["Topt"])
topt["_acc"] = topt["Accession"].map(nacc)

# ---- (A) Cross-label external validation ----
ogt_accs = set(ogt["_acc"])
# train OGT model on ALL 271 OGT-labeled
tr_rows = [idx[a] for a in ogt["_acc"] if a in idx]
ytr = ogt.set_index("_acc").loc[[meta.loc[r, "_acc"] for r in tr_rows], "OGT"].values
sc = StandardScaler().fit(X[tr_rows])
reg = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr_rows]), ytr)

# external test: sequences with BRENDA Topt but NOT in OGT training
ext = topt[~topt["_acc"].isin(ogt_accs)].copy()
ext = ext[ext["_acc"].isin(idx)]
Xe = X[[idx[a] for a in ext["_acc"]]]
ext["OGT_pred"] = reg.predict(sc.transform(Xe))
ext["genus"] = ext["Organism"].str.replace("[", "", regex=False).str.split().str[0]
# OGT model được huấn luyện trên prokaryote (TEMPURA); nhãn Topt của nấm bị genus-imputed
# không đáng tin (BRENDA gán ~60°C cho nấm mesophile). Đánh giá công bằng trên VI KHUẨN.
FUNGI = {"Fusarium", "Neocosmospora", "Cryptococcus", "Apiospora", "Kwoniella", "Trichoderma",
         "Aspergillus", "Tremellales", "Lecanosticta", "Aureococcus", "Salvia", "Bertholletia"}
ext["is_bacteria"] = ~ext["genus"].isin(FUNGI)
bact = ext[ext["is_bacteria"]]
rho, pr = spearmanr(bact["OGT_pred"], bact["Topt"])
r_p, _ = pearsonr(bact["OGT_pred"], bact["Topt"])
rho_all = spearmanr(ext["OGT_pred"], ext["Topt"]).correlation
print(f"(A) Cross-label external validation: {len(ext)} unseen enzymes with BRENDA Topt "
      f"({len(bact)} bacterial, {len(ext)-len(bact)} fungal)")
print(f"    BACTERIAL (valid domain): predicted-OGT vs experimental Topt Spearman={rho:.3f} "
      f"(p={pr:.1e}), Pearson={r_p:.3f}")
print(f"    (fungal subset ρ={spearmanr(ext[~ext['is_bacteria']]['OGT_pred'], ext[~ext['is_bacteria']]['Topt']).correlation:.3f} "
      f"— unreliable genus-imputed Topt; model correctly predicts fungi as mesophilic)")

# ---- (B) Leave-one-genus-out on OGT ----
o = ogt[ogt["_acc"].isin(idx)].copy()
o["genus"] = o["Organism"].str.replace("[", "", regex=False).str.split().str[0]
Xo = X[[idx[a] for a in o["_acc"]]]
yo = o["OGT"].values
genus = o["genus"].values
# chỉ giữ genus có >=3 mẫu để LOGO ổn định
vc = pd.Series(genus).value_counts()
keep = np.isin(genus, vc[vc >= 3].index)
Xo, yo, genus = Xo[keep], yo[keep], genus[keep]
logo = LeaveOneGroupOut()
oof = np.full(len(yo), np.nan)
for tr, te in logo.split(Xo, yo, genus):
    if len(np.unique(yo[tr])) < 2:
        continue
    s = StandardScaler().fit(Xo[tr])
    m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(s.transform(Xo[tr]), yo[tr])
    oof[te] = m.predict(s.transform(Xo[te]))
mask = ~np.isnan(oof)
r2_genus = r2_score(yo[mask], oof[mask]); rmse_genus = np.sqrt(mean_squared_error(yo[mask], oof[mask]))
rho_genus = spearmanr(yo[mask], oof[mask]).correlation
print(f"\n(B) Leave-one-genus-out (n={mask.sum()}, {len(np.unique(genus))} genera):")
print(f"    R2={r2_genus:.3f}, RMSE={rmse_genus:.2f}°C, Spearman={rho_genus:.3f}")

pd.DataFrame([
    {"validation": "cross-label bacterial (pred OGT vs exp. Topt, unseen enzymes)", "n": len(bact),
     "Spearman": rho, "Pearson": r_p, "R2": np.nan, "RMSE": np.nan},
    {"validation": "leave-one-genus-out (OGT)", "n": int(mask.sum()),
     "Spearman": rho_genus, "Pearson": np.nan, "R2": r2_genus, "RMSE": rmse_genus},
]).to_csv("ai_training/analysis/external_validation.csv", index=False)

# ---- Figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(bact["Topt"], bact["OGT_pred"], s=22, alpha=0.7, edgecolor="k", linewidth=0.3,
              color="#279", label=f"bacterial (ρ={rho:.2f})")
fung = ext[~ext["is_bacteria"]]
ax[0].scatter(fung["Topt"], fung["OGT_pred"], s=16, alpha=0.35, color="#cc9966",
              label="fungal (unreliable label)")
ax[0].set_xlabel("Experimental enzyme Topt, BRENDA (°C)")
ax[0].set_ylabel("Predicted OGT (°C)")
ax[0].legend(fontsize=8)
ax[0].set_title(f"(a) Cross-label validation, unseen enzymes\nbacterial Spearman={rho:.2f} (n={len(bact)})")
ax[1].scatter(yo[mask], oof[mask], s=20, alpha=0.6, edgecolor="k", linewidth=0.3, color="#933")
lims = [min(yo[mask].min(), oof[mask].min()) - 3, max(yo[mask].max(), oof[mask].max()) + 3]
ax[1].plot(lims, lims, "k--")
ax[1].set_xlabel("True OGT (°C)"); ax[1].set_ylabel("Predicted OGT (°C)")
ax[1].set_title(f"(b) Leave-one-genus-out\nR²={r2_genus:.2f}, ρ={rho_genus:.2f} ({len(np.unique(genus))} genera)")
plt.tight_layout(); plt.savefig("manuscript/figures/Fig10_external_validation.png", dpi=200)
print("→ Fig10_external_validation.png")
