"""
39_sota_baselines.py

So sánh với các phương pháp nền/SOTA cho dự đoán nhiệt độ từ trình tự (OGT), cùng tập & cùng GroupKFold:
  - Zeldovich IVYWREL (2007): hồi quy tuyến tính theo tỉ lệ residue IVYWREL (phương pháp composition kinh điển)
  - Amino-acid composition (20) + Random Forest
  - Dipeptide composition (400) + Ridge  (đặc trưng cổ điển cho bền nhiệt)
  - Physicochemical (5) + Random Forest   (cách tiếp cận "cổ điển" trong bản thảo cũ)
  - ESM2-650M embedding + Ridge  (đề xuất của chúng tôi)

Output: manuscript/figures/Fig12_sota_comparison.png
        ai_training/analysis/sota_comparison.csv
"""
import os
import re
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AA = "ACDEFGHIKLMNPQRSTVWY"
IVYWREL = set("IVYWREL")
DIPEP = [a + b for a in AA for b in AA]


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def aac(s):
    n = len(s)
    return np.array([s.count(a) / n for a in AA])


def dipep(s):
    n = len(s) - 1
    c = {d: 0 for d in DIPEP}
    for i in range(n):
        c[s[i:i+2]] = c.get(s[i:i+2], 0) + 1
    return np.array([c[d] / n for d in DIPEP]) if n > 0 else np.zeros(400)


def cv_eval(X, y, groups, model="ridge"):
    gkf = GroupKFold(5); oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if model == "ridge":
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict(sc.transform(X[te]))
        else:
            m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
    return r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), spearmanr(y, oof).correlation


def main():
    os.makedirs("ai_training/analysis", exist_ok=True)
    df = pd.read_csv("ai_training/gh5_ogt_features.csv")
    g = pd.read_csv("ai_training/groups_map_ogt_cdhit70.csv")
    df["_acc"] = df["Accession"].astype(str).str.split(".").str[0]
    g["_acc"] = g["Accession"].astype(str).str.split(".").str[0]
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner").dropna(subset=["OGT"]).reset_index(drop=True)
    seqs = [clean(s) for s in df["Sequence"]]
    y = df["OGT"].to_numpy(float); groups = df["ClusterID"].to_numpy()

    X_ivy = np.array([[100 * sum(s.count(a) for a in IVYWREL) / len(s)] for s in seqs])
    X_aac = np.vstack([aac(s) for s in seqs])
    X_dip = np.vstack([dipep(s) for s in seqs])
    X_phys = df[["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]].to_numpy(float)
    X_esm = np.load("ai_training/plm_bench/esm2_650M.npy")

    rows = []
    for name, X, mdl, ref in [
        ("Zeldovich IVYWREL (2007)", X_ivy, "ridge", "composition, SOTA baseline"),
        ("AA composition + RF", X_aac, "rf", "classical"),
        ("Dipeptide comp. + Ridge", X_dip, "ridge", "classical"),
        ("Physicochemical(5) + RF", X_phys, "rf", "prior manuscript"),
        ("ESM2-650M + Ridge (ours)", X_esm, "ridge", "this work"),
    ]:
        r2, rmse, rho = cv_eval(X, y, groups, mdl)
        rows.append({"Method": name, "type": ref, "R2": r2, "RMSE": rmse, "Spearman": rho})
        print(f"  {name:32} R2={r2:.3f} RMSE={rmse:.2f} rho={rho:.3f}")
    res = pd.DataFrame(rows).sort_values("R2")
    res.to_csv("ai_training/analysis/sota_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if "ours" in m else "#888" for m in res["Method"]]
    ax.barh(res["Method"], res["R2"], color=colors)
    for i, (v, e) in enumerate(zip(res["R2"], res["RMSE"])):
        ax.text(v + 0.01, i, f"R²={v:.2f}, RMSE={e:.1f}°C", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("R² (OOF, GroupKFold)")
    ax.set_title("Comparison with baseline / SOTA sequence-based temperature predictors")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig12_sota_comparison.png", dpi=200)
    print("→ Fig12_sota_comparison.png")


if __name__ == "__main__":
    main()
