"""
32_xai_shap.py — Explainable AI cho dự đoán nhiệt độ (OGT).

Huấn luyện mô hình trên các đặc trưng SINH HỌC diễn giải được (10 chỉ số) rồi giải thích
bằng SHAP (hướng & độ lớn tác động) và permutation importance (đánh giá không rò rỉ theo cụm).

Output: manuscript/figures/Fig5_SHAP.png
        manuscript/figures/Fig5b_permutation_importance.png
        ai_training/analysis/xai_importance.csv
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"
IVYWREL = set("IVYWREL")


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def feats(seq):
    s = clean(seq); n = len(s)
    if n < 40: return None
    pa = ProteinAnalysis(s); comp = {a: s.count(a) / n for a in AA}
    return {
        "Aliphatic index": 100 * (comp["A"] + 2.9 * comp["V"] + 3.9 * (comp["I"] + comp["L"])),
        "Aromaticity": pa.aromaticity(),
        "GRAVY": pa.gravy(),
        "Instability": pa.instability_index(),
        "Charged %": 100 * (comp["D"] + comp["E"] + comp["K"] + comp["R"]),
        "E+K %": 100 * (comp["E"] + comp["K"]),
        "IVYWREL %": 100 * sum(comp[a] for a in IVYWREL),
        "Arg/(Arg+Lys)": comp["R"] / (comp["R"] + comp["K"]) if (comp["R"] + comp["K"]) else 0.0,
        "Gln+His %": 100 * (comp["Q"] + comp["H"]),
        "Proline %": 100 * comp["P"],
    }


def main():
    os.makedirs("ai_training/analysis", exist_ok=True)
    df = pd.read_csv("ai_training/gh5_ogt_labeled.csv").dropna(subset=["OGT"]).reset_index(drop=True)
    g = pd.read_csv("ai_training/groups_map_ogt_cdhit70.csv")
    df["_acc"] = df["Accession"].astype(str).str.split(".").str[0]
    g["_acc"] = g["Accession"].astype(str).str.split(".").str[0]
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner")

    rows = [feats(s) for s in df["Sequence"]]
    keep = [i for i, r in enumerate(rows) if r]
    X = pd.DataFrame([rows[i] for i in keep])
    y = df["OGT"].values[keep]
    groups = df["ClusterID"].values[keep]
    print(f"n={len(X)} features={list(X.columns)}")

    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # SHAP
    expl = shap.TreeExplainer(rf)
    sv = expl.shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(8, 5))
    plt.title("SHAP: contribution of sequence features to predicted OGT", fontsize=11)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig5_SHAP.png", dpi=200); plt.close()

    # Permutation importance (không rò rỉ: đánh giá OOF theo cụm)
    gkf = GroupKFold(5)
    imp_acc = np.zeros(X.shape[1])
    for tr, te in gkf.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X.iloc[tr], y[tr])
        pi = permutation_importance(m, X.iloc[te], y[te], n_repeats=20, random_state=0, n_jobs=-1)
        imp_acc += pi.importances_mean
    imp_acc /= 5
    order = np.argsort(imp_acc)
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(X.columns)[order], imp_acc[order], color="#279")
    plt.xlabel("Permutation importance (mean ΔMSE, grouped CV)")
    plt.title("Feature importance for OGT prediction")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig5b_permutation_importance.png", dpi=200); plt.close()

    out = pd.DataFrame({"feature": X.columns, "permutation_importance": imp_acc,
                        "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    out.to_csv("ai_training/analysis/xai_importance.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("→ Fig5_SHAP.png, Fig5b_permutation_importance.png")


if __name__ == "__main__":
    main()
