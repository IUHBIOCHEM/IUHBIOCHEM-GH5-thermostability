"""
31_seq_feature_comparison.py

So sánh đặc trưng trình tự giữa nhóm CHỊU NHIỆT và ƯA ẤM (theo OGT, ngưỡng 55°C).
Tính các chỉ số liên quan bền nhiệt đã biết, kiểm định Mann-Whitney, cỡ hiệu ứng (Cliff's delta),
xuất bảng thống kê + hình boxplot chất lượng công bố.

Output: manuscript/figures/Fig4_seq_features.png
        ai_training/analysis/seq_feature_stats.csv
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis

OUT_FIG = "manuscript/figures/Fig4_seq_features.png"
OUT_CSV = "ai_training/analysis/seq_feature_stats.csv"
THR = 55.0

AA = "ACDEFGHIKLMNPQRSTVWY"
# Tập residue "IVYWREL" tăng ở ưa nhiệt (Zeldovich et al. 2007)
IVYWREL = set("IVYWREL")


def clean(s):
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(s).upper())


def feats(seq):
    s = clean(seq)
    n = len(s)
    if n < 40:
        return None
    pa = ProteinAnalysis(s)
    comp = {a: s.count(a) / n for a in AA}
    charged = comp["D"] + comp["E"] + comp["K"] + comp["R"]
    ek = comp["E"] + comp["K"]
    arg_frac = comp["R"] / (comp["R"] + comp["K"]) if (comp["R"] + comp["K"]) > 0 else np.nan
    ali = 100 * (comp["A"] + 2.9 * comp["V"] + 3.9 * (comp["I"] + comp["L"]))
    return {
        "Aliphatic index": ali,
        "Aromaticity (FWY)": pa.aromaticity(),
        "GRAVY": pa.gravy(),
        "Instability index": pa.instability_index(),
        "Charged (DEKR) %": 100 * charged,
        "E+K %": 100 * ek,
        "IVYWREL %": 100 * sum(comp[a] for a in IVYWREL),
        "Arg/(Arg+Lys)": arg_frac,
        "Gln+His %": 100 * (comp["Q"] + comp["H"]),
        "Proline %": 100 * comp["P"],
    }


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def main():
    os.makedirs("ai_training/analysis", exist_ok=True)
    df = pd.read_csv("ai_training/gh5_ogt_labeled.csv").dropna(subset=["OGT"])
    rows = []
    for _, r in df.iterrows():
        f = feats(r["Sequence"])
        if f:
            f["thermo"] = int(r["OGT"] >= THR)
            rows.append(f)
    fd = pd.DataFrame(rows)
    thermo = fd[fd["thermo"] == 1]
    meso = fd[fd["thermo"] == 0]
    print(f"Thermostable (OGT>={THR:.0f}): {len(thermo)} | Mesophilic: {len(meso)}")

    metrics = [c for c in fd.columns if c != "thermo"]
    stats = []
    for m in metrics:
        a, b = thermo[m].dropna(), meso[m].dropna()
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        d = cliffs_delta(a.values, b.values)
        stats.append({"Feature": m, "Thermo_mean": a.mean(), "Meso_mean": b.mean(),
                      "Cliffs_delta": d, "p_value": p})
    sdf = pd.DataFrame(stats).sort_values("Cliffs_delta", key=abs, ascending=False)
    sdf.to_csv(OUT_CSV, index=False)
    print(sdf.round(3).to_string(index=False))

    # boxplot panel
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, m in zip(axes.ravel(), metrics):
        data = [meso[m].dropna(), thermo[m].dropna()]
        bp = ax.boxplot(data, labels=["Meso", "Thermo"], patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], ["#6699cc", "#cc5544"]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        p = sdf.loc[sdf["Feature"] == m, "p_value"].values[0]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{m}\n{star} (p={p:.1e})", fontsize=9)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    print("→", OUT_FIG)


if __name__ == "__main__":
    main()
