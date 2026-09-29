"""60_review_supplementary_figures.py — build Supplementary Figures 15-18 added during revision.

Reads the analysis outputs produced by scripts 56-59 and regenerates the four revision figures.
Run from the repository root after scripts 56, 57, 58 and 59.

Outputs: manuscript/figures/Supp15_verification.png ... Supp18_crossfamily.png
"""
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr, mannwhitneyu
from Bio.PDB import PDBParser
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "ai_training", "analysis")
FIG = os.path.join(ROOT, "manuscript", "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
TEAL, RED, BLUE = "#1B9E77", "#C0392B", "#2166AB"


def fig15():
    v = dict(zip(pd.read_csv(f"{A}/full_query_verification.csv").stage,
                 pd.read_csv(f"{A}/full_query_verification.csv")["count"]))
    tot = int(v["full query"]); npass = int(v["passed PF00150 (true GH5)"]); fail = tot - npass
    ann = int(v["annotated 'glycoside hydrolase family 5'"]); ap = int(v["  of these passed"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].bar([f"Full query\n({tot:,})", f"Verified GH5\n({npass:,})", f"Not GH5\n({fail:,})"],
              [tot, npass, fail], color=["#95a5a6", TEAL, RED])
    ax[0].set_ylabel("sequences"); ax[0].set_title("Domain verification of the complete query", fontweight="bold", fontsize=11)
    ax[0].text(1, npass + tot * 0.09, f"{100*npass/tot:.1f}%", ha="center", color=TEAL, fontweight="bold")
    ax[1].bar([f"pass ({ap:,})\n{100*ap/ann:.0f}%", f"fail ({ann-ap})\n{100*(ann-ap)/ann:.0f}%"],
              [ap, ann - ap], color=[TEAL, RED])
    ax[1].set_title(f'Sequences annotated\n"glycoside hydrolase family 5" ({ann:,})', fontweight="bold", fontsize=11)
    ax[1].set_ylabel("sequences")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Supp15_verification.png", dpi=300, bbox_inches="tight"); plt.close()


def fig16():
    sub = pd.read_csv(f"{A}/gh5_757_subfamily.csv")
    cnt = {k: v for k, v in sorted(Counter(sub.subf).items(), key=lambda x: -x[1]) if v >= 5}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(list(cnt), list(cnt.values()), color=TEAL)
    ax[0].set_ylabel("sequences (of 757)"); ax[0].set_title("GH5 subfamily composition of the dataset", fontweight="bold", fontsize=11)
    ax[0].tick_params(axis="x", rotation=45); ax[0].spines[["top", "right"]].set_visible(False)
    sig = pd.read_csv(f"{A}/signature_scores.csv").merge(sub[["Accession", "subf"]], on="Accession", how="left")
    g = sig[(sig.subf == "GH5_2") & sig.OGT.notna()]
    r = spearmanr(g.OGT, g.signature)
    ax[1].scatter(g.OGT, g.signature, s=22, alpha=0.6, color=BLUE, edgecolor="none")
    ax[1].set_xlabel("OGT (°C)"); ax[1].set_ylabel("charge signature")
    ax[1].set_title(f"Signature vs OGT within GH5_2\n(rho = {r.correlation:.2f}, P = {r.pvalue:.0e}, n={len(g)})",
                    fontweight="bold", fontsize=11)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Supp16_subfamily.png", dpi=300, bbox_inches="tight"); plt.close()


def fig17():
    ch = next(iter(PDBParser(QUIET=True).get_structure("3AMC", f"{ROOT}/ai_training/structures/3AMC.pdb")[0]))
    bf = {r.id[1]: [a for a in r if a.name == "CA"][0].bfactor for r in ch
          if r.id[0] == " " and any(a.name == "CA" for a in r)}
    hs = pd.read_csv(f"{A}/hotspot_supptable.csv")["res3AMC"].dropna().astype(int).tolist()
    allb = list(bf.values()); hb = [bf[r] for r in hs if r in bf]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    bp = ax.boxplot([allb, hb], labels=[f"all residues\n(n={len(allb)})", f"hotspots\n(n={len(hb)})"],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#95a5a6", TEAL]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("C-alpha B-factor (A^2)")
    ax.set_title(f"Hotspots occupy more flexible positions\n(Mann-Whitney P = {mannwhitneyu(hb, allb, alternative='greater').pvalue:.3f})",
                 fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Supp17_bfactor.png", dpi=300, bbox_inches="tight"); plt.close()


def fig18():
    c = pd.read_csv(f"{A}/cross_family_hotspot_location.csv").set_index("family")
    fams = ["GH5", "GH1", "GH10"]
    surf = [c.loc[f, "pct_surface"] for f in fams]; distal = [c.loc[f, "pct_distal"] for f in fams]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.bar(x - w / 2, distal, w, label="distal (>12 A from active site)", color=TEAL)
    ax.bar(x + w / 2, surf, w, label="surface-exposed", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(fams); ax.set_ylabel("% of mapped hotspots"); ax.set_ylim(0, 105)
    ax.set_title("Hotspots are peripheral across TIM-barrel families", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    for i, (dd, ss) in enumerate(zip(distal, surf)):
        ax.text(i - w / 2, dd + 1.5, f"{dd:.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, ss + 1.5, f"{ss:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/Supp18_crossfamily.png", dpi=300, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    fig15(); fig16(); fig17(); fig18()
    print("Supplementary Figures 15-18 written to manuscript/figures/")
