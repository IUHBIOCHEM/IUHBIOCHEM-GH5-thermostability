"""
34_structural_mapping.py

Structural mapping of thermostability determinants on experimentally solved GH5 structures.
So sánh mật độ residue tích điện BỀ MẶT (cầu muối) giữa một GH5 ưa nhiệt và một GH5 ưa ấm,
dùng solvent accessibility (Shrake-Rupley). Củng cố phát hiện mức trình tự (E+K tăng ở thermophile).

Structures: Thermotoga maritima GH5 (3AMC, thermophile) vs Bacillus GH5 (3PZT, mesophile).
Output: manuscript/figures/Fig8_structure.png
        ai_training/analysis/structure_surface_stats.csv
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from Bio.PDB import PDBParser, ShrakeRupley

SD = "ai_training/structures"
os.makedirs(SD, exist_ok=True)
os.makedirs("ai_training/analysis", exist_ok=True)

# Max ASA (Tien et al. 2013, theoretical) để tính RSA
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,
          "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
THREE2ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
             "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
             "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
POS = {"K","R"}; NEG = {"D","E"}


def fetch(pdbid):
    path = f"{SD}/{pdbid}.pdb"
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdbid}.pdb", path)
    return path


def analyse(pdbid, label):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdbid, fetch(pdbid))
    model = s[0]
    sr = ShrakeRupley()
    sr.compute(model, level="R")
    chain = next(c for c in model if any(r.id[0] == " " for r in c))
    rows = []
    for res in chain:
        if res.id[0] != " " or res.resname not in THREE2ONE:
            continue
        aa = THREE2ONE[res.resname]
        rsa = res.sasa / MAXASA[aa]
        ca = res["CA"].coord if "CA" in res else None
        rows.append({"aa": aa, "rsa": rsa, "surface": rsa > 0.25,
                     "x": ca[0] if ca is not None else np.nan,
                     "y": ca[1] if ca is not None else np.nan,
                     "z": ca[2] if ca is not None else np.nan})
    df = pd.DataFrame(rows).dropna(subset=["x"])
    n = len(df)
    surf = df[df["surface"]]
    stat = {
        "structure": pdbid, "label": label, "n_residues": n,
        "surface_charged_per100": 100 * surf["aa"].isin(POS | NEG).sum() / n,
        "surface_EK_per100": 100 * surf["aa"].isin({"E", "K"}).sum() / n,
        "surface_pos_per100": 100 * surf["aa"].isin(POS).sum() / n,
        "surface_neg_per100": 100 * surf["aa"].isin(NEG).sum() / n,
    }
    return df, stat


def main():
    (dft, st), (dfm, sm) = analyse("3AMC", "Thermophile (T. maritima)"), analyse("3PZT", "Mesophile (Bacillus)")
    stats = pd.DataFrame([st, sm])
    stats.to_csv("ai_training/analysis/structure_surface_stats.csv", index=False)
    print(stats.round(1).to_string(index=False))

    fig = plt.figure(figsize=(13, 6))
    for i, (df, title) in enumerate([(dft, st["label"]), (dfm, sm["label"])]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # backbone trace
        ax.plot(df["x"], df["y"], df["z"], color="#bbbbbb", lw=1.0, alpha=0.7)
        # surface charged residues
        surf = df[df["surface"]]
        pos = surf[surf["aa"].isin(POS)]; neg = surf[surf["aa"].isin(NEG)]
        ax.scatter(pos["x"], pos["y"], pos["z"], c="#2b6cb0", s=45, alpha=0.9, label="surface K/R (+)", depthshade=True)
        ax.scatter(neg["x"], neg["y"], neg["z"], c="#c0392b", s=45, alpha=0.9, label="surface D/E (−)", depthshade=True)
        dens = 100 * surf["aa"].isin(POS | NEG).sum() / len(df)
        ax.set_title(f"{title}\nsurface charged density = {dens:.1f}/100 res", fontsize=11)
        ax.legend(fontsize=8, loc="upper left"); ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("manuscript/figures/Fig8_structure.png", dpi=200)
    print("→ Fig8_structure.png")


if __name__ == "__main__":
    main()
