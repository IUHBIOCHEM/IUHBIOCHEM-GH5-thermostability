"""
38_saltbridge_survey.py

Khảo sát THỐNG KÊ salt bridge trên TẤT CẢ cấu trúc GH5 (PF00150) đã giải thực nghiệm,
thay cho so sánh 1-vs-1 minh họa. Tải toàn bộ cấu trúc thermophile vs mesophile,
đếm salt bridge thật (ion pair Asp/Glu–Lys/Arg/His < 4.0 Å) chuẩn hóa /100 residue,
và mật độ residue tích điện bề mặt. Kiểm định Mann-Whitney.

Output: manuscript/figures/Fig8_saltbridge_survey.png
        ai_training/analysis/saltbridge_survey.csv
"""
import os
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, NeighborSearch, ShrakeRupley
import warnings
warnings.filterwarnings("ignore")

SD = "ai_training/structures"
os.makedirs(SD, exist_ok=True)
os.makedirs("ai_training/analysis", exist_ok=True)

# Phân loại thermophile theo LOÀI (đã sửa: Ruminiclostridium cellulolyticum = mesophile)
THERMO_KEYS = ["Thermotoga", "Acetivibrio thermocellus", "Clostridium thermocellum",
               "Fervidobacterium", "Acidothermus", "Thermogutta terrifontis",
               "Caldicellulosiruptor", "Thermoanaerobacter", "Thermobifida"]
NEG = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
POS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
       ("HIS", "ND1"), ("HIS", "NE2")}
MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,"I":197,
          "L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H",
       "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def is_thermo(name):
    return any(k in name for k in THERMO_KEYS)


def fetch(pdbid):
    p = f"{SD}/{pdbid}.pdb"
    if not os.path.exists(p):
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdbid}.pdb", p)
        except Exception:
            return None
    return p


def analyse(pdbid):
    p = fetch(pdbid)
    if not p:
        return None
    try:
        s = PDBParser(QUIET=True).get_structure(pdbid, p)[0]
    except Exception:
        return None
    chain = next((c for c in s if sum(1 for r in c if r.id[0] == " ") > 100), None)
    if chain is None:
        return None
    residues = [r for r in chain if r.id[0] == " " and r.resname in T2O]
    n = len(residues)
    if n < 100:
        return None
    # salt bridges: charged atoms
    neg_atoms, pos_atoms = [], []
    for r in residues:
        for a in r:
            if (r.resname, a.name) in NEG:
                neg_atoms.append((r, a))
            elif (r.resname, a.name) in POS:
                pos_atoms.append((r, a))
    # đếm cặp residue có ít nhất 1 cặp nguyên tử < 4.0 Å
    pairs = set()
    if neg_atoms and pos_atoms:
        ns = NeighborSearch([a for _, a in pos_atoms])
        for rn, an in neg_atoms:
            for ap in ns.search(an.coord, 4.0):
                pairs.add((rn.id[1], ap.get_parent().id[1]))
    sb = len(pairs)
    # surface charged density
    try:
        ShrakeRupley().compute(s, level="R")
        surf_charged = sum(1 for r in residues if r.resname in ("ASP", "GLU", "LYS", "ARG")
                           and (r.sasa / MAXASA[T2O[r.resname]]) > 0.25)
    except Exception:
        surf_charged = np.nan
    return {"pdb": pdbid, "n_res": n, "salt_bridges": sb,
            "sb_per100": 100 * sb / n,
            "surf_charged_per100": 100 * surf_charged / n if not np.isnan(surf_charged) else np.nan}


def main():
    org = json.load(open(os.path.join(SD, "pdb_org.json")))
    rows = []
    for pid, name in org.items():
        if name == "NA":
            continue
        r = analyse(pid)
        if r:
            r["organism"] = name
            r["group"] = "thermophile" if is_thermo(name) else "mesophile"
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("ai_training/analysis/saltbridge_survey.csv", index=False)
    th = df[df["group"] == "thermophile"]; me = df[df["group"] == "mesophile"]
    print(f"Thermophile: {len(th)} structures / {th['organism'].nunique()} organisms")
    print(f"Mesophile:   {len(me)} structures / {me['organism'].nunique()} organisms")

    def compare(metric, unit):
        a, b = th[metric].dropna(), me[metric].dropna()
        U, p = mannwhitneyu(a, b, alternative="greater")
        print(f"  {metric}: thermo={a.mean():.2f} vs meso={b.mean():.2f} {unit}  "
              f"(Mann-Whitney one-sided p={p:.2e})")
        return a, b, p

    print("\n=== Structure-level comparison ===")
    sb_a, sb_b, sb_p = compare("sb_per100", "/100 res")
    sc_a, sc_b, sc_p = compare("surf_charged_per100", "/100 res")

    # per-organism aggregation (giảm pseudoreplication)
    org_mean = df.groupby(["organism", "group"])["sb_per100"].mean().reset_index()
    ot = org_mean[org_mean["group"] == "thermophile"]["sb_per100"]
    om = org_mean[org_mean["group"] == "mesophile"]["sb_per100"]
    _, org_p = mannwhitneyu(ot, om, alternative="greater")
    print(f"\nPer-organism (n={len(ot)} thermo vs {len(om)} meso): "
          f"thermo={ot.mean():.2f} vs meso={om.mean():.2f}, p={org_p:.3f}")

    # ---- Figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for k, (a, b, p, title, ylab) in enumerate([
            (sb_a, sb_b, sb_p, "(a) Salt bridges (ion pairs < 4 Å)", "Salt bridges / 100 residues"),
            (sc_a, sc_b, sc_p, "(b) Surface charged residues", "Surface D/E/K/R / 100 residues")]):
        bp = ax[k].boxplot([b, a], tick_labels=["Mesophile", "Thermophile"], patch_artist=True,
                           widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], ["#6699cc", "#cc5544"]):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        # jitter points
        for i, d in enumerate([b, a]):
            ax[k].scatter(np.random.normal(i + 1, 0.06, len(d)), d, s=14, color="k", alpha=0.35, zorder=3)
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        ax[k].set_ylabel(ylab)
        ax[k].set_title(f"{title}\nthermo n={len(a)}, meso n={len(b)}; {star} (p={p:.1e})", fontsize=10)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig8_saltbridge_survey.png", dpi=200)
    print("→ Fig8_saltbridge_survey.png")


if __name__ == "__main__":
    main()
