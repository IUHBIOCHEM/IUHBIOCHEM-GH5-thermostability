"""59_cross_family_hotspot.py — structural location of charge hotspots across GH1, GH5 and GH10.

Recomputes the charged-residue hotspots for each family (Fisher exact test with Benjamini-Hochberg
FDR), maps them onto a reference (β/α)8 structure with an insert-aware alignment, and reports the
fraction that are surface-exposed and active-site-distal. Reproduces Supplementary Fig. 18.
Reference structures are downloaded from the RCSB PDB. Run from the repository root.

Output: ai_training/analysis/cross_family_hotspot_location.csv
"""
import os
import urllib.request
import warnings
import pyhmmer
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from Bio.PDB import PDBParser, ShrakeRupley
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "_scratch"))
os.makedirs(SCR, exist_ok=True)
A = os.path.join(ROOT, "ai_training", "analysis")
alpha = pyhmmer.easel.Alphabet.amino()
CHARGED = set("DEKR")
MAXASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104, "H": 224,
          "I": 197, "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285,
          "Y": 263, "V": 174}
T2O = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def pdb(code):
    p = os.path.join(SCR, f"{code}.pdb")
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{code}.pdb", p)
    return p


def analyze(csv, hmmfile, code, fam):
    df = pd.read_csv(csv).dropna(subset=["OGT"])
    df["thermo"] = df["OGT"] >= 55
    with pyhmmer.plan7.HMMFile(os.path.join(ROOT, "pfam", hmmfile)) as hf:
        hmm = hf.read()
    seqs = [(r.thermo, pyhmmer.easel.TextSequence(name=str(r.Accession).encode(),
             sequence=str(r.Sequence).replace("*", "").replace("-", "")).digitize(alpha))
            for r in df.itertuples() if len(str(r.Sequence)) >= 50]
    arr = np.array([list(str(a)) for a in pyhmmer.hmmalign(hmm, [s[1] for s in seqs]).alignment])
    lab = np.array([s[0] for s in seqs])
    mcols = [j for j in range(arr.shape[1]) if not (arr[0][j].islower() or arr[0][j] == ".")]
    pv, info = [], []
    for k, j in enumerate(mcols):
        col = arr[:, j]
        tc = sum(col[i] in CHARGED for i in range(len(col)) if lab[i]); tn = lab.sum() - tc
        mc = sum(col[i] in CHARGED for i in range(len(col)) if not lab[i]); mn = (~lab).sum() - mc
        pv.append(fisher_exact([[tc, tn], [mc, mn]])[1])
        info.append((k, tc / (tc + tn) - mc / (mc + mn), np.mean(col == "E")))
    fdr = multipletests(pv, method="fdr_bh")[1]
    hs = {info[i][0] for i in range(len(info)) if fdr[i] < 0.05 and info[i][1] > 0}
    cat = [o for o, _, _ in sorted(info, key=lambda x: -x[2])[:2]]
    # map onto structure (insert-aware) -> match-state ordinal -> residue number
    m = PDBParser(QUIET=True).get_structure(code, pdb(code))[0]
    ch = next(iter(m))
    resn = [r.id[1] for r in ch if r.id[0] == " " and r.resname in T2O]
    seq = "".join(T2O[r.resname] for r in ch if r.id[0] == " " and r.resname in T2O)
    ral = str(pyhmmer.hmmalign(hmm, [pyhmmer.easel.TextSequence(name=b"r", sequence=seq).digitize(alpha)]).alignment[0])
    o2r, ri, mo = {}, 0, -1
    for c in ral:
        is_m = not (c.islower() or c == ".")
        if c not in "-.":
            if is_m:
                mo += 1; o2r[mo] = resn[ri]; ri += 1
            else:
                ri += 1
        elif is_m:
            mo += 1
    ShrakeRupley().compute(m, level="R")
    by = {r.id[1]: r for r in ch if r.id[0] == " "}
    catr = [by[o2r[o]] for o in cat if o in o2r and o2r[o] in by]
    surf = distal = mapped = 0
    for o in hs:
        if o in o2r and o2r[o] in by:
            r = by[o2r[o]]; mapped += 1
            if r.sasa / MAXASA.get(T2O.get(r.resname, "A"), 200) > 0.15:
                surf += 1
            if "CA" in r and catr and min(np.linalg.norm(r["CA"].coord - c["CA"].coord)
                                          for c in catr if "CA" in c) > 12:
                distal += 1
    print(f"{fam} (ref {code}): {len(hs)} hotspots, {mapped} mapped | "
          f"surface {100*surf/mapped:.0f}% | distal {100*distal/mapped:.0f}%")
    return {"family": fam, "ref": code, "n_hotspots": len(hs), "mapped": mapped,
            "pct_surface": round(100 * surf / mapped, 1), "pct_distal": round(100 * distal / mapped, 1)}


def gh5_from_primary():
    """For GH5 use the primary insert-aware hotspot mapping (hotspot_supptable) so that the
    cross-family comparison is consistent with the main-text GH5 analysis (66 hotspots)."""
    supp = pd.read_csv(os.path.join(A, "hotspot_supptable.csv"))
    mapped = supp[supp["location"].isin(["surface", "core"])]
    surf = 100 * (mapped["location"] == "surface").mean()
    distal = 100 * (supp["dist_catGlu_A"].dropna() > 12).mean()
    print(f"GH5 (ref 3AMC, primary mapping): {len(supp)} hotspots, {len(mapped)} mapped | "
          f"surface {surf:.0f}% | distal {distal:.0f}%")
    return {"family": "GH5", "ref": "3AMC", "n_hotspots": len(supp), "mapped": len(mapped),
            "pct_surface": round(surf, 1), "pct_distal": round(distal, 1)}


if __name__ == "__main__":
    rows = [
        gh5_from_primary(),
        analyze(os.path.join(ROOT, "data_family", "GH1_labeled.csv"), "PF00232.hmm", "1OD0", "GH1"),
        analyze(os.path.join(ROOT, "data_family", "GH10_labeled.csv"), "PF00331.hmm", "1V0K", "GH10"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(A, "cross_family_hotspot_location.csv"), index=False)
