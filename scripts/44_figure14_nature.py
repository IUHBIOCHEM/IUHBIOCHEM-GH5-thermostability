"""
44_figure14_nature.py — Figure 14 kiểu Nature (4 panel):
  (a) Volcano plot: Δcharged (thermo−meso) vs −log10(FDR), hotspot highlight.
  (b) Hotspot heatmap: top vị trí × [thermo%, meso%] tần suất tích điện.
  (c) PyMOL structural mapping (render thật từ PyMOL.app): hotspot đỏ trên GH5 fold, Glu xúc tác vàng.
  (d) Candidate score: signature vs predicted OGT, top candidates highlight.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import subprocess
import shutil
import numpy as np
import pandas as pd
import pyhmmer
from scipy.stats import fisher_exact, spearmanr
from Bio.PDB import PDBParser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HMM = "pfam/PF00150.hmm"
alphabet = pyhmmer.easel.Alphabet.amino()
CHARGED = set("DEKR")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "_scratch"))
os.makedirs(SCR, exist_ok=True)
PML = os.environ.get("PYMOL") or shutil.which("pymol") or "/Applications/PyMOL.app/Contents/MacOS/PyMOL"
T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
       "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def _s(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def matchcols(aln):
    return [j for j in range(len(aln[0])) if all((r[j] in ("-", ".")) or r[j].isupper() for r in aln)]


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f:
        hmm = f.read()
    df = pd.read_csv("ai_training/gh5_all_verified.csv")
    ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]].dropna()
    om = dict(zip(ogt["Accession"].astype(str).str.split(".").str[0], ogt["OGT"]))

    # reference 3AMC seq + resnums
    m = PDBParser(QUIET=True).get_structure("x", "ai_training/structures/3AMC.pdb")[0]
    ch = max(m, key=lambda c: sum(1 for r in c if r.id[0] == " "))
    refseq, refnums = "", []
    for r in ch:
        if r.id[0] == " " and r.resname in T2O and "CA" in r:
            refseq += T2O[r.resname]; refnums.append(r.id[1])

    seqs, names = [], []
    for _, r in df.iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alphabet)); names.append(str(r["Accession"]))
    seqs.append(pyhmmer.easel.TextSequence(name=b"REF", sequence=refseq).digitize(alphabet)); names.append("REF")
    msa = pyhmmer.hmmalign(hmm, seqs, trim=False)
    aln = [_s(x) for x in msa.alignment]; mn = [_s(n) for n in msa.names]
    arr = np.array([list(x) for x in aln]); mc = matchcols(aln); n2i = {n: i for i, n in enumerate(mn)}
    thermo = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] >= 55]
    meso = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] < 55]

    rec = []
    for j in mc:
        tn = arr[thermo, j]; tn = tn[tn != "-"]; mnn = arr[meso, j]; mnn = mnn[mnn != "-"]
        if len(tn) < 20 or len(mnn) < 20:
            continue
        tc = sum(c in CHARGED for c in tn); mcc = sum(c in CHARGED for c in mnn)
        _, p = fisher_exact([[tc, len(tn) - tc], [mcc, len(mnn) - mcc]])
        rec.append({"col": j, "tf": tc / len(tn), "mf": mcc / len(mnn), "delta": tc / len(tn) - mcc / len(mnn), "p": p})
    C = pd.DataFrame(rec); C["fdr"] = bh(C["p"].values)
    C["sig"] = (C["fdr"] < 0.05) & (C["delta"] > 0)

    # map cols -> 3AMC resnum (INSERT-AWARE: walk all columns, advance ptr on any letter)
    refrow = aln[n2i["REF"]]; c2n = {}; pos = 0; mcset = set(mc)
    for j, ch in enumerate(refrow):
        if ch in ("-", "."):
            continue
        if ch.isalpha():
            if pos < len(refnums):
                if j in mcset:
                    c2n[j] = refnums[pos]
                pos += 1
    # catalytic Glu = 2 cột E bảo tồn cao nhất
    Ef = sorted([(j, (arr[:, j][arr[:, j] != "-"] == "E").mean()) for j in mc], key=lambda x: -x[1])[:2]
    cat_nums = [136, 253]  # geometrically verified catalytic Glu pair (OE-OE 4.8 A) in 3AMC
    hot_nums = sorted({c2n[int(j)] for j in C[C["sig"]]["col"] if int(j) in c2n})

    # ---- PyMOL render ----
    pml = f"{SCR}/fig14_pymol.pml"; png = f"{SCR}/fig14_structure.png"
    hot_sel = "+".join(map(str, hot_nums)); cat_sel = "+".join(map(str, cat_nums))
    with open(pml, "w") as f:
        f.write(f"""
load ai_training/structures/3AMC.pdb, s
hide everything
bg_color white
remove not chain A
remove solvent
show cartoon, s
color grey80, s
set cartoon_transparency, 0.35
select hots, resi {hot_sel}
show spheres, hots and name CA
set sphere_scale, 0.6, hots
color firebrick, hots
select cat, resi {cat_sel}
show sticks, cat and not name C+N+O
color gold, cat
show spheres, cat and name CA
set sphere_scale, 0.8, cat and name CA
orient s
turn y, 20
set ray_shadows, 0
set antialias, 2
ray 1500, 1300
png {png}, dpi=200
""")
    subprocess.run([PML, "-cq", pml], cwd=ROOT, timeout=300)
    print("PyMOL render:", os.path.exists(png), "| hotspots:", len(hot_nums), "| catalytic:", cat_nums)

    # ---- candidate data ----
    sig = pd.read_csv("ai_training/analysis/signature_scores.csv")
    pred = pd.read_csv("ai_training/analysis/all757_predictions.csv")
    pred["_a"] = pred["Accession"].astype(str).str.split(".").str[0]
    sig["_a"] = sig["Accession"].astype(str).str.split(".").str[0]
    pp = pred.merge(sig[["_a", "signature"]], on="_a", how="left").dropna(subset=["signature", "OGT_pred"])
    tc = pd.read_csv("ai_training/analysis/top_candidates_validated.csv")
    tc["_a"] = tc["Accession"].astype(str).str.split(".").str[0]
    tc = tc.merge(sig[["_a", "signature"]], on="_a", how="left")
    rho_pred = spearmanr(pp["signature"], pp["OGT_pred"]).correlation

    # ---- compose figure ----
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.24)
    # (a) volcano
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(C["delta"] * 100, -np.log10(C["p"] + 1e-300), s=22,
               c=np.where(C["sig"], "#c0392b", "#c8c8c8"), edgecolor="none")
    ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.7)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("Δ charged frequency (thermophile − mesophile), %")
    ax.set_ylabel("−log₁₀ p")
    ax.set_title(f"a   Volcano: {int(C['sig'].sum())} thermal hotspots (FDR<0.05)", fontweight="bold", loc="left")
    # (b) heatmap top 25 by delta (chỉ vị trí map được sang 3AMC)
    sig_mapped = C[C["sig"] & C["col"].map(lambda j: int(j) in c2n)]
    top = sig_mapped.sort_values("delta", ascending=False).head(25).copy()
    top["lab"] = [f"{c2n[int(j)]}" for j in top["col"]]
    hmap = np.vstack([top["tf"].values, top["mf"].values]) * 100
    axb = fig.add_subplot(gs[0, 1])
    im = axb.imshow(hmap, aspect="auto", cmap="RdBu_r", vmin=0, vmax=100)
    axb.set_yticks([0, 1]); axb.set_yticklabels(["Thermophile", "Mesophile"])
    axb.set_xticks(range(len(top))); axb.set_xticklabels(top["lab"], rotation=90, fontsize=6)
    axb.set_xlabel("3AMC residue position")
    axb.set_title("b   Charge frequency at top hotspots", fontweight="bold", loc="left")
    fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04, label="% charged")
    # (c) PyMOL structure
    axc = fig.add_subplot(gs[1, 0])
    if os.path.exists(png):
        axc.imshow(mpimg.imread(png)); axc.axis("off")
    axc.set_title("c   Hotspots (red) on GH5 fold; catalytic Glu (gold)", fontweight="bold", loc="left")
    # (d) candidate score
    axd = fig.add_subplot(gs[1, 1])
    axd.scatter(pp["signature"] * 100, pp["OGT_pred"], s=14, alpha=0.3, color="#b8b8b8", label="all GH5 (n=757)")
    axd.scatter(tc["signature"] * 100, tc["Predicted_OGT_C"], s=60, color="#c0392b", edgecolor="k",
                zorder=3, label="top validated candidates")
    axd.set_xlabel("GH5 thermal-charge signature (%)"); axd.set_ylabel("Model predicted OGT (°C)")
    axd.set_title(f"d   Signature explains ranking (ρ={rho_pred:.2f})", fontweight="bold", loc="left")
    axd.legend(fontsize=8, loc="lower right")
    fig.savefig("manuscript/figures/Fig14_discovery.png", dpi=220, bbox_inches="tight")
    print("→ Fig14_discovery.png (Nature-style, replaced)")


if __name__ == "__main__":
    main()
