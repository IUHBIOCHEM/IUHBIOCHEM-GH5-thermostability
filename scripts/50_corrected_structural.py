"""
50_corrected_structural.py

FIX: insert-aware column->residue mapping (bản cũ đếm match-column bỏ qua insert -> lệch).
Tính lại toàn bộ phần cấu trúc: hotspot 3AMC đúng, RSA/surface, distance tới Glu xúc tác (136/253),
và chọn lại đột biến engineering trên 3PZT. Regenerate Fig15.

Output: ai_training/analysis/hotspot_supptable.csv (corrected)
        ai_training/analysis/group_stats_fdr.csv
        ai_training/analysis/engineering_mutations.csv
        foldx_run/individual_list_singles.txt  (đột biến đúng để chạy FoldX)
        manuscript/figures/Fig15_engineering.png
"""
import os
import numpy as np
import pandas as pd
import pyhmmer, freesasa
from scipy.stats import fisher_exact, mannwhitneyu
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HMM = "pfam/PF00150.hmm"; alph = pyhmmer.easel.Alphabet.amino()
CHARGED = set("DEKR"); AROM = set("FWY")
CATS = {"E": {"E"}, "K": {"K"}, "D": {"D"}, "R": {"R"}, "Aromatic": AROM,
        "P": {"P"}, "G": {"G"}, "Q": {"Q"}, "H": {"H"}}
T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
       "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
# geometrically + conservation verified catalytic pairs
CAT3AMC = [136, 253]; CAT3PZT = [143, 186]


def _s(v): return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1): prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def struct(pdb):
    m = PDBParser(QUIET=True).get_structure("x", pdb)[0]
    ch = max(m, key=lambda c: sum(1 for r in c if r.id[0] == " "))
    r = [(x.id[1], T2O[x.resname], x["CA"].coord) for x in ch if x.id[0] == " " and x.resname in T2O and "CA" in x]
    seq = "".join(a for _, a, _ in r); nums = [n for n, _, _ in r]
    ca = {n: c for n, _, c in r}
    st = freesasa.calc(freesasa.Structure(pdb)).residueAreas()
    rsa = {}
    if ch.id in st:
        for rn in st[ch.id]:
            a = st[ch.id][rn]
            if a.relativeTotal is not None:
                try: rsa[int(rn)] = a.relativeTotal
                except ValueError: pass
    return seq, nums, ca, rsa


def correct_map(alnrow, nums, mset):
    """insert-aware: walk row, advance ptr on any letter, map only at match columns."""
    mp = {}; ptr = 0
    for j, c in enumerate(alnrow):
        if c in ("-", "."): continue
        if c.isalpha():
            if ptr < len(nums):
                rn = nums[ptr]
                if j in mset: mp[j] = rn
            ptr += 1
    return mp


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f: hmm = f.read()
    df = pd.read_csv("ai_training/gh5_all_verified.csv")
    ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]].dropna()
    om = dict(zip(ogt["Accession"].astype(str).str.split(".").str[0], ogt["OGT"]))
    s_th, n_th, ca_th, rsa_th = struct("ai_training/structures/3AMC.pdb")
    s_me, n_me, ca_me, rsa_me = struct("ai_training/structures/3PZT.pdb")

    seqs, names = [], []
    for _, r in df.iterrows():
        s = "".join(c for c in str(r["Sequence"]).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alph)); names.append(str(r["Accession"]))
    seqs.append(pyhmmer.easel.TextSequence(name=b"TH", sequence=s_th).digitize(alph)); names.append("TH")
    seqs.append(pyhmmer.easel.TextSequence(name=b"ME", sequence=s_me).digitize(alph)); names.append("ME")
    msa = pyhmmer.hmmalign(hmm, seqs, trim=False)
    aln = [_s(x) for x in msa.alignment]; mn = [_s(n) for n in msa.names]; arr = np.array([list(x) for x in aln])
    n2i = {n: i for i, n in enumerate(mn)}
    mset = set(j for j in range(arr.shape[1]) if not any((c.islower() or c == ".") for c in arr[:, j]))
    mcols = sorted(mset)
    c2n_th = correct_map(aln[n2i["TH"]], n_th, mset)
    c2n_me = correct_map(aln[n2i["ME"]], n_me, mset)
    thermo = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] >= 55]
    meso = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] < 55]
    print(f"thermo={len(thermo)} meso={len(meso)} match_cols={len(mcols)}")

    def col_vals(idxs, j):
        v = arr[idxs, j]; return v[(v != "-") & (v != ".")]

    # per-category enrichment (Fig 15a) — alignment-based
    recs = []
    for j in mcols:
        tn = col_vals(thermo, j); me = col_vals(meso, j)
        if len(tn) < 20 or len(me) < 20: continue
        for cat, mem in CATS.items():
            tc = sum(c in mem for c in tn); mc = sum(c in mem for c in me)
            _, p = fisher_exact([[tc, len(tn) - tc], [mc, len(me) - mc]])
            recs.append({"col": j, "category": cat, "delta": tc / len(tn) - mc / len(me), "p": p})
    R = pd.DataFrame(recs); R["fdr"] = bh(R["p"].values)
    sig_cat = R[(R["fdr"] < 0.05) & (R["delta"] > 0)]

    # charge hotspots
    hrec = []
    for j in mcols:
        tn = col_vals(thermo, j); me = col_vals(meso, j)
        if len(tn) < 20 or len(me) < 20: continue
        tf = sum(c in CHARGED for c in tn) / len(tn); mf = sum(c in CHARGED for c in me) / len(me)
        _, p = fisher_exact([[sum(c in CHARGED for c in tn), len(tn) - sum(c in CHARGED for c in tn)],
                             [sum(c in CHARGED for c in me), len(me) - sum(c in CHARGED for c in me)]])
        hrec.append({"col": j, "thermo_charged": tf, "meso_charged": mf, "delta": tf - mf, "p": p})
    H = pd.DataFrame(hrec); H["fdr"] = bh(H["p"].values)
    hot = H[(H["fdr"] < 0.05) & (H["delta"] > 0)].copy()
    print(f"charge hotspots: {len(hot)}")

    def pref(j):
        v = col_vals(thermo, j)
        if len(v) == 0: return "-"
        u, c = np.unique(v, return_counts=True); return u[c.argmax()]

    cat_co_th = np.array([ca_th[r] for r in CAT3AMC])
    rows = []
    for _, r in hot.iterrows():
        j = int(r["col"]); rn = c2n_th.get(j)
        rsa = rsa_th.get(rn, np.nan) if rn else np.nan
        dist = float(min(np.sqrt(((ca_th[rn] - cat_co_th) ** 2).sum(1)))) if (rn in ca_th) else np.nan
        rows.append({"PF00150_col": j, "res3AMC": rn,
                     "res3AMC_aa": (s_th[n_th.index(rn)] if rn in n_th else "-"),
                     "thermo_pref_residue": pref(j),
                     "thermo_charged_%": round(100 * r["thermo_charged"], 1),
                     "meso_charged_%": round(100 * r["meso_charged"], 1),
                     "delta_%": round(100 * r["delta"], 1),
                     "RSA": round(rsa, 2) if not np.isnan(rsa) else np.nan,
                     "location": ("surface" if (not np.isnan(rsa) and rsa > 0.25) else "core") if not np.isnan(rsa) else "NA",
                     "active_site": ("proximal" if (not np.isnan(dist) and dist < 12) else "distal") if not np.isnan(dist) else "NA",
                     "dist_catGlu_A": round(dist, 1) if not np.isnan(dist) else np.nan, "fdr": r["fdr"]})
    supp = pd.DataFrame(rows).sort_values("delta_%", ascending=False)
    supp.to_csv("ai_training/analysis/hotspot_supptable.csv", index=False)
    mapped = supp.dropna(subset=["dist_catGlu_A"])
    surf = int((supp["location"] == "surface").sum()); distal = int((mapped["active_site"] == "distal").sum())
    alld = [float(min(np.sqrt(((ca_th[k] - cat_co_th) ** 2).sum(1)))) for k in ca_th]
    _, pdist = mannwhitneyu(mapped["dist_catGlu_A"], alld, alternative="greater")
    print(f"mappable={len(mapped)} surface={surf} core={len(supp)-surf} distal={distal} "
          f"hotspot_meanDist={mapped['dist_catGlu_A'].mean():.1f} bg={np.mean(alld):.1f} p_distal={pdist:.3f}")
    nonhot = np.array([rsa_th[c2n_th[j]] for j in mcols if j in c2n_th and c2n_th[j] in rsa_th and j not in set(hot["col"])])
    _, psurf = mannwhitneyu(mapped["RSA"].dropna(), nonhot, alternative="greater")
    print(f"surface enrich p={psurf:.2f}; %surface hotspots={100*(mapped['RSA']>0.25).mean():.0f}")

    # group stats (alignment-based; unchanged) — recompute for consistency
    def feats(s):
        n = len(s); comp = {a: s.count(a) / n for a in "ACDEFGHIKLMNPQRSTVWY"}; pa = ProteinAnalysis(s)
        return {"E+K %": 100*(comp["E"]+comp["K"]), "D+R %": 100*(comp["D"]+comp["R"]), "Aromaticity": pa.aromaticity(),
                "Pro/Gly ratio": comp["P"]/comp["G"] if comp["G"] else np.nan, "GRAVY": pa.gravy(),
                "Aliphatic index": 100*(comp["A"]+2.9*comp["V"]+3.9*(comp["I"]+comp["L"]))}
    fd = []
    for i in thermo + meso:
        base = mn[i].split(".")[0]; row = df[df["Accession"].astype(str).str.split(".").str[0] == base]
        if len(row):
            s = "".join(c for c in str(row.iloc[0]["Sequence"]).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            f = feats(s); f["grp"] = "thermo" if i in thermo else "meso"; fd.append(f)
    FD = pd.DataFrame(fd); srows = []
    for mm in ["E+K %", "D+R %", "Aromaticity", "Pro/Gly ratio", "GRAVY", "Aliphatic index"]:
        a = FD[FD["grp"] == "thermo"][mm].dropna(); b = FD[FD["grp"] == "meso"][mm].dropna()
        _, p = mannwhitneyu(a, b); srows.append({"metric": mm, "thermo_mean": a.mean(), "meso_mean": b.mean(), "p": p})
    Sg = pd.DataFrame(srows); Sg["fdr"] = bh(Sg["p"].values); Sg.to_csv("ai_training/analysis/group_stats_fdr.csv", index=False)

    # ENGINEERING on 3PZT (correct mapping): surface, distal, meso uncharged, thermo -> E/K
    cat_co_me = np.array([ca_me[r] for r in CAT3PZT])
    mut = []
    for _, r in hot.iterrows():
        j = int(r["col"]); rn = c2n_me.get(j)
        if rn is None: continue
        wt = s_me[n_me.index(rn)] if rn in n_me else None
        if wt is None or wt in CHARGED: continue
        to = pref(j)
        if to not in set("EK"): continue
        rsa = rsa_me.get(rn, np.nan)
        if np.isnan(rsa) or rsa <= 0.25: continue
        dist = float(min(np.sqrt(((ca_me[rn] - cat_co_me) ** 2).sum(1))))
        if dist < 12: continue
        mut.append({"mutation": f"{wt}{rn}{to}", "wt": wt, "pos": rn, "to": to,
                    "thermo_charged_%": round(100 * r["thermo_charged"], 1), "meso_charged_%": round(100 * r["meso_charged"], 1),
                    "delta_%": round(100 * r["delta"], 1), "RSA_3PZT": round(rsa, 2), "dist_catGlu_A": round(dist, 1)})
    M = pd.DataFrame(mut).drop_duplicates("mutation").sort_values("delta_%", ascending=False)
    M.to_csv("ai_training/analysis/engineering_candidates_all.csv", index=False)
    M.head(9).to_csv("ai_training/analysis/engineering_mutations.csv", index=False)  # top for scanning
    print(f"\nEngineering candidates (correct 3PZT positions): {len(M)}")
    print(M.head(12).to_string(index=False))
    # foldx singles list (chain from 3PZT largest chain)
    mchain = max(PDBParser(QUIET=True).get_structure("x", "ai_training/structures/3PZT.pdb")[0],
                 key=lambda c: sum(1 for r in c if r.id[0] == " ")).id
    os.makedirs("foldx_run", exist_ok=True)
    with open("foldx_run/individual_list_singles.txt", "w") as f:
        for _, r in M.head(9).iterrows():
            f.write(f"{r['wt']}{mchain}{r['pos']}{r['to']};\n")
    print("chain", mchain, "-> foldx_run/individual_list_singles.txt")

    # ---- Fig 15 ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cc = sig_cat["category"].value_counts().reindex(list(CATS.keys())).fillna(0)
    ax[0].bar(cc.index, cc.values, color=["#c0392b" if c in ("E", "K") else "#888" for c in cc.index])
    ax[0].set_ylabel("# thermophile-enriched positions\n(FDR<0.05)"); ax[0].set_title("a  Enrichment by residue type")
    tab = supp.groupby(["location", "active_site"]).size().unstack(fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=ax[1], color=["#cc5544", "#6699cc"][:tab.shape[1]])
    ax[1].set_title("b  Location of charge hotspots"); ax[1].set_ylabel("count"); ax[1].tick_params(axis="x", rotation=0)
    ax[2].boxplot([nonhot, mapped["RSA"].dropna()], tick_labels=["other", "hotspot"], patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor="#6699cc", alpha=0.7))
    ax[2].axhline(0.25, color="r", ls="--", lw=1, label="surface cutoff")
    ax[2].set_ylabel("Relative solvent accessibility"); ax[2].set_title(f"c  Surface (p={psurf:.2f})"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig15_engineering.png", dpi=200)
    print("→ Fig15_engineering.png")
    # save summary numbers
    pd.DataFrame([{"mappable": len(mapped), "surface": surf, "distal": distal,
                   "hotspot_meanDist": round(mapped["dist_catGlu_A"].mean(), 1), "bg_meanDist": round(np.mean(alld), 1),
                   "p_distal": round(pdist, 3), "p_surface": round(psurf, 2),
                   "pct_surface": round(100 * (mapped["RSA"] > 0.25).mean())}]).to_csv(
        "ai_training/analysis/structural_summary.csv", index=False)


if __name__ == "__main__":
    main()
