"""
42_hotspot_engineering.py

Mở rộng khám phá GH5 thành khuyến nghị thực nghiệm kiểm chứng được.

(A) Sequence: align toàn bộ GH5 verified + cấu trúc tham chiếu (3AMC thermo, 3PZT meso) vào PF00150;
    tần suất E,K,D,R,aromatic(FWY),P,G,Q,H theo từng vị trí; Fisher exact thermo vs meso + BH-FDR
    trên TẤT CẢ (vị trí × loại residue). Xác định vị trí giàu ở thermophile.
(B) Statistics: E+K, D+R, aromaticity, Pro/Gly ratio, GRAVY, aliphatic index thermo vs meso
    (Mann-Whitney + BH-FDR); kiểm tra hotspot có giàu E/K & surface không.
(C) Structure: map lên 3AMC (freesasa RSA), phân loại surface/core & proximal/distal (theo Glu xúc tác);
    kiểm định E/K-hotspot có tụ trên bề mặt không.
(D) Engineering: map lên GH5 mesophile 3PZT, chọn 3-5 đột biến điểm (→E/K) tại vị trí surface, distal,
    nơi meso đang mang residue không tích điện.

Output: manuscript/figures/Fig15_engineering.png
        ai_training/analysis/hotspot_supptable.csv
        ai_training/analysis/group_stats_fdr.csv
        ai_training/analysis/engineering_mutations.csv
"""
import os
import numpy as np
import pandas as pd
import pyhmmer
import freesasa
from scipy.stats import fisher_exact, mannwhitneyu
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HMM = "pfam/PF00150.hmm"
alphabet = pyhmmer.easel.Alphabet.amino()
os.makedirs("ai_training/analysis", exist_ok=True)
CHARGED = set("DEKR"); EK = set("EK"); DR = set("DR"); AROM = set("FWY")
T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
       "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
CATS = {"E": {"E"}, "K": {"K"}, "D": {"D"}, "R": {"R"}, "Aromatic": AROM,
        "P": {"P"}, "G": {"G"}, "Q": {"Q"}, "H": {"H"}}


def _s(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def bh_fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def match_columns(aln):
    return [j for j in range(len(aln[0])) if all((r[j] == "-") or r[j].isupper() for r in aln)]


def pdb_info(pdb):
    """seq, resnums, CA coords, RSA per resnum (freesasa, monomer = first chain)."""
    m = PDBParser(QUIET=True).get_structure("x", pdb)[0]
    chain = max(m, key=lambda c: sum(1 for r in c if r.id[0] == " "))
    seq, nums, coords = "", [], []
    for r in chain:
        if r.id[0] == " " and r.resname in T2O and "CA" in r:
            seq += T2O[r.resname]; nums.append(r.id[1]); coords.append(r["CA"].coord)
    st = freesasa.Structure(pdb)
    res = freesasa.calc(st).residueAreas()
    ch = chain.id
    rsa = {}
    if ch in res:
        for rn in res[ch]:
            a = res[ch][rn]
            if a.relativeTotal is not None:
                try:
                    rsa[int(rn)] = a.relativeTotal
                except ValueError:
                    pass
    return seq, nums, np.array(coords), rsa


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f:
        hmm = f.read()
    df = pd.read_csv("ai_training/gh5_all_verified.csv").reset_index(drop=True)
    ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]].dropna()
    ogt_map = dict(zip(ogt["Accession"].astype(str).str.split(".").str[0], ogt["OGT"]))

    seq_th, num_th, co_th, rsa_th = pdb_info("ai_training/structures/3AMC.pdb")   # thermophile ref
    seq_me, num_me, co_me, rsa_me = pdb_info("ai_training/structures/3PZT.pdb")    # mesophile ref

    seqs, names = [], []
    for _, r in df.iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alphabet)); names.append(str(r["Accession"]))
    for nm, sq in [("REF_3AMC", seq_th), ("REF_3PZT", seq_me)]:
        seqs.append(pyhmmer.easel.TextSequence(name=nm.encode(), sequence=sq).digitize(alphabet)); names.append(nm)

    msa = pyhmmer.hmmalign(hmm, seqs, trim=True)
    aln = [_s(x) for x in msa.alignment]; mnames = [_s(n) for n in msa.names]
    arr = np.array([list(x) for x in aln]); mcols = match_columns(aln)
    n2i = {n: i for i, n in enumerate(mnames)}

    thermo = [i for i, n in enumerate(mnames) if n.split(".")[0] in ogt_map and ogt_map[n.split(".")[0]] >= 55]
    meso = [i for i, n in enumerate(mnames) if n.split(".")[0] in ogt_map and ogt_map[n.split(".")[0]] < 55]
    print(f"thermo={len(thermo)}, meso={len(meso)}, cols={len(mcols)}")

    # ---- (A) per-position per-category Fisher ----
    recs = []
    for j in mcols:
        tcol = arr[thermo, j]; mcol = arr[meso, j]
        tn = tcol[tcol != "-"]; mn = mcol[mcol != "-"]
        if len(tn) < 20 or len(mn) < 20:
            continue
        for cat, members in CATS.items():
            tc = sum(c in members for c in tn); mc = sum(c in members for c in mn)
            _, p = fisher_exact([[tc, len(tn) - tc], [mc, len(mn) - mc]])
            recs.append({"col": j, "category": cat, "thermo_freq": tc / len(tn),
                         "meso_freq": mc / len(mn), "delta": tc / len(tn) - mc / len(mn), "p": p})
    R = pd.DataFrame(recs)
    R["fdr"] = bh_fdr(R["p"].values)
    sig = R[(R["fdr"] < 0.05) & (R["delta"] > 0)]
    print("Thermophile-enriched (FDR<0.05) by category:")
    print(sig["category"].value_counts().to_string())

    # ---- map columns -> structure residue numbers ----
    def col_to_num(refname, nums):
        row = arr[n2i[refname]]; mp = {}; pos = 0
        for j in mcols:
            if row[j] != "-":
                if pos < len(nums):
                    mp[j] = nums[pos]
                pos += 1
        return mp
    c2n_th = col_to_num("REF_3AMC", num_th)
    c2n_me = col_to_num("REF_3PZT", num_me)
    coord_th = {n: c for n, c in zip(num_th, co_th)}
    # catalytic Glu cols (2 most E-conserved)
    Ef = sorted([(j, (arr[:, j][arr[:, j] != "-"] == "E").mean()) for j in mcols], key=lambda x: -x[1])[:2]
    catcols = [j for j, _ in Ef]
    cat_coords_th = np.array([coord_th[136], coord_th[253]])  # geometrically verified catalytic Glu pair (3AMC)

    # ---- charge hotspots (E/K/D/R combined) per position for supp table ----
    hot_recs = []
    for j in mcols:
        tcol = arr[thermo, j]; mcol = arr[meso, j]
        tn = tcol[tcol != "-"]; mn = mcol[mcol != "-"]
        if len(tn) < 20 or len(mn) < 20:
            continue
        tf = sum(c in CHARGED for c in tn) / len(tn); mf = sum(c in CHARGED for c in mn) / len(mn)
        _, p = fisher_exact([[sum(c in CHARGED for c in tn), len(tn) - sum(c in CHARGED for c in tn)],
                             [sum(c in CHARGED for c in mn), len(mn) - sum(c in CHARGED for c in mn)]])
        hot_recs.append({"col": j, "thermo_charged": tf, "meso_charged": mf, "delta": tf - mf, "p": p})
    H = pd.DataFrame(hot_recs); H["fdr"] = bh_fdr(H["p"].values)
    hot = H[(H["fdr"] < 0.05) & (H["delta"] > 0)].copy()

    # thermophile-preferred residue per hotspot + structure annotation
    def top_res(col, idxs):
        vals = arr[idxs, col]; vals = vals[vals != "-"]
        if len(vals) == 0:
            return "-"
        u, c = np.unique(vals, return_counts=True); return u[c.argmax()]
    rows = []
    for _, r in hot.iterrows():
        j = int(r["col"]); rn = c2n_th.get(j); pref = top_res(j, thermo)
        rsa = rsa_th.get(rn, np.nan) if rn else np.nan
        dist = np.nan
        if rn in coord_th and len(cat_coords_th):
            dist = float(min(np.sqrt(((coord_th[rn] - cat_coords_th) ** 2).sum(1))))
        rows.append({"PF00150_col": j, "res3AMC": rn,
                     "res3AMC_aa": seq_th[num_th.index(rn)] if rn in num_th else "-",
                     "thermo_pref_residue": pref,
                     "thermo_charged_%": round(100 * r["thermo_charged"], 1),
                     "meso_charged_%": round(100 * r["meso_charged"], 1),
                     "delta_%": round(100 * r["delta"], 1),
                     "RSA": round(rsa, 2) if not np.isnan(rsa) else np.nan,
                     "location": ("surface" if (not np.isnan(rsa) and rsa > 0.25) else "core") if not np.isnan(rsa) else "NA",
                     "active_site": ("proximal" if (not np.isnan(dist) and dist < 12) else "distal") if not np.isnan(dist) else "NA",
                     "dist_catGlu_A": round(dist, 1) if not np.isnan(dist) else np.nan,
                     "fdr": r["fdr"]})
    supp = pd.DataFrame(rows).sort_values("delta_%", ascending=False)
    supp.to_csv("ai_training/analysis/hotspot_supptable.csv", index=False)
    print(f"\nCharge hotspots: {len(supp)} | surface {sum(supp['location']=='surface')} | "
          f"distal {sum(supp['active_site']=='distal')}")

    # ---- (B) group-level stats ----
    def feats(s):
        n = len(s); comp = {a: s.count(a) / n for a in "ACDEFGHIKLMNPQRSTVWY"}
        pa = ProteinAnalysis(s)
        return {"E+K %": 100 * (comp["E"] + comp["K"]), "D+R %": 100 * (comp["D"] + comp["R"]),
                "Aromaticity": pa.aromaticity(), "Pro/Gly ratio": comp["P"] / comp["G"] if comp["G"] else np.nan,
                "GRAVY": pa.gravy(),
                "Aliphatic index": 100 * (comp["A"] + 2.9 * comp["V"] + 3.9 * (comp["I"] + comp["L"]))}
    fd = []
    for i in thermo + meso:
        base = mnames[i].split(".")[0]
        row = df[df["Accession"].astype(str).str.split(".").str[0] == base]
        if len(row):
            f = feats(clean(row.iloc[0]["Sequence"])); f["grp"] = "thermo" if i in thermo else "meso"; fd.append(f)
    FD = pd.DataFrame(fd)
    stat_rows = []
    for m in ["E+K %", "D+R %", "Aromaticity", "Pro/Gly ratio", "GRAVY", "Aliphatic index"]:
        a = FD[FD["grp"] == "thermo"][m].dropna(); b = FD[FD["grp"] == "meso"][m].dropna()
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        stat_rows.append({"metric": m, "thermo_mean": a.mean(), "meso_mean": b.mean(), "p": p})
    S = pd.DataFrame(stat_rows); S["fdr"] = bh_fdr(S["p"].values)
    S.to_csv("ai_training/analysis/group_stats_fdr.csv", index=False)
    print("\nGroup stats (BH-FDR):")
    print(S.round(3).to_string(index=False))

    # E/K vs D/R enrichment among hotspots?
    ek_hot = sig[(sig["category"].isin(["E", "K"]))]["col"].nunique()
    dr_hot = sig[(sig["category"].isin(["D", "R"]))]["col"].nunique()
    print(f"\nHotspot residue identity: E/K-enriched positions={ek_hot}, D/R-enriched positions={dr_hot}")
    # surface enrichment of charge hotspots (proper freesasa)
    hs = supp.dropna(subset=["RSA"]); nonhot_nums = [c2n_th[j] for j in mcols if j in c2n_th and j not in set(hot["col"])]
    nonhot_rsa = np.array([rsa_th[n] for n in nonhot_nums if n in rsa_th])
    _, surf_p = mannwhitneyu(hs["RSA"], nonhot_rsa, alternative="greater")
    print(f"Surface enrichment (freesasa): hotspot RSA={hs['RSA'].mean():.2f} vs others {nonhot_rsa.mean():.2f} "
          f"(p={surf_p:.2e}); {100*(hs['RSA']>0.25).mean():.0f}% hotspots surface")

    # ---- (D) engineering mutations on mesophile 3PZT ----
    coord_me = {n: c for n, c in zip(num_me, co_me)}
    cat_coords_me = np.array([coord_me[143], coord_me[186]])  # geometrically verified catalytic Glu pair (3PZT)
    mut_rows = []
    for _, r in hot.iterrows():
        j = int(r["col"])
        if j not in c2n_me:
            continue
        rn = c2n_me[j]
        wt = seq_me[num_me.index(rn)] if rn in num_me else None
        if wt is None or wt in CHARGED:
            continue  # meso đã tích điện thì bỏ
        pref = top_res(j, thermo)
        if pref not in EK:
            continue  # chỉ đề xuất tới E/K
        rsa = rsa_me.get(rn, np.nan)
        if np.isnan(rsa) or rsa <= 0.25:
            continue  # phải surface
        dist = min(np.sqrt(((coord_me[rn] - cat_coords_me) ** 2).sum(1))) if (rn in coord_me and len(cat_coords_me)) else np.nan
        if not np.isnan(dist) and dist < 12:
            continue  # phải distal (không đụng active site)
        mut_rows.append({"mutation": f"{wt}{rn}{pref}", "PF00150_col": j, "wt_meso": wt,
                         "to": pref, "thermo_charged_%": round(100 * r["thermo_charged"], 1),
                         "meso_charged_%": round(100 * r["meso_charged"], 1),
                         "delta_%": round(100 * r["delta"], 1), "RSA_3PZT": round(rsa, 2),
                         "dist_catGlu_A": round(float(dist), 1) if not np.isnan(dist) else np.nan})
    M = pd.DataFrame(mut_rows).sort_values("delta_%", ascending=False).head(5)
    M.to_csv("ai_training/analysis/engineering_mutations.csv", index=False)
    print("\n=== Đề xuất 5 đột biến (trên GH5 mesophile Bacillus 3PZT) ===")
    print(M.to_string(index=False))

    # ---- Figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) category enrichment counts
    cc = sig["category"].value_counts().reindex(list(CATS.keys())).fillna(0)
    ax[0].bar(cc.index, cc.values, color=["#c0392b" if c in ("E", "K") else "#888" for c in cc.index])
    ax[0].set_ylabel("# thermophile-enriched positions\n(FDR<0.05)"); ax[0].set_title("(a) Enrichment by residue type")
    # (b) surface vs core / proximal vs distal of charge hotspots
    tab = supp.groupby(["location", "active_site"]).size().unstack(fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=ax[1], color=["#cc5544", "#6699cc"])
    ax[1].set_title("(b) Location of charge hotspots"); ax[1].set_ylabel("count"); ax[1].tick_params(axis="x", rotation=0)
    # (c) RSA hotspot vs non-hotspot
    ax[2].boxplot([nonhot_rsa, hs["RSA"]], tick_labels=["other", "hotspot"], patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor="#6699cc", alpha=0.7))
    ax[2].axhline(0.25, color="r", ls="--", lw=1, label="surface cutoff")
    ax[2].set_ylabel("Relative solvent accessibility"); ax[2].set_title(f"(c) Hotspots surface-enriched\n(p={surf_p:.1e})"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig15_engineering.png", dpi=200)
    print("→ Fig15_engineering.png")


if __name__ == "__main__":
    main()
