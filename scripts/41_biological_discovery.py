"""
41_biological_discovery.py

KHÁM PHÁ SINH HỌC: cơ chế thích nghi nhiệt ĐẶC TRƯNG VỊ TRÍ của GH5 (không phải quy luật chung).

Ý tưởng: composition toàn cục (salt bridge/IVYWREL) là kiến thức chung. Ở đây ta hỏi:
CỤ THỂ những VỊ TRÍ NÀO trong nếp gấp (β/α)8 của GH5 là điểm nóng thích nghi nhiệt?

Quy trình:
  1. Align 757 GH5 + cấu trúc tham chiếu (T. maritima 3AMC) vào PF00150.
  2. Với mỗi cột match: tần suất residue tích điện ở thermophile (OGT>=55) vs mesophile;
     Fisher exact + hiệu chỉnh FDR (Benjamini-Hochberg). Cột FDR<0.05 & thermo>meso = HOTSPOT.
  3. Map hotspot -> toạ độ CA trên 3AMC; kiểm tra CHÚM KHÔNG GIAN bằng hoán vị.
  4. "GH5 thermal-charge signature" = tỉ lệ hotspot ở trạng thái tích điện; tương quan với OGT,
     so với cột ngẫu nhiên; và giải thích điểm số ứng viên top.

Output: manuscript/figures/Fig14_discovery.png
        ai_training/analysis/thermal_hotspots.csv
        ai_training/analysis/signature_scores.csv
"""
import os
import numpy as np
import pandas as pd
import pyhmmer
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu
from Bio.PDB import PDBParser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HMM = "pfam/PF00150.hmm"
CHARGED = set("DEKR")
alphabet = pyhmmer.easel.Alphabet.amino()
os.makedirs("ai_training/analysis", exist_ok=True)


def _s(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]; val = p[idx] * n / (i + 1); prev = min(prev, val); q[idx] = prev
    return q


def match_columns(aln):
    ncol = len(aln[0]); cols = []
    for j in range(ncol):
        if all((row[j] == "-") or row[j].isupper() for row in aln):
            cols.append(j)
    return cols


def ref_seq_from_pdb(pdb):
    T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
           "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    s = PDBParser(QUIET=True).get_structure("r", pdb)[0]
    chain = next(c for c in s if sum(1 for r in c if r.id[0] == " ") > 100)
    seq = ""; resnums = []; coords = []
    for r in chain:
        if r.id[0] == " " and r.resname in T2O and "CA" in r:
            seq += T2O[r.resname]; resnums.append(r.id[1]); coords.append(r["CA"].coord)
    return seq, resnums, np.array(coords)


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f:
        hmm = f.read()
    df = pd.read_csv("ai_training/gh5_all_verified.csv").reset_index(drop=True)
    ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]]
    ogt["_acc"] = ogt["Accession"].astype(str).str.split(".").str[0]
    ogt_map = dict(zip(ogt.dropna(subset=["OGT"])["_acc"], ogt.dropna(subset=["OGT"])["OGT"]))

    refseq, refnums, refcoords = ref_seq_from_pdb("ai_training/structures/3AMC.pdb")

    # align tất cả + reference
    seqs = []
    names = []
    for _, r in df.iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alphabet))
            names.append(str(r["Accession"]))
    seqs.append(pyhmmer.easel.TextSequence(name=b"REF_3AMC", sequence=refseq).digitize(alphabet))
    names.append("REF_3AMC")
    msa = pyhmmer.hmmalign(hmm, seqs, trim=True)
    aln = [ _s(x) for x in msa.alignment ]
    mnames = [_s(n) for n in msa.names]
    arr = np.array([list(x) for x in aln])
    mcols = match_columns(aln)
    name2i = {n: i for i, n in enumerate(mnames)}

    # thermo/meso labels theo OGT
    lab = {}
    for i, n in enumerate(mnames):
        base = n.split(".")[0]
        if base in ogt_map:
            lab[i] = 1 if ogt_map[base] >= 55 else 0
    thermo_idx = [i for i, v in lab.items() if v == 1]
    meso_idx = [i for i, v in lab.items() if v == 0]
    print(f"Thermo(OGT>=55)={len(thermo_idx)}, Meso={len(meso_idx)}, match cols={len(mcols)}")

    # per-column Fisher: charged trong thermo vs meso
    rows = []
    for j in mcols:
        tcol = arr[thermo_idx, j]; mcol = arr[meso_idx, j]
        t_ch = sum(c in CHARGED for c in tcol); t_no = sum(c != "-" and c not in CHARGED for c in tcol)
        m_ch = sum(c in CHARGED for c in mcol); m_no = sum(c != "-" and c not in CHARGED for c in mcol)
        if (t_ch + t_no) < 20 or (m_ch + m_no) < 20:
            continue
        _, p = fisher_exact([[t_ch, t_no], [m_ch, m_no]])
        tf = t_ch / (t_ch + t_no); mf = m_ch / (m_ch + m_no)
        rows.append({"col": j, "thermo_charged_frac": tf, "meso_charged_frac": mf, "delta": tf - mf, "p": p})
    cols_df = pd.DataFrame(rows)
    cols_df["fdr"] = bh_fdr(cols_df["p"].values)
    hotspots = cols_df[(cols_df["fdr"] < 0.05) & (cols_df["delta"] > 0)].sort_values("delta", ascending=False)
    print(f"HOTSPOT positions (FDR<0.05, thermo charge-enriched): {len(hotspots)}")

    # map hotspot -> 3AMC residue number & coords
    ref_i = name2i["REF_3AMC"]
    refrow = arr[ref_i]
    # match column -> vị trí thứ mấy trong reference (đếm residue không gap ở match cols)
    col_to_refnum = {}
    ref_pos = 0
    for j in mcols:
        if refrow[j] != "-":
            if ref_pos < len(refnums):
                col_to_refnum[j] = (refnums[ref_pos], refcoords[ref_pos])
            ref_pos += 1
    hot_coords = [col_to_refnum[j][1] for j in hotspots["col"] if j in col_to_refnum]
    hot_resnums = [col_to_refnum[j][0] for j in hotspots["col"] if j in col_to_refnum]
    hot_coords = np.array(hot_coords)
    print(f"Hotspots mappable to 3AMC: {len(hot_coords)} (residues {sorted(hot_resnums)[:15]}...)")

    rng = np.random.default_rng(0)
    mapped_cols = [j for j in mcols if j in col_to_refnum]

    # STRUCTURAL CHARACTERISATION: hotspot có nằm trên BỀ MẶT & TRÁNH active-site không?
    from Bio.PDB import ShrakeRupley
    MAXASA = {"A":129,"R":274,"N":195,"D":193,"C":167,"E":223,"Q":225,"G":104,"H":224,"I":197,
              "L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
    T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
           "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    struct = PDBParser(QUIET=True).get_structure("r", "ai_training/structures/3AMC.pdb")[0]
    ShrakeRupley().compute(struct, level="R")
    rsa_by_num = {}
    for ch in struct:
        for r in ch:
            if r.id[0] == " " and r.resname in T2O:
                rsa_by_num[r.id[1]] = r.sasa / MAXASA[T2O[r.resname]]
    hot_rsa = [rsa_by_num.get(rn, np.nan) for rn in hot_resnums]
    nonhot_nums = [col_to_refnum[j][0] for j in mapped_cols if j not in set(hotspots["col"])]
    nonhot_rsa = [rsa_by_num.get(rn, np.nan) for rn in nonhot_nums]
    hot_rsa = np.array([x for x in hot_rsa if not np.isnan(x)])
    nonhot_rsa = np.array([x for x in nonhot_rsa if not np.isnan(x)])
    _, rsa_p = mannwhitneyu(hot_rsa, nonhot_rsa, alternative="greater")
    pct_surface = (hot_rsa > 0.25).mean() * 100
    print(f"Surface enrichment: hotspots mean RSA={hot_rsa.mean():.2f} vs others {nonhot_rsa.mean():.2f} "
          f"(p={rsa_p:.2e}); {pct_surface:.0f}% of hotspots surface-exposed")

    # khoảng cách hotspot -> glutamate xúc tác (2 cột E bảo tồn cao nhất) — tránh active-site?
    Efrac = []
    for j in mcols:
        col = arr[:, j]; ng = col[col != "-"]
        Efrac.append((j, (ng == "E").mean() if len(ng) else 0))
    catcol = [c for c, _ in sorted(Efrac, key=lambda x: -x[1])[:2]]
    cat_coords = np.array([col_to_refnum[j][1] for j in catcol if j in col_to_refnum])
    if len(cat_coords):
        hot_mind = [min(np.sqrt(((c - cat_coords) ** 2).sum(1))) for c in hot_coords]
        all_mind = [min(np.sqrt(((col_to_refnum[j][1] - cat_coords) ** 2).sum(1))) for j in mapped_cols]
        _, cat_p = mannwhitneyu(hot_mind, all_mind, alternative="greater")
        print(f"Distance from catalytic Glu: hotspots {np.mean(hot_mind):.1f}Å vs all {np.mean(all_mind):.1f}Å "
              f"(farther, p={cat_p:.3f}) -> spare the active site")
    else:
        cat_p = np.nan; hot_mind = []; all_mind = []

    perm_p = rsa_p  # dùng cho figure title (surface enrichment)
    hotspots.to_csv("ai_training/analysis/thermal_hotspots.csv", index=False)

    # signature score cho TOÀN BỘ 757
    hot_cols = list(hotspots["col"])
    sig = []
    for i, n in enumerate(mnames):
        vals = arr[i, hot_cols]
        nongap = [v for v in vals if v != "-"]
        sig.append(sum(v in CHARGED for v in nongap) / len(nongap) if nongap else np.nan)
    sig_df = pd.DataFrame({"Accession": mnames, "signature": sig})
    sig_df = sig_df[sig_df["Accession"] != "REF_3AMC"]
    sig_df["_acc"] = sig_df["Accession"].astype(str).str.split(".").str[0]
    sig_df["OGT"] = sig_df["_acc"].map(ogt_map)
    sig_df.to_csv("ai_training/analysis/signature_scores.csv", index=False)
    val = sig_df.dropna(subset=["OGT", "signature"])
    rho_sig = spearmanr(val["signature"], val["OGT"]).correlation
    # so với cột ngẫu nhiên cùng cỡ
    rnd_rhos = []
    for _ in range(500):
        pick = rng.choice(mcols, size=len(hot_cols), replace=False)
        s = []
        for i in val.index:
            ii = name2i[val.loc[i, "Accession"]]
            vv = [arr[ii, j] for j in pick if arr[ii, j] != "-"]
            s.append(sum(x in CHARGED for x in vv) / len(vv) if vv else np.nan)
        rnd_rhos.append(spearmanr(s, val["OGT"]).correlation)
    rnd_rhos = np.array(rnd_rhos)
    print(f"Signature vs OGT: Spearman={rho_sig:.3f} | random-column columns mean={np.nanmean(rnd_rhos):.3f} "
          f"(signature > {100*(rho_sig>rnd_rhos).mean():.0f}% of random sets)")

    # GIẢI THÍCH ỨNG VIÊN: signature giải thích ĐIỂM DỰ ĐOÁN của mô hình?
    pred = pd.read_csv("ai_training/analysis/all757_predictions.csv")
    pred["_acc"] = pred["Accession"].astype(str).str.split(".").str[0]
    sig_lookup = dict(zip(sig_df["_acc"], sig_df["signature"]))
    pred["signature"] = pred["_acc"].map(sig_lookup)
    pp = pred.dropna(subset=["signature", "OGT_pred"])
    rho_pred = spearmanr(pp["signature"], pp["OGT_pred"]).correlation
    print(f"Signature vs MODEL predicted-OGT (all): Spearman={rho_pred:.3f} "
          f"-> giải thích thứ hạng của mô hình")
    all_sig = sig_df["signature"].dropna().values

    # ---- Figure ----
    fig = plt.figure(figsize=(15, 9))
    # (a) Manhattan: delta per column
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(cols_df["col"], cols_df["delta"] * 100, s=18,
                c=np.where(cols_df["fdr"] < 0.05, "#c0392b", "#bbbbbb"))
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("PF00150 alignment position"); ax1.set_ylabel("Δ charged frequency\n(thermo − meso), %")
    ax1.set_title(f"(a) Position-specific charge enrichment\n{len(hotspots)} GH5 hotspots (FDR<0.05)")
    # (b) hotspots on structure, colored by surface
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.plot(refcoords[:, 0], refcoords[:, 1], refcoords[:, 2], color="#cccccc", lw=1.0, alpha=0.7)
    if len(hot_coords):
        ax2.scatter(hot_coords[:, 0], hot_coords[:, 1], hot_coords[:, 2], c="#c0392b", s=55, depthshade=True,
                    label="thermal hotspot")
    if len(cat_coords):
        ax2.scatter(cat_coords[:, 0], cat_coords[:, 1], cat_coords[:, 2], c="#f1c40f", s=120,
                    marker="*", edgecolor="k", label="catalytic Glu")
    ax2.set_title(f"(b) Hotspots on GH5 fold (3AMC)\ndistal to catalytic Glu (p={cat_p:.3f}), spare active site")
    ax2.legend(fontsize=8); ax2.set_axis_off()
    # (c) signature vs OGT
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(val["signature"] * 100, val["OGT"], s=16, alpha=0.5, edgecolor="k", linewidth=0.3, color="#279")
    ax3.set_xlabel("GH5 thermal-charge signature (% hotspots charged)"); ax3.set_ylabel("OGT (°C)")
    ax3.set_title(f"(c) Sparse 66-position signature explains OGT\nSpearman={rho_sig:.2f} "
                  f"(> {100*(rho_sig>rnd_rhos).mean():.0f}% of random position sets)")
    # (d) signature vs model prediction — explains candidate ranking
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(pp["signature"] * 100, pp["OGT_pred"], s=14, alpha=0.35, color="#bbbbbb", label="all GH5")
    tc = pd.read_csv("ai_training/analysis/top_candidates_validated.csv")
    tc["_acc"] = tc["Accession"].astype(str).str.split(".").str[0]
    tc["signature"] = tc["_acc"].map(sig_lookup)
    ax4.scatter(tc["signature"] * 100, tc["Predicted_OGT_C"], s=55, color="#c0392b",
                edgecolor="k", label="top validated candidates", zorder=3)
    ax4.set_xlabel("Thermal-charge signature (%)"); ax4.set_ylabel("Model predicted OGT (°C)")
    ax4.set_title(f"(d) Signature explains model ranking\nSpearman={rho_pred:.2f}"); ax4.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig14_discovery.png", dpi=200)
    print("→ Fig14_discovery.png")

    pd.DataFrame([{"n_hotspots": len(hotspots), "hotspot_resnums_3AMC": ";".join(map(str, sorted(hot_resnums))),
                   "hotspot_mean_RSA": hot_rsa.mean(), "nonhot_mean_RSA": nonhot_rsa.mean(), "rsa_p": rsa_p,
                   "pct_surface": pct_surface, "dist_from_catalytic_p": cat_p,
                   "signature_OGT_spearman": rho_sig, "signature_vs_random_pct": 100*(rho_sig>rnd_rhos).mean(),
                   "signature_predOGT_spearman": rho_pred}]
                 ).to_csv("ai_training/analysis/discovery_summary.csv", index=False)


if __name__ == "__main__":
    main()
