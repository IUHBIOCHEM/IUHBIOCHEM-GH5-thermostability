"""
45_generate_combo_mutations.py

Sinh danh sách đột biến ĐƠN mở rộng trên GH5 mesophile 3PZT (chain B) tại các hotspot bề mặt:
điều kiện = hotspot (FDR<0.05, charge-enriched), 3PZT mang residue KHÔNG tích điện,
surface (RSA>0.25, freesasa), distal (>12Å từ Glu xúc tác), thermo ưu tiên residue tích điện.

Output: foldx_run/singles_list.txt (mỗi dòng 1 đột biến)
        ai_training/analysis/combo_candidates.csv (chi tiết)
"""
import os
import numpy as np
import pandas as pd
import pyhmmer
import freesasa
from scipy.stats import fisher_exact
from Bio.PDB import PDBParser

HMM = "pfam/PF00150.hmm"
alphabet = pyhmmer.easel.Alphabet.amino()
CHARGED = set("DEKR"); TARGET = set("DEKR")  # mọi residue tích điện
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


def mcols(aln):
    return [j for j in range(len(aln[0])) if all((r[j] == "-") or r[j].isupper() for r in aln)]


def pdb_info(pdb):
    m = PDBParser(QUIET=True).get_structure("x", pdb)[0]
    ch = max(m, key=lambda c: sum(1 for r in c if r.id[0] == " "))
    seq, nums, coords = "", [], []
    for r in ch:
        if r.id[0] == " " and r.resname in T2O and "CA" in r:
            seq += T2O[r.resname]; nums.append(r.id[1]); coords.append(r["CA"].coord)
    res = freesasa.calc(freesasa.Structure(pdb)).residueAreas()
    rsa = {}
    if ch.id in res:
        for rn in res[ch.id]:
            a = res[ch.id][rn]
            if a.relativeTotal is not None:
                try:
                    rsa[int(rn)] = a.relativeTotal
                except ValueError:
                    pass
    return seq, nums, np.array(coords), rsa, ch.id


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f:
        hmm = f.read()
    df = pd.read_csv("ai_training/gh5_all_verified.csv")
    ogt = pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession", "OGT"]].dropna()
    om = dict(zip(ogt["Accession"].astype(str).str.split(".").str[0], ogt["OGT"]))
    seq, nums, coords, rsa, chain = pdb_info("ai_training/structures/3PZT.pdb")

    seqs, names = [], []
    for _, r in df.iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alphabet)); names.append(str(r["Accession"]))
    seqs.append(pyhmmer.easel.TextSequence(name=b"REF", sequence=seq).digitize(alphabet)); names.append("REF")
    msa = pyhmmer.hmmalign(hmm, seqs, trim=True)
    aln = [_s(x) for x in msa.alignment]; mn = [_s(n) for n in msa.names]
    arr = np.array([list(x) for x in aln]); mc = mcols(aln); n2i = {n: i for i, n in enumerate(mn)}
    thermo = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] >= 55]
    meso = [i for i, n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]] < 55]

    # hotspot charge (FDR)
    rec = []
    for j in mc:
        tn = arr[thermo, j]; tn = tn[tn != "-"]; mnn = arr[meso, j]; mnn = mnn[mnn != "-"]
        if len(tn) < 20 or len(mnn) < 20:
            continue
        tc = sum(c in CHARGED for c in tn); mcc = sum(c in CHARGED for c in mnn)
        _, p = fisher_exact([[tc, len(tn) - tc], [mcc, len(mnn) - mcc]])
        rec.append({"col": j, "tf": tc / len(tn), "mf": mcc / len(mnn), "delta": tc / len(tn) - mcc / len(mnn), "p": p})
    C = pd.DataFrame(rec); C["fdr"] = bh(C["p"].values)
    hot = C[(C["fdr"] < 0.05) & (C["delta"] > 0)]

    # map col -> 3PZT resnum
    refrow = arr[n2i["REF"]]; c2n = {}; pos = 0
    for j in mc:
        if refrow[j] != "-":
            if pos < len(nums):
                c2n[j] = nums[pos]
            pos += 1
    # catalytic Glu (2 cột E bảo tồn cao) -> 3PZT coords
    Ef = sorted([(j, (arr[:, j][arr[:, j] != "-"] == "E").mean()) for j in mc], key=lambda x: -x[1])[:2]
    coordmap = {n: c for n, c in zip(nums, coords)}
    cat_co = np.array([coordmap[c2n[j]] for j, _ in Ef if j in c2n and c2n[j] in coordmap])

    def pref(col):
        v = arr[thermo, col]; v = v[v != "-"]
        if len(v) == 0:
            return "-"
        u, c = np.unique(v, return_counts=True); return u[c.argmax()]

    rows = []
    for _, r in hot.iterrows():
        j = int(r["col"])
        if j not in c2n:
            continue
        rn = c2n[j]; wt = seq[nums.index(rn)] if rn in nums else None
        if wt is None or wt in CHARGED:
            continue
        to = pref(j)
        if to not in TARGET:
            continue
        rr = rsa.get(rn, np.nan)
        if np.isnan(rr) or rr <= 0.25:
            continue
        dist = min(np.sqrt(((coordmap[rn] - cat_co) ** 2).sum(1))) if (rn in coordmap and len(cat_co)) else np.nan
        if not np.isnan(dist) and dist < 12:
            continue
        rows.append({"mutation": f"{wt}{rn}{to}", "wt": wt, "pos": rn, "to": to,
                     "delta_%": round(100 * r["delta"], 1), "RSA": round(rr, 2),
                     "dist_catGlu": round(float(dist), 1)})
    cand = pd.DataFrame(rows).drop_duplicates("mutation").sort_values("delta_%", ascending=False).head(14)
    cand.to_csv("ai_training/analysis/combo_candidates.csv", index=False)
    os.makedirs("foldx_run", exist_ok=True)
    with open("foldx_run/singles_list.txt", "w") as f:
        for _, r in cand.iterrows():
            f.write(f"{r['wt']}{chain}{r['pos']}{r['to']};\n")
    print(f"Chain={chain}, ứng viên đơn: {len(cand)}")
    print(cand.to_string(index=False))


if __name__ == "__main__":
    main()
