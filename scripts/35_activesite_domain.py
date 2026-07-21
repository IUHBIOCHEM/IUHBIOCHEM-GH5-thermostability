"""
35_activesite_domain.py

Chứng minh tính toàn vẹn enzyme của các trình tự GH5:
  (1) Domain completeness: độ phủ vùng PF00150 (hmmsearch envelope / chiều dài HMM).
  (2) Catalytic residues: xác định 2 glutamate xúc tác (acid/base + nucleophile) của clan GH-A
      qua cột match bảo tồn cao trong alignment PF00150, kiểm tra bảo tồn ở các ứng viên top.
  (3) Bảo tồn các residue active-site clan GH-A khác (R, H, N, Y, W).

Output: manuscript/figures/Fig9_catalytic_conservation.png
        ai_training/analysis/domain_completeness.csv
        ai_training/analysis/catalytic_conservation.csv
"""
import os
import numpy as np
import pandas as pd
import pyhmmer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HMM = "pfam/PF00150.hmm"
os.makedirs("ai_training/analysis", exist_ok=True)
alphabet = pyhmmer.easel.Alphabet.amino()


def _s(v):
    return v.decode() if isinstance(v,(bytes,bytearray)) else str(v)


def load_hmm():
    with pyhmmer.plan7.HMMFile(HMM) as f:
        return f.read()


def digital_seqs(df):
    seqs = []
    for _, r in df.iterrows():
        s = "".join(c for c in str(r["Sequence"]).upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        if len(s) >= 40:
            seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(), sequence=s).digitize(alphabet))
    return seqs


def domain_completeness(df, hmm):
    hmm_len = hmm.M
    seqs = digital_seqs(df)
    bg = pyhmmer.plan7.Background(alphabet)
    pipe = pyhmmer.plan7.Pipeline(alphabet, background=bg, bit_cutoffs="gathering")
    hits = pipe.search_hmm(hmm, pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs))
    rows = []
    for hit in hits:
        best = max(hit.domains, key=lambda d: d.score)
        aln = best.alignment
        cov = (aln.hmm_to - aln.hmm_from + 1) / hmm_len
        rows.append({"Accession": _s(hit.name), "hmm_from": aln.hmm_from,
                     "hmm_to": aln.hmm_to, "coverage": cov, "score": hit.score})
    return pd.DataFrame(rows), hmm_len


def match_columns(msa_alignment):
    """Trả về danh sách chỉ số cột match (chỉ chứa [A-Z-], không có insert lowercase/'.')."""
    ncol = len(msa_alignment[0])
    cols = []
    for j in range(ncol):
        column = [row[j] for row in msa_alignment]
        if all((c == "-") or c.isupper() for c in column):
            cols.append(j)
    return cols


def main():
    hmm = load_hmm()
    allv = pd.read_csv("ai_training/gh5_all_verified.csv")
    topc = pd.read_csv("ai_training/analysis/top_candidates.csv")

    # (1) Domain completeness cho toàn tập + ứng viên
    dc, hmm_len = domain_completeness(allv, hmm)
    dc.to_csv("ai_training/analysis/domain_completeness.csv", index=False)
    top_acc = set(topc["Accession"].astype(str).str.split(".").str[0])
    dc["_acc"] = dc["Accession"].astype(str).str.split(".").str[0]
    dc_top = dc[dc["_acc"].isin(top_acc)]
    print(f"HMM length (PF00150) = {hmm_len}")
    print(f"Toàn tập: coverage mean={dc['coverage'].mean():.2f}, "
          f"≥90%: {(dc['coverage']>=0.9).mean()*100:.0f}%")
    print(f"Ứng viên top: coverage mean={dc_top['coverage'].mean():.2f}, "
          f"≥90%: {(dc_top['coverage']>=0.9).mean()*100:.0f}%")

    # (2) Align toàn tập vào HMM để tìm cột glutamate xúc tác
    seqs = digital_seqs(allv)
    msa = pyhmmer.hmmalign(hmm, seqs, trim=True)
    aln = [s for s in msa.alignment]
    names = [_s(n) for n in msa.names]
    mcols = match_columns(aln)
    arr = np.array([list(s) for s in aln])

    # bảo tồn E theo cột match
    consE = []
    for j in mcols:
        col = arr[:, j]
        nongap = col[col != "-"]
        fE = (nongap == "E").mean() if len(nongap) else 0
        consE.append((j, fE, len(nongap) / len(col)))
    consE.sort(key=lambda x: x[1], reverse=True)
    # 2 cột glutamate bảo tồn cao nhất = acid/base + nucleophile
    cat_cols = sorted([c[0] for c in consE[:2]])
    print("\nHai cột glutamate xúc tác bảo tồn cao nhất:")
    for j in cat_cols:
        col = arr[:, j]; ng = col[col != "-"]
        print(f"  col {j}: E={ (ng=='E').mean()*100:.1f}%  (n={len(ng)})")

    # (3) residue active-site clan GH-A khác: các cột bảo tồn cao (>80%) bất kỳ
    conserved_cols = []
    for j in mcols:
        col = arr[:, j]; ng = col[col != "-"]
        if len(ng) == 0:
            continue
        vals, cnts = np.unique(ng, return_counts=True)
        top = vals[cnts.argmax()]; frac = cnts.max() / len(ng)
        if frac >= 0.8:
            conserved_cols.append((j, top, frac))
    print(f"\nSố cột match bảo tồn ≥80%: {len(conserved_cols)} "
          f"(residue: {''.join(sorted(set(t for _,t,_ in conserved_cols)))})")

    # ---- bảng active-site cho TOÀN TẬP (để lọc ứng viên) ----
    cons_col_idx = [j for j, _, _ in conserved_cols]
    cons_res = {j: t for j, t, _ in conserved_cols}
    covmap = dict(zip(dc["Accession"].astype(str).str.split(".").str[0], dc["coverage"]))
    full_rows = []
    for i, n in enumerate(names):
        base = n.split(".")[0]
        e1 = arr[i, cat_cols[0]] == "E"
        e2 = arr[i, cat_cols[1]] == "E"
        n_as = sum(arr[i, j] == cons_res[j] for j in cons_col_idx)
        full_rows.append({"Accession": n, "_acc": base,
                          "coverage": covmap.get(base, np.nan),
                          "both_catalytic_Glu": bool(e1 and e2),
                          "active_site_cols_matched": n_as,
                          "active_site_total": len(cons_col_idx)})
    full = pd.DataFrame(full_rows)
    full.to_csv("ai_training/analysis/active_site_integrity_all.csv", index=False)
    print(f"\nActive-site clan GH-A ({len(cons_col_idx)} cột): toàn tập trung bình "
          f"{full['active_site_cols_matched'].mean():.1f}/{len(cons_col_idx)} khớp")

    # kiểm tra bảo tồn 2 glutamate xúc tác ở ỨNG VIÊN top
    name2row = {n: i for i, n in enumerate(names)}
    rows = []
    for acc in topc["Accession"].astype(str):
        base = acc.split(".")[0]
        idx = next((name2row[n] for n in names if n.split(".")[0] == base), None)
        if idx is None:
            rows.append({"Accession": acc, "cat_Glu1": "NA", "cat_Glu2": "NA", "both_E": False})
            continue
        r1, r2 = arr[idx, cat_cols[0]], arr[idx, cat_cols[1]]
        rows.append({"Accession": acc, "cat_Glu1": r1, "cat_Glu2": r2, "both_E": (r1 == "E" and r2 == "E")})
    cat = pd.DataFrame(rows)
    cat.to_csv("ai_training/analysis/catalytic_conservation.csv", index=False)
    print(f"\nỨng viên top giữ đủ 2 glutamate xúc tác: {cat['both_E'].mean()*100:.0f}%")

    # dataset-wide: fraction giữ cả 2 Glu
    both = ((arr[:, cat_cols[0]] == "E") & (arr[:, cat_cols[1]] == "E")).mean()
    print(f"Toàn tập giữ đủ 2 glutamate xúc tác: {both*100:.0f}%")

    # ---- Figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(dc["coverage"] * 100, bins=25, color="#279", alpha=0.85)
    ax[0].axvline(90, color="r", ls="--", label="90% completeness")
    ax[0].set_xlabel("PF00150 domain coverage (%)"); ax[0].set_ylabel("Sequences")
    ax[0].set_title(f"(a) Domain completeness (n={len(dc)})"); ax[0].legend()

    # conservation of catalytic + top conserved columns
    labels = ["Cat. Glu1\n(acid/base)", "Cat. Glu2\n(nucleophile)"]
    vals = []
    for j in cat_cols:
        col = arr[:, j]; ng = col[col != "-"]; vals.append((ng == "E").mean() * 100)
    ax[1].bar(labels, vals, color=["#c0392b", "#b03030"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 1, f"{v:.0f}%", ha="center")
    ax[1].set_ylim(0, 105); ax[1].set_ylabel("% sequences with Glu (E)")
    ax[1].set_title("(b) Catalytic glutamate conservation")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig9_catalytic_conservation.png", dpi=200)
    print("→ Fig9_catalytic_conservation.png")


if __name__ == "__main__":
    main()
