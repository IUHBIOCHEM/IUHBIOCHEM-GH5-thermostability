"""
01d_attach_ogt_labels.py

Gán nhãn hồi quy = nhiệt độ sinh trưởng tối ưu (OGT) của loài nguồn, lấy từ TEMPURA.
Nối theo LOÀI (2 token đầu), fallback theo CHI. Xuất cột OGT làm target hồi quy.

Input : ai_training/gh5_all_verified.csv  (cột Accession, Organism, Sequence, ...)
TEMPURA: data_raw/tempura.csv  (genus_and_species, genus, Topt_ave)
Output: ai_training/gh5_ogt_labeled.csv  (+ cột OGT, OGT_source)
"""
import argparse
import re
import pandas as pd

TEMPURA = "data_raw/tempura.csv"


def clean_org(x):
    return str(x).replace("[", "").replace("]", "").replace('"', "").strip()


def species2(x):
    p = clean_org(x).split()
    return " ".join(p[:2]) if len(p) >= 2 else clean_org(x)


def genus1(x):
    c = clean_org(x)
    return c.split()[0] if c else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="ai_training/gh5_all_verified.csv")
    ap.add_argument("--out_csv", default="ai_training/gh5_ogt_labeled.csv")
    ap.add_argument("--tempura", default=TEMPURA)
    args = ap.parse_args()

    d = pd.read_csv(args.in_csv)
    t = pd.read_csv(args.tempura)

    d["_species"] = d["Organism"].map(species2)
    d["_genus"] = d["Organism"].map(genus1)
    t["_species"] = t["genus_and_species"].map(species2)
    t["_genus"] = t["genus"].astype(str)

    sp_map = t.dropna(subset=["Topt_ave"]).groupby("_species")["Topt_ave"].mean()
    gen_map = t.dropna(subset=["Topt_ave"]).groupby("_genus")["Topt_ave"].mean()

    d["_ogt_species"] = d["_species"].map(sp_map)
    d["_ogt_genus"] = d["_genus"].map(gen_map)
    d["OGT"] = d["_ogt_species"].fillna(d["_ogt_genus"])
    d["OGT_source"] = "none"
    d.loc[d["_ogt_genus"].notna(), "OGT_source"] = "genus"
    d.loc[d["_ogt_species"].notna(), "OGT_source"] = "species"

    labeled = d.drop(columns=["_species", "_genus", "_ogt_species", "_ogt_genus"])
    labeled.to_csv(args.out_csv, index=False)

    n = len(d)
    cov = d["OGT"].notna().sum()
    print(f"Tổng: {n} | có OGT: {cov} ({100*cov/n:.1f}%) "
          f"[species={ (d['OGT_source']=='species').sum() }, genus={ (d['OGT_source']=='genus').sum() }]")
    if cov:
        s = d["OGT"].dropna()
        print(f"OGT: min={s.min():.1f}, max={s.max():.1f}, mean={s.mean():.1f}, std={s.std():.1f}")
        print(f"  thermophile (OGT>=55): {(s>=55).sum()} | mesophile (<55): {(s<55).sum()}")
    print(f"→ {args.out_csv}")


if __name__ == "__main__":
    main()
