"""
28_build_topt_labels.py

Gộp Topt enzyme (BRENDA API, EC 3.2.1.4) theo LOÀI và nối vào tập 757 GH5 đã xác minh.
Bổ sung thêm cột OGT (TEMPURA) để so sánh/hỗ trợ.

Input : ai_training/brenda_topt_api.csv   (Organism, Topt, ...)
        ai_training/gh5_all_verified.csv
        ai_training/gh5_ogt_labeled.csv    (để lấy OGT)
Output: ai_training/gh5_topt_labeled.csv   (+ Topt, Topt_source, OGT)
"""
import pandas as pd

BR = "ai_training/brenda_topt_api.csv"
ALLV = "ai_training/gh5_all_verified.csv"
OGT = "ai_training/gh5_ogt_labeled.csv"
OUT = "ai_training/gh5_topt_labeled.csv"


def clean(x):
    return str(x).replace("[", "").replace("]", "").replace('"', "").strip()


def sp2(x):
    p = clean(x).split()
    return " ".join(p[:2]) if len(p) >= 2 else clean(x)


def gen1(x):
    c = clean(x)
    return c.split()[0] if c else ""


def main():
    br = pd.read_csv(BR)
    d = pd.read_csv(ALLV)

    br["_sp"] = br["Organism"].map(sp2)
    br["_gen"] = br["Organism"].map(gen1)
    d["_sp"] = d["Organism"].map(sp2)
    d["_gen"] = d["Organism"].map(gen1)

    sp_map = br.groupby("_sp")["Topt"].mean()
    gen_map = br.groupby("_gen")["Topt"].mean()

    d["_topt_sp"] = d["_sp"].map(sp_map)
    d["_topt_gen"] = d["_gen"].map(gen_map)
    d["Topt"] = d["_topt_sp"].fillna(d["_topt_gen"])
    d["Topt_source"] = "none"
    d.loc[d["_topt_gen"].notna(), "Topt_source"] = "genus"
    d.loc[d["_topt_sp"].notna(), "Topt_source"] = "species"

    # thêm OGT để đối chiếu
    ogt = pd.read_csv(OGT)[["Accession", "OGT"]]
    d = d.merge(ogt, on="Accession", how="left")

    d = d.drop(columns=["_sp", "_gen", "_topt_sp", "_topt_gen"])
    d.to_csv(OUT, index=False)

    s = d["Topt"].dropna()
    print(f"Tổng: {len(d)} | có Topt: {len(s)} "
          f"[species={ (d['Topt_source']=='species').sum() }, genus={ (d['Topt_source']=='genus').sum() }]")
    print(f"Topt: min={s.min():.0f}, max={s.max():.0f}, mean={s.mean():.1f}, std={s.std():.1f} | "
          f"thermo(>=55)={ (s>=55).sum() }, meso(<55)={ (s<55).sum() }")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
