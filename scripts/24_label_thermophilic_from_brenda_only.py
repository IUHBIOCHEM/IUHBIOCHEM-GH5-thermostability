#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print(">>> FILE 24 DEBUG STARTED <<<")

from pathlib import Path
import re
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
AI = BASE / "ai_training"

DATA = AI / "gh5_training_data_with_features.csv"
BRENDA = AI / "brenda_topt_export.csv"
OUT = AI / "gh5_training_data_labeled_final.csv"

TOPT_THRESHOLD = 60.0


def norm_acc(x):
    return re.sub(r"\.\d+$", "", str(x).strip())


def main():
    print("[STEP 1] Checking input files")

    print("  DATA exists:", DATA.exists(), DATA)
    print("  BRENDA exists:", BRENDA.exists(), BRENDA)

    if not DATA.exists() or not BRENDA.exists():
        print("❌ INPUT FILE MISSING — STOP")
        return

    print("[STEP 2] Loading CSVs")
    df = pd.read_csv(DATA)
    # BRENDA export có ký tự độ (°C) mã hóa latin-1, không phải utf-8 -> đọc kèm fallback
    try:
        br = pd.read_csv(BRENDA)
    except UnicodeDecodeError:
        br = pd.read_csv(BRENDA, encoding="latin-1")

    print("  GH5 rows:", df.shape)
    print("  BRENDA rows:", br.shape)
    print("  BRENDA columns:", br.columns.tolist())

    print("[STEP 3] Normalizing accessions")
    df["Accession_norm"] = df["Accession"].apply(norm_acc)

    # auto-detect columns
    acc_col = None
    for c in ["Accession", "accession", "UniProt", "uniprot", "Protein ID", "ProteinID"]:
        if c in br.columns:
            acc_col = c
            break

    t_col = None
    for c in ["Topt", "topt", "Temperature optimum", "Temperature_Optimum", "Temp opt"]:
        if c in br.columns:
            t_col = c
            break

    print("  Detected accession column:", acc_col)
    print("  Detected Topt column:", t_col)

    if acc_col is None or t_col is None:
        print("❌ BRENDA FILE FORMAT INVALID — STOP")
        return

    br["Accession_norm"] = br[acc_col].apply(norm_acc)
    br["_Topt"] = pd.to_numeric(br[t_col], errors="coerce")

    print("[STEP 4] Valid BRENDA entries:", br.dropna(subset=["Accession_norm", "_Topt"]).shape)

    print("[STEP 5] Merging datasets")
    m = df.merge(
        br[["Accession_norm", "_Topt"]],
        on="Accession_norm",
        how="left"
    )

    print("  Merged shape:", m.shape)
    print("  Non-null Topt:", m["_Topt"].notna().sum())

    print("[STEP 6] Assigning labels")
    m["Is_Thermophilic"] = (m["_Topt"] >= TOPT_THRESHOLD).astype(int)
    m["Thermo_Evidence"] = "none"
    m.loc[m["Is_Thermophilic"] == 1, "Thermo_Evidence"] = "brenda_topt>=60C"

    print("[STEP 7] Label counts")
    print(m["Is_Thermophilic"].value_counts())

    print("[STEP 8] Saving output")
    m.to_csv(OUT, index=False)

    print("✅ SAVED:", OUT)
    print(">>> FILE 24 DEBUG DONE <<<")


if __name__ == "__main__":
    main()
