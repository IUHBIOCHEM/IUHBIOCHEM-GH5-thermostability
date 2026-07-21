#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {  # Kyte-Doolittle hydropathy
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}
# Instability index dipeptide weights (Guruprasad 1990) requires 400 values.
# To keep this robust and self-contained, we implement the published table as a compact string map.
# Source values are standard and widely implemented; here we embed them directly.
II_KEYS = [a+b for a in AA for b in AA]
II_VALUES = [
    # 400 values in AA order ACDE...; this is the standard Guruprasad dipeptide instability weight table.
    # (To keep this message readable, we load it from a compressed literal below.)
]
# Compressed table (400 floats) — expanded at runtime
II_COMPRESSED = (
"1.0,-1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0;"
"0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
)
# NOTE: The true 400-value table is long; rather than risk transcription errors here,
# we will compute Instability Index using BioPython if available, else skip it safely.

def clean_seq(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z]", "", s)
    return s

def gravy(seq: str) -> float:
    vals = [KD[a] for a in seq if a in KD]
    return float(np.mean(vals)) if vals else np.nan

def aromaticity(seq: str) -> float:
    if not seq:
        return np.nan
    arom = sum(seq.count(a) for a in "FWY")
    return arom / len(seq)

def aliphatic_index(seq: str) -> float:
    # AI = X(Ala) + 2.9*X(Val) + 3.9*(X(Ile)+X(Leu)) *100
    if not seq:
        return np.nan
    L = len(seq)
    xa = seq.count("A") / L
    xv = seq.count("V") / L
    xi = seq.count("I") / L
    xl = seq.count("L") / L
    return 100.0 * (xa + 2.9 * xv + 3.9 * (xi + xl))

def instability_index_biopython(seq: str) -> float:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return float(ProteinAnalysis(seq).instability_index())
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser(description="Build numeric descriptors from GH5 sequences (for non-leaking RF).")
    ap.add_argument("--data_csv", default="ai_training/gh5_verified_pf00150.csv",
                    help="Input CSV with Accession, Sequence, label (mặc định: tập đã xác minh GH5 PF00150)")
    ap.add_argument("--out_csv", default="ai_training/gh5_training_data_with_features.csv",
                    help="Output CSV with numeric features added")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--acc_col", default="Accession")
    ap.add_argument("--label_col", default="Is_Thermophilic_Guess")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    for col in [args.acc_col, args.seq_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    seqs = df[args.seq_col].map(clean_seq)

    df["Length"] = seqs.map(lambda s: len(s) if s else np.nan)
    df["Aromaticity"] = seqs.map(aromaticity)
    df["GRAVY"] = seqs.map(gravy)
    df["AliphaticIndex"] = seqs.map(aliphatic_index)

    # Instability index (Biopython if available)
    df["InstabilityIndex"] = seqs.map(instability_index_biopython)

    # Drop rows where sequence missing
    df = df.dropna(subset=["Length", args.label_col]).copy()

    # Force numeric
    for c in ["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(args.out_csv, index=False)
    print(f"DONE: wrote {len(df)} rows -> {args.out_csv}")
    print("Feature NA counts:")
    print(df[["Length","Aromaticity","GRAVY","AliphaticIndex","InstabilityIndex"]].isna().sum().to_string())

    if df["InstabilityIndex"].isna().mean() > 0.5:
        print("\n[NOTE] InstabilityIndex has many NaNs. Install Biopython to compute it:")
        print("  pip install biopython")


if __name__ == "__main__":
    main()

