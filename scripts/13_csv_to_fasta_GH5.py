#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import os
import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z]", "", s)
    return s

def main():
    ap = argparse.ArgumentParser(description="Convert GH5 CSV to FASTA (Accession + AA_sequence).")
    ap.add_argument("-i", "--input", default="ai_training/gh5_verified_pf00150.csv",
                    help="Input CSV path (mặc định: tập đã xác minh GH5 PF00150)")
    ap.add_argument("-o", "--output", default="GH5_bacteria.fasta", help="Output FASTA path")
    ap.add_argument("--acc_col", default="Accession", help="Accession column name")
    ap.add_argument("--seq_col", default="Sequence", help="AA sequence column name")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    rows = []
    bad = []
    for i, r in df.iterrows():
        acc = str(r.get(args.acc_col, "")).strip()
        seq = clean_seq(r.get(args.seq_col, ""))

        if not acc or not seq:
            bad.append((i, acc, "missing"))
            continue

        noncanon = [c for c in seq if c not in VALID_AA]
        if len(noncanon) / len(seq) > 0.05:
            bad.append((i, acc, "too many non-canonical AA"))
            continue

        rows.append((acc, seq))

    with open(args.output, "w") as f:
        for acc, seq in rows:
            f.write(f">{acc}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j+60] + "\n")

    print(f"DONE: wrote {len(rows)} sequences -> {args.output} ({os.path.getsize(args.output)} bytes)")
    if bad:
        print(f"Filtered out: {len(bad)} rows (showing first 10): {bad[:10]}")

if __name__ == "__main__":
    main()
