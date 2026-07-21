#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15_parse_cdhit_clstr_to_groups.py (FIX)
Parse CD-HIT .clstr to Accession -> ClusterID map (robust for truncated headers like 'WP_....1...').
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def extract_accession(line: str) -> str:
    """
    CD-HIT .clstr line examples:
      0   458aa, >WP_225736107.1... *
      1   455aa, >sp|P07982|CELB_BACSU... at 98%
      2   460aa, >tr|Q9XXXX|SOMENAME... at 92%

    We:
      - capture token after '>'
      - cut off trailing '...' if present
      - if pipe format: take 2nd field (ACC)
      - drop isoform suffix -1 and version suffix .1
    """
    m = re.search(r'>\s*([^\s,]+)', line)
    if not m:
        return ""
    token = m.group(1)

    # remove trailing "..." from truncated headers
    token = re.sub(r'\.\.\.$', '', token)

    # if token still contains "...", cut at first occurrence
    token = token.split("...")[0]

    # UniProt style sp|ACC|NAME
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2:
            token = parts[1].strip()

    # remove isoform suffix -1, -2...
    token = re.sub(r"-\d+$", "", token)
    # remove version suffix .1 .2 ...
    token = re.sub(r"\.\d+$", "", token)

    return token.strip()


def main():
    ap = argparse.ArgumentParser(description="Parse CD-HIT .clstr to Accession -> ClusterID map.")
    ap.add_argument("-i", "--input", required=True, help="Input .clstr file (e.g., GH5_cdhit70.fasta.clstr)")
    ap.add_argument("-o", "--output", default="groups_map_cdhit70.csv", help="Output CSV")
    args = ap.parse_args()

    clstr_path = Path(args.input)
    if not clstr_path.exists():
        raise FileNotFoundError(f"Not found: {clstr_path}")

    cluster_id = None
    records = []

    with open(clstr_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">Cluster"):
                # >Cluster 0
                cluster_id = int(line.split()[1])
                continue

            if cluster_id is None:
                continue

            acc = extract_accession(line)
            if acc:
                records.append({"Accession": acc, "ClusterID": cluster_id})

    if not records:
        raise RuntimeError("No records parsed. Check the .clstr format.")

    df = pd.DataFrame(records).drop_duplicates()
    df.to_csv(args.output, index=False)

    print(f"DONE: wrote {len(df)} rows | clusters={df['ClusterID'].nunique()}")
    print(f"Saved -> {args.output}")
    print("Example rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
