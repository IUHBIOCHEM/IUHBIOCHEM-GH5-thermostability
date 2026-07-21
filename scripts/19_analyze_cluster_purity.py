#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "ai_training"

DATA = BASE / "gh5_training_data_with_features.csv"
GROUPS = BASE / "groups_map_cdhit70.csv"
OUT = BASE / "cluster_purity_cdhit70.csv"

LABEL = "Is_Thermophilic_Guess"

def normalize(x):
    return str(x).split(".")[0]

def main():
    df = pd.read_csv(DATA)
    g = pd.read_csv(GROUPS)

    df["_acc"] = df["Accession"].map(normalize)
    g["_acc"] = g["Accession"].map(normalize)

    m = df.merge(g, on="_acc", how="inner")

    rows = []
    for cid, sub in m.groupby("ClusterID"):
        counts = sub[LABEL].value_counts().to_dict()
        total = len(sub)
        maj_frac = max(counts.values()) / total
        rows.append({
            "ClusterID": cid,
            "size": total,
            "class_counts": counts,
            "majority_fraction": maj_frac,
            "is_mixed": len(counts) > 1
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("Clusters:", len(out))
    print("Mixed clusters:", out["is_mixed"].sum())
    print("Pure clusters:", (out["is_mixed"] == False).sum())
    print("Mean majority fraction:", out["majority_fraction"].mean())

if __name__ == "__main__":
    main()

