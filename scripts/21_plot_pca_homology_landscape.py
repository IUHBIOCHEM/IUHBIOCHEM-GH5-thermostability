#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
21_plot_pca_homology_landscape.py

Figure 1 (Q1-style):
- PCA landscape from classical physicochemical features
- CD-HIT 70% homology-aware dataset (groups file used for merge sanity, not required for PCA)
- Plot mesophilic first (gray circles), thermophilic on top (red-edged triangles)
- Legend (no colorbar)

Inputs (expected in ai_training/):
- gh5_training_data_with_features.csv
- groups_map_cdhit70.csv   (optional but recommended)

Output:
- ai_training/figures/main/Figure1_PCA_landscape.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def norm_acc(series: pd.Series) -> pd.Series:
    """Remove version suffix like '.1' from accessions to match CD-HIT outputs."""
    return series.astype(str).str.replace(r"\.\d+$", "", regex=True)


def main():
    # ================= PATHS =================
    BASE = Path(__file__).resolve().parent.parent              # .../Cellulase
    AI = BASE / "ai_training"
    FIG = AI / "figures" / "main"
    FIG.mkdir(parents=True, exist_ok=True)

    DATA_FILE = AI / "gh5_training_data_with_features.csv"
    GROUP_FILE = AI / "groups_map_cdhit70.csv"
    OUT_PNG = FIG / "Figure1_PCA_landscape.png"

    print("[INFO] Base:", BASE)
    print("[INFO] Data:", DATA_FILE)
    print("[INFO] Groups:", GROUP_FILE)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    # ================= LOAD DATA =================
    df = pd.read_csv(DATA_FILE)

    required_cols = ["Accession", "Is_Thermophilic_Guess",
                     "Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in data CSV: {missing}")

    # Normalize accession in data (remove .1, .2, ...)
    df["Accession"] = norm_acc(df["Accession"])

    # ================= OPTIONAL MERGE (SANITY) =================
    # We don't *need* clusters for PCA, but we check merge health for pipeline sanity.
    use_merge = False
    if GROUP_FILE.exists():
        g = pd.read_csv(GROUP_FILE)
        if "Accession" in g.columns:
            g["Accession"] = norm_acc(g["Accession"])
            n0 = len(df)
            m = df.merge(g, on="Accession", how="inner")
            print(f"[INFO] Rows in data: {n0}")
            print(f"[INFO] Rows in groups: {len(g)}")
            print(f"[INFO] Rows after merge: {len(m)}")
            if "ClusterID" in m.columns and len(m) > 0:
                print(f"[INFO] Unique clusters after merge: {m['ClusterID'].nunique()}")
            # If merge looks healthy, keep merged df (not mandatory)
            if len(m) >= int(0.9 * n0):
                df = m
                use_merge = True
            else:
                print("[WARN] Merge is small. Proceeding WITHOUT merge for PCA.")
        else:
            print("[WARN] groups_map file missing 'Accession' column. Skipping merge.")
    else:
        print("[WARN] groups_map_cdhit70.csv not found. Proceeding WITHOUT merge for PCA.")

    # ================= FEATURES =================
    features = ["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]
    label_col = "Is_Thermophilic_Guess"

    # Coerce to numeric
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep features with any non-NaN values
    valid = [c for c in features if df[c].notna().any()]
    print("[INFO] Valid numeric features:", valid)

    if len(valid) == 0:
        raise RuntimeError("No usable numeric features found after coercion.")

    X = df[valid].values
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values

    # Impute NaN
    X = SimpleImputer(strategy="median").fit_transform(X)

    # Remove zero-variance columns (avoid PCA/standardization issues)
    var = X.var(axis=0)
    keep = np.where(var > 0)[0]
    X = X[:, keep]
    kept_features = [valid[i] for i in keep]

    print("[INFO] Features kept (variance>0):", kept_features)
    print("[INFO] X shape:", X.shape)

    if X.shape[1] == 0:
        raise RuntimeError(
            "All features have zero variance after preprocessing.\n"
            "This usually means you ended up with too few rows or features are constant."
        )

    # Standardize
    X = StandardScaler().fit_transform(X)

    # ================= PCA (2D or fallback) =================
    if X.shape[1] >= 2:
        Xp = PCA(n_components=2, random_state=0).fit_transform(X)

        # ================= PLOT (Q1 STYLE) =================
        plt.figure(figsize=(7, 6))

        idx_meso = (y == 0)
        plt.scatter(
            Xp[idx_meso, 0], Xp[idx_meso, 1],
            s=22, c="lightgray", alpha=0.6,
            label="Mesophilic", edgecolor="none"
        )

        idx_thermo = (y == 1)
        plt.scatter(
            Xp[idx_thermo, 0], Xp[idx_thermo, 1],
            s=55, c="none", edgecolor="red",
            marker="^", linewidth=1.2,
            label="Thermophilic"
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        ttl = "PCA landscape of GH5 cellulases (CD-HIT 70%)"
        if use_merge:
            ttl += " [merged]"
        plt.title(ttl)
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=300)
        plt.close()

        print("✅ Saved Figure 1 (Q1-style, PCA 2D) ->", OUT_PNG)

    else:
        # 1D fallback if only 1 feature survives
        pc1 = PCA(n_components=1, random_state=0).fit_transform(X).ravel()
        jitter = np.random.default_rng(0).normal(0, 0.02, size=len(pc1))

        plt.figure(figsize=(7, 3.5))

        idx_meso = (y == 0)
        plt.scatter(
            pc1[idx_meso], jitter[idx_meso],
            s=22, c="lightgray", alpha=0.6,
            label="Mesophilic", edgecolor="none"
        )

        idx_thermo = (y == 1)
        plt.scatter(
            pc1[idx_thermo], jitter[idx_thermo],
            s=55, c="none", edgecolor="red",
            marker="^", linewidth=1.2,
            label="Thermophilic"
        )

        plt.yticks([])
        plt.xlabel("PC1 (1D fallback)")
        ttl = "Sequence landscape of GH5 cellulases (CD-HIT 70%)"
        if use_merge:
            ttl += " [merged]"
        plt.title(ttl)
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=300)
        plt.close()

        print("⚠️ Only 1 usable feature. Saved Figure 1 (Q1-style, 1D fallback) ->", OUT_PNG)


if __name__ == "__main__":
    main()
