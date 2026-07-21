#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent.parent
AI_DIR = BASE_DIR / "ai_training"

DATA_CSV = AI_DIR / "gh5_training_data_with_features.csv"
GROUPS_CSV = AI_DIR / "groups_map_cdhit80.csv"
OUT_DIR = AI_DIR / "nonleaking_rf_final"

FEATURE_COLS = [
    "Length", "Aromaticity", "GRAVY",
    "AliphaticIndex", "InstabilityIndex"
]

LABEL_COL = "Is_Thermophilic_Guess"
ACC_COL = "Accession"


def normalize_accession(x):
    if pd.isna(x):
        return ""
    s = str(x)
    if "|" in s:
        s = s.split("|")[1]
    s = re.sub(r"-\d+$", "", s)
    s = re.sub(r"\.\d+$", "", s)
    return s.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    gmap = pd.read_csv(GROUPS_CSV)

    df["_acc"] = df[ACC_COL].map(normalize_accession)
    gmap["_acc"] = gmap["Accession"].map(normalize_accession)

    df = df.merge(gmap[["_acc", "ClusterID"]], on="_acc", how="inner")

    X = df[FEATURE_COLS].to_numpy(float)
    y = df[LABEL_COL].astype(int).to_numpy()
    groups = df["ClusterID"].to_numpy()

    mask = ~np.isnan(X).any(axis=1)
    X, y, groups = X[mask], y[mask], groups[mask]
    df = df.loc[mask].reset_index(drop=True)

    outer = GroupKFold(n_splits=5)

    rows = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        if len(np.unique(ytr)) < 2:
            pred = np.full(len(yte), ytr[0])
            proba = pred.astype(float)
            auc = np.nan
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
            model.fit(Xtr, ytr)

            if len(model.classes_) < 2:
                proba = np.zeros(len(yte))
            else:
                proba = model.predict_proba(Xte)[:, 1]

            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(yte, proba) if len(np.unique(yte)) == 2 else np.nan

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, zero_division=0)
        mcc = matthews_corrcoef(yte, pred)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()

        rows.append({
            "fold": fold,
            "auc": auc,
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        })

        print(f"[Fold {fold}] AUC={auc if not np.isnan(auc) else 'NA'} ACC={acc:.3f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    print("DONE. Results saved to", OUT_DIR)


if __name__ == "__main__":
    main()

