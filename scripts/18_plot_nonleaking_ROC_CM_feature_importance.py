#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18_plot_nonleaking_ROC_CM_feature_importance.py

Rebuild non-leaking OOF predictions (GroupKFold by CD-HIT clusters),
then plot:
- ROC curve (from OOF)
- Confusion Matrix
- Feature importance (mean ± SD across folds)

Fixed paths:
  ai_training/gh5_training_data_with_features.csv
  ai_training/groups_map_cdhit80.csv
Outputs to:
  ai_training/nonleaking_rf_final/

Run:
  python 18_plot_nonleaking_ROC_CM_feature_importance.py
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
)

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent.parent
AI_DIR = BASE_DIR / "ai_training"

DATA_CSV = AI_DIR / "gh5_training_data_with_features.csv"
GROUPS_CSV = AI_DIR / "groups_map_cdhit80.csv"
OUT_DIR = AI_DIR / "nonleaking_rf_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONFIG =================
FEATURE_COLS = ["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]
LABEL_COL = "Is_Thermophilic_Guess"
ACC_COL = "Accession"

OUTER_SPLITS = 5
SEED = 42

# Use the SAME simple model as your minimal Step 17 (stable, no GridSearch)
RF_PARAMS = dict(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1
)

# ================= HELPERS =================
def normalize_accession(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2:
            s = parts[1]
    s = re.sub(r"-\d+$", "", s)
    s = re.sub(r"\.\d+$", "", s)
    return s.strip()

def safe_predict_proba_1(model, X):
    """Return P(class=1) safely even if model learned single class."""
    if not hasattr(model, "classes_") or len(model.classes_) < 2:
        return np.zeros(X.shape[0], dtype=float)
    # find index of class 1
    cls = list(model.classes_)
    if 1 not in cls:
        return np.zeros(X.shape[0], dtype=float)
    j = cls.index(1)
    return model.predict_proba(X)[:, j]

# ================= MAIN =================
def main():
    print("Loading data for plotting ...")
    df = pd.read_csv(DATA_CSV)
    gmap = pd.read_csv(GROUPS_CSV)

    df["_acc"] = df[ACC_COL].map(normalize_accession)
    gmap["_acc"] = gmap["Accession"].map(normalize_accession)

    df = df.merge(gmap[["_acc", "ClusterID"]], on="_acc", how="inner")

    # build arrays
    X = df[FEATURE_COLS].to_numpy(float)
    y = df[LABEL_COL].astype(int).to_numpy()
    groups = df["ClusterID"].to_numpy()

    # drop rows with NaN features
    mask = ~np.isnan(X).any(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    X, y, groups = X[mask], y[mask], groups[mask]

    print(f"Samples used: {len(df)}")
    print("Features:", FEATURE_COLS)

    outer = GroupKFold(n_splits=OUTER_SPLITS)

    oof_rows = []
    importances = []

    # ---- build OOF predictions ----
    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # if TRAIN has single class, fallback
        if len(np.unique(ytr)) < 2:
            maj = int(np.unique(ytr)[0])
            proba = np.full(len(yte), float(maj))
            pred = np.full(len(yte), maj, dtype=int)
            print(f"[Fold {fold}] TRAIN single-class -> fallback (majority={maj})")
        else:
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(Xtr, ytr)

            proba = safe_predict_proba_1(model, Xte)
            pred = (proba >= 0.5).astype(int)

            # feature importance
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)

        for i, idx in enumerate(te):
            oof_rows.append({
                "Accession": df.iloc[idx][ACC_COL],
                "ClusterID": int(groups[idx]),
                "fold": fold,
                "y_true": int(yte[i]),
                "y_pred": int(pred[i]),
                "y_proba": float(proba[i]),
            })

    oof = pd.DataFrame(oof_rows)
    oof_path = OUT_DIR / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    print(f"Saved OOF -> {oof_path}")

    # ---- overall metrics ----
    y_true = oof["y_true"].to_numpy()
    y_pred = oof["y_pred"].to_numpy()
    y_proba = oof["y_proba"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan

    (OUT_DIR / "summary_from_oof.txt").write_text(
        f"OOF (non-leaking GroupKFold)\n"
        f"AUC={auc:.4f}\nACC={acc:.4f}\nF1={f1:.4f}\nMCC={mcc:.4f}\n",
        encoding="utf-8"
    )

    # ---- ROC curve ----
    if len(np.unique(y_true)) == 2:
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
        roc_df.to_csv(OUT_DIR / "roc_points.csv", index=False)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Non-leaking ROC (OOF) | AUC={auc:.3f}")
        plt.tight_layout()
        roc_png = OUT_DIR / "ROC_nonleaking.png"
        plt.savefig(roc_png, dpi=300)
        plt.close()
        print(f"Saved -> {roc_png}")
    else:
        print("[WARN] ROC not defined (only one class in y_true overall). Skipping ROC plot.")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Non-leaking Confusion Matrix (OOF)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    cm_png = OUT_DIR / "ConfusionMatrix_nonleaking.png"
    plt.savefig(cm_png, dpi=300)
    plt.close()
    print(f"Saved -> {cm_png}")

    # ---- Feature importance (mean ± SD) ----
    if importances:
        imp = np.vstack(importances)
        mean = imp.mean(axis=0)
        sd = imp.std(axis=0)

        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_importance": mean,
            "sd_importance": sd
        }).sort_values("mean_importance", ascending=False)

        imp_df.to_csv(OUT_DIR / "feature_importance_mean_sd.csv", index=False)

        plt.figure()
        plt.bar(imp_df["feature"], imp_df["mean_importance"], yerr=imp_df["sd_importance"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance (mean ± SD)")
        plt.title("Non-leaking Feature Importance (across folds)")
        plt.tight_layout()
        fi_png = OUT_DIR / "FeatureImportance_nonleaking.png"
        plt.savefig(fi_png, dpi=300)
        plt.close()
        print(f"Saved -> {fi_png}")
    else:
        print("[WARN] No feature importance collected (all folds were fallback?). Skipping importance plot.")

    print("\nDONE: All plots/results saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

