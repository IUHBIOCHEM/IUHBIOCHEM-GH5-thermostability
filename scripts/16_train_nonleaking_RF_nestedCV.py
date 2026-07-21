#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix


NON_FEATURE_HINTS = {
    "sequence", "organism", "description", "name", "protein", "label", "clusterid"
}


def normalize_accession(x: str) -> str:
    """
    Normalize accession keys to maximize join success.
    Examples handled:
      - "sp|P12345|ABC" -> "P12345"
      - "tr|A0A0B1|..." -> "A0A0B1"
      - "A0A0B1.1" -> "A0A0B1"
      - "A0A0B1-1" -> "A0A0B1"
      - strips whitespace
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # take middle part if pipe format
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2:
            s = parts[1].strip()
    # remove isoform suffix -1, -2...
    s = re.sub(r"-\d+$", "", s)
    # remove version suffix .1 .2 ...
    s = re.sub(r"\.\d+$", "", s)
    return s.strip()


def find_col_case_insensitive(df: pd.DataFrame, wanted: str) -> str:
    if wanted in df.columns:
        return wanted
    low = {c.lower(): c for c in df.columns}
    if wanted.lower() in low:
        return low[wanted.lower()]
    raise ValueError(f"Column not found: {wanted} (available: {list(df.columns)[:30]}...)")

def auto_feature_cols(df: pd.DataFrame, label_col: str, acc_col: str) -> list:
    # try to coerce everything except obvious text columns
    candidates = []
    for c in df.columns:
        if c in {label_col, acc_col, "ClusterID"}:
            continue
        if c.lower() in NON_FEATURE_HINTS:
            continue
        # attempt numeric coercion check
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)

    # If none numeric dtype, try coercion for all non-excluded cols
    if not candidates:
        for c in df.columns:
            if c in {label_col, acc_col, "ClusterID"}:
                continue
            if c.lower() in NON_FEATURE_HINTS:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        candidates = [c for c in df.columns
                      if c not in {label_col, acc_col, "ClusterID"}
                      and pd.api.types.is_numeric_dtype(df[c])]

    # remove cols that are all NaN
    candidates = [c for c in candidates if df[c].notna().sum() > 0]
    return candidates


def main():
    ap = argparse.ArgumentParser(description="Non-leaking RF with nested GroupKFold (CD-HIT clusters).")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--groups_csv", required=True)
    ap.add_argument("--out_dir", default="ai_training/nonleaking_rf_v1")

    ap.add_argument("--label_col", default="Is_Thermophilic_Guess")
    ap.add_argument("--accession_col", default="Accession")  # will be found case-insensitive
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    gmap = pd.read_csv(args.groups_csv)

    acc_col = find_col_case_insensitive(df, args.accession_col)
    label_col = find_col_case_insensitive(df, args.label_col)
    g_acc_col = find_col_case_insensitive(gmap, "Accession")
    g_cluster_col = find_col_case_insensitive(gmap, "ClusterID")

    # normalize accessions
    df["_acc_norm"] = df[acc_col].map(normalize_accession)
    gmap["_acc_norm"] = gmap[g_acc_col].map(normalize_accession)

    # quick overlap diagnostics
    inter = set(df["_acc_norm"]) & set(gmap["_acc_norm"])
    print(f"Accession overlap: {len(inter)} / data={df['_acc_norm'].nunique()} / groups={gmap['_acc_norm'].nunique()}")
    if len(inter) == 0:
        print("[DEBUG] Example accessions from data:", df["_acc_norm"].dropna().astype(str).head(5).tolist())
        print("[DEBUG] Example accessions from groups:", gmap["_acc_norm"].dropna().astype(str).head(5).tolist())
        raise RuntimeError("No overlap between data and groups accessions. Check parsing of .clstr or accession formats.")

    # merge using normalized key
    m = df.merge(gmap[["_acc_norm", g_cluster_col]], on="_acc_norm", how="inner")
    m = m.rename(columns={g_cluster_col: "ClusterID"}).copy()

    print(f"Merged rows: {len(m)} / {len(df)}")

    # drop missing label
    m = m.dropna(subset=[label_col]).copy()
    y = m[label_col].astype(int).to_numpy()
    groups = m["ClusterID"].astype(int).to_numpy()

    # detect / coerce feature columns
    feature_cols = auto_feature_cols(m, label_col=label_col, acc_col=acc_col)
    if not feature_cols:
        print("[DEBUG] Columns:", list(m.columns))
        raise RuntimeError("No numeric feature columns detected after coercion. Check your descriptor columns.")

    # build X
    X = m[feature_cols].to_numpy(dtype=float)

    print(f"Samples used: {len(m)} | Features: {len(feature_cols)} | Clusters: {len(np.unique(groups))}")
    print("First features:", feature_cols[:10])

    pipe = Pipeline([
        ("rf", RandomForestClassifier(random_state=args.random_state, n_jobs=args.n_jobs))
    ])

    param_grid = {
        "rf__n_estimators": [300, 600, 1000],
        "rf__max_depth": [None, 10, 30],
        "rf__max_features": ["sqrt", 0.3, 0.5],
        "rf__min_samples_leaf": [1, 2, 5],
        "rf__class_weight": [None, "balanced"],
    }

    outer_cv = GroupKFold(n_splits=min(args.outer_splits, len(np.unique(groups))))

    fold_metrics = []
    oof_rows = []
    best_params = {}

    for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        g_tr = groups[tr]

        inner_splits = min(args.inner_splits, len(np.unique(g_tr)))
        inner_cv = GroupKFold(n_splits=inner_splits)

        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv.split(X_tr, y_tr, groups=g_tr),
            n_jobs=args.n_jobs,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best_params[f"fold_{fold}"] = search.best_params_

        model = search.best_estimator_
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) == 2 else np.nan
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        mcc = matthews_corrcoef(y_te, pred) if len(np.unique(y_te)) == 2 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

        fold_metrics.append({
            "fold": fold,
            "n_test": len(te),
            "auc": auc,
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

        for j, idx in enumerate(te):
            oof_rows.append({
                "Accession": m.iloc[idx][acc_col],
                "Accession_norm": m.iloc[idx]["_acc_norm"],
                "ClusterID": int(groups[idx]),
                "fold": fold,
                "y_true": int(y_te[j]),
                "y_pred": int(pred[j]),
                "y_proba": float(proba[j]),
            })

        print(f"[Fold {fold}] AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f} MCC={mcc:.4f}")

    fold_df = pd.DataFrame(fold_metrics)
    oof_df = pd.DataFrame(oof_rows)

    fold_df.to_csv(out_dir / "results_fold_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "results_oof_predictions.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(out_dir / "features_used.csv", index=False)

    with open(out_dir / "best_params_per_fold.json", "w") as f:
        json.dump(best_params, f, indent=2)

    auc_mean, auc_sd = float(np.nanmean(fold_df["auc"])), float(np.nanstd(fold_df["auc"]))
    acc_mean, acc_sd = float(np.nanmean(fold_df["accuracy"])), float(np.nanstd(fold_df["accuracy"]))
    f1_mean, f1_sd = float(np.nanmean(fold_df["f1"])), float(np.nanstd(fold_df["f1"]))
    mcc_mean, mcc_sd = float(np.nanmean(fold_df["mcc"])), float(np.nanstd(fold_df["mcc"]))

    summary = (
        "Non-leaking RF (Nested GroupKFold)\n"
        f"Samples: {len(m)} | Features: {len(feature_cols)} | Clusters: {len(np.unique(groups))}\n"
        f"AUC: {auc_mean:.4f} ± {auc_sd:.4f}\n"
        f"ACC: {acc_mean:.4f} ± {acc_sd:.4f}\n"
        f"F1 : {f1_mean:.4f} ± {f1_sd:.4f}\n"
        f"MCC: {mcc_mean:.4f} ± {mcc_sd:.4f}\n"
    )
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
