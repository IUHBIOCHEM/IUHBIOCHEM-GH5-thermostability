"""
29_topt_classifier_esm2.py

Phân loại nhị phân "cellulase bền nhiệt" (Topt >= ngưỡng, mặc định 60°C) dựa trên
Topt enzyme THẬT từ BRENDA (EC 3.2.1.4), dùng embedding ESM2-650M.
Đánh giá không rò rỉ bằng GroupKFold theo cụm cd-hit.

Lý do dùng phân loại thay vì hồi quy: nhãn Topt (nối theo loài, 56% dồn ở 60°C)
gần như hằng số trong từng cụm -> hồi quy thất bại (R2 âm), nhưng ranh giới
bền nhiệt/không vẫn học được tốt (AUC ~0.80).

Input : ai_training/gh5_topt_features.csv, ai_training/groups_map_topt_cdhit70.csv
        ai_training/topt_esm2_650M/esm2_650M_emb.npy  (embedding đã cache từ script 26)
Output: ai_training/topt_classifier/
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, roc_curve, confusion_matrix)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "ai_training/gh5_topt_features.csv"
GROUPS = "ai_training/groups_map_topt_cdhit70.csv"
EMB = "ai_training/topt_esm2_650M/esm2_650M_emb.npy"
OUT = "ai_training/topt_classifier"
THRESHOLD = 60.0


def norm_acc(x):
    return str(x).split(".")[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    g = pd.read_csv(GROUPS)
    df["_acc"] = df["Accession"].map(norm_acc)
    g["_acc"] = g["Accession"].map(norm_acc)
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner")
    df = df.dropna(subset=["Topt"]).reset_index(drop=True)

    X = np.load(EMB)
    assert len(X) == len(df), f"embedding {len(X)} != rows {len(df)} (chạy script 26 --target Topt trước)"

    y = (df["Topt"].to_numpy() >= THRESHOLD).astype(int)
    groups = df["ClusterID"].to_numpy()
    print(f"n={len(y)} | bền nhiệt(>= {THRESHOLD:.0f}°C)={y.sum()} | không={len(y)-y.sum()} | "
          f"groups={len(np.unique(groups))}")

    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(y[tr][0])
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    pred = (oof >= 0.5).astype(int)
    auc = roc_auc_score(y, oof)
    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    print(f"\n=== OOF (GroupKFold không rò rỉ) ===")
    print(f"AUC={auc:.3f}  Accuracy={acc:.3f}  BalancedAcc={bacc:.3f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)

    pd.DataFrame([{"threshold": THRESHOLD, "n": len(y), "n_pos": int(y.sum()),
                   "AUC": auc, "Accuracy": acc, "BalancedAcc": bacc}]).to_csv(
        f"{OUT}/metrics.csv", index=False)
    out = df[["Accession", "Organism", "Topt"]].copy()
    out["y_true"] = y
    out["prob_thermostable"] = oof
    out.to_csv(f"{OUT}/oof_predictions.csv", index=False)

    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"GH5 thermostable (Topt≥{THRESHOLD:.0f}°C) — ESM2-650M\nn={len(y)}, GroupKFold")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{OUT}/roc_curve.png", dpi=150)

    sc = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    joblib.dump({"scaler": sc, "model": final, "threshold": THRESHOLD,
                 "embedding": "esm2_t33_650M layer33 mean-pool"}, f"{OUT}/topt_clf_model.pkl")
    print(f"\n→ Kết quả tại {OUT}/")


if __name__ == "__main__":
    main()
