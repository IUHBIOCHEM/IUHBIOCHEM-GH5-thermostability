"""
25_train_ogt_regression.py

Hồi quy OGT (nhiệt độ sinh trưởng tối ưu, proxy cho độ bền nhiệt) của GH5 cellulase
từ đặc trưng trình tự, đánh giá không rò rỉ bằng GroupKFold theo cụm cd-hit.

Input:
  ai_training/gh5_ogt_features.csv        (đặc trưng + cột OGT)
  ai_training/groups_map_ogt_cdhit70.csv  (Accession -> ClusterID)
Output:
  ai_training/ogt_regression/  (metrics, OOF predictions, scatter, model.pkl)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "ai_training/gh5_ogt_features.csv"
GROUPS = "ai_training/groups_map_ogt_cdhit70.csv"
OUT = "ai_training/ogt_regression"
FEATURES = ["Length", "Aromaticity", "GRAVY", "AliphaticIndex", "InstabilityIndex"]
TARGET = "OGT"


def norm_acc(x):
    return str(x).split(".")[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    g = pd.read_csv(GROUPS)

    df["_acc"] = df["Accession"].map(norm_acc)
    g["_acc"] = g["Accession"].map(norm_acc)
    df = df.merge(g[["_acc", "ClusterID"]], on="_acc", how="inner")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    groups = df["ClusterID"].to_numpy()
    print(f"n={len(df)} | n_groups={len(np.unique(groups))} | "
          f"OGT range {y.min():.0f}-{y.max():.0f}, mean {y.mean():.1f}")

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                      n_jobs=-1, random_state=42)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        rmse = np.sqrt(mean_squared_error(y[te], oof[te]))
        print(f"  fold {fold}: n_test={len(te)}  RMSE={rmse:.2f}")

    rmse = np.sqrt(mean_squared_error(y, oof))
    mae = mean_absolute_error(y, oof)
    r2 = r2_score(y, oof)
    rho = spearmanr(y, oof).correlation
    # baseline: dự đoán trung bình
    base_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))

    print("\n=== OOF (GroupKFold, không rò rỉ) ===")
    print(f"RMSE={rmse:.2f}°C  MAE={mae:.2f}°C  R2={r2:.3f}  Spearman={rho:.3f}")
    print(f"Baseline (mean) RMSE={base_rmse:.2f}°C  -> giảm {100*(1-rmse/base_rmse):.0f}%")

    # lưu kết quả
    df_out = df[["Accession", "Organism", TARGET]].copy()
    df_out["OGT_pred"] = oof
    df_out.to_csv(f"{OUT}/oof_predictions.csv", index=False)
    pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman": rho,
                   "baseline_RMSE": base_rmse, "n": len(y),
                   "n_groups": len(np.unique(groups))}]).to_csv(
        f"{OUT}/metrics.csv", index=False)

    # scatter thực vs dự đoán
    plt.figure(figsize=(5, 5))
    plt.scatter(y, oof, s=18, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y.min(), oof.min()) - 3, max(y.max(), oof.max()) + 3]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("Measured OGT (°C)"); plt.ylabel("Predicted OGT (°C)")
    plt.title(f"GH5 OGT regression (GroupKFold)\nR²={r2:.2f}, RMSE={rmse:.1f}°C, n={len(y)}")
    plt.tight_layout(); plt.savefig(f"{OUT}/scatter_true_vs_pred.png", dpi=150)

    # feature importance từ mô hình cuối train trên toàn bộ
    final = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                  n_jobs=-1, random_state=42).fit(X, y)
    joblib.dump(final, f"{OUT}/ogt_rf_model.pkl")
    imp = pd.DataFrame({"feature": FEATURES, "importance": final.feature_importances_}
                       ).sort_values("importance", ascending=False)
    imp.to_csv(f"{OUT}/feature_importance.csv", index=False)
    print("\nĐộ quan trọng đặc trưng:")
    print(imp.to_string(index=False))
    print(f"\n→ Kết quả lưu tại {OUT}/")


if __name__ == "__main__":
    main()
