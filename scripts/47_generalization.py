"""
47_generalization.py — Tổng quát hoá khung + cơ chế sang các họ CAZy khác.

Cho mỗi họ (GH5, GH1, GH10 = clan GH-A TIM-barrel; GH11 = jelly-roll đối chứng):
  (1) Method: ESM2-150M + RidgeCV, GroupKFold theo cụm cd-hit 70% -> R²/RMSE/Spearman (dự đoán OGT).
  (2) Discovery: align vào PF của họ, Fisher charged thermo(OGT>=55) vs meso + BH-FDR -> số hotspot;
      signature (charged tại hotspot) tương quan OGT.
Xuất bảng so sánh + hình.

Output: manuscript/figures/Fig17_generalization.png , ai_training/analysis/generalization.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re, subprocess
import numpy as np
import pandas as pd
import torch, esm
import pyhmmer
from scipy.stats import fisher_exact, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

alphabet = pyhmmer.easel.Alphabet.amino()
CHARGED = set("DEKR")
CACHE = "ai_training/family_emb"; os.makedirs(CACHE, exist_ok=True)
FAMS = [("GH5", "PF00150", "clan GH-A (TIM-barrel)"),
        ("GH1", "PF00232", "clan GH-A (TIM-barrel)"),
        ("GH10", "PF00331", "clan GH-A (TIM-barrel)"),
        ("GH11", "PF00457", "jelly-roll (control)")]


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def load_family(gh):
    if gh == "GH5":
        d = pd.read_csv("ai_training/gh5_ogt_labeled.csv")
    else:
        d = pd.read_csv(f"data_family/{gh}_labeled.csv")
    d = d.dropna(subset=["OGT"]).reset_index(drop=True)
    d["seq"] = d["Sequence"].map(clean)
    d = d[d["seq"].str.len() >= 80].reset_index(drop=True)
    return d


_model = None; _bc = None


def embed(gh, seqs):
    cp = f"{CACHE}/{gh}_150M.npy"
    if os.path.exists(cp):
        return np.load(cp)
    global _model, _bc
    if _model is None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _model, alph = esm.pretrained.esm2_t30_150M_UR50D(); _model = _model.to(dev).eval(); _bc = alph.get_batch_converter()
    dev = next(_model.parameters()).device
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs, 1):
            _, _, toks = _bc([("x", s[:1022])])
            rep = _model(toks.to(dev), repr_layers=[30])["representations"][30]
            out.append(rep[:, 1:-1, :].mean(1).squeeze(0).cpu().numpy())
            if i % 100 == 0:
                print(f"    {gh} emb {i}/{len(seqs)}", flush=True)
    X = np.vstack(out).astype(np.float32); np.save(cp, X); return X


def cluster_groups(gh, df):
    fa = f"{CACHE}/{gh}.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(df["seq"]):
            f.write(f">{i}\n{s}\n")
    subprocess.run(["cd-hit", "-i", fa, "-o", f"{CACHE}/{gh}70", "-c", "0.7", "-n", "5", "-T", "4", "-M", "0"],
                   capture_output=True)
    groups = np.arange(len(df)); cid = -1
    for line in open(f"{CACHE}/{gh}70.clstr"):
        if line.startswith(">Cluster"):
            cid += 1
        else:
            m = re.search(r">(\d+)\.\.\.", line)
            if m:
                groups[int(m.group(1))] = cid
    return groups


def regress(X, y, groups):
    gkf = GroupKFold(min(5, len(np.unique(groups)))); oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr]); m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), spearmanr(y, oof).correlation


def hotspots(gh, pf, df):
    with pyhmmer.plan7.HMMFile(f"pfam/{pf}.hmm") as f:
        hmm = f.read()
    seqs = [pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet) for i, s in enumerate(df["seq"])]
    msa = pyhmmer.hmmalign(hmm, seqs, trim=True)
    aln = [x.decode() if isinstance(x, bytes) else x for x in msa.alignment]
    arr = np.array([list(x) for x in aln])
    mcols = [j for j in range(arr.shape[1]) if all((c == "-") or c.isupper() for c in arr[:, j])]
    y = df["OGT"].values
    th = np.where(y >= 55)[0]; me = np.where(y < 55)[0]
    if len(th) < 8:
        return np.nan, np.nan, len(th)
    rec = []
    for j in mcols:
        tn = arr[th, j]; tn = tn[tn != "-"]; mn = arr[me, j]; mn = mn[mn != "-"]
        if len(tn) < 8 or len(mn) < 8:
            continue
        tc = sum(c in CHARGED for c in tn); mc = sum(c in CHARGED for c in mn)
        _, p = fisher_exact([[tc, len(tn) - tc], [mc, len(mn) - mc]])
        rec.append({"col": j, "delta": tc / len(tn) - mc / len(mn), "p": p})
    R = pd.DataFrame(rec); R["fdr"] = bh(R["p"].values)
    hot = list(R[(R["fdr"] < 0.05) & (R["delta"] > 0)]["col"])
    if not hot:
        return 0, np.nan, len(th)
    sig = []
    for i in range(len(df)):
        v = [arr[i, j] for j in hot if arr[i, j] != "-"]
        sig.append(sum(c in CHARGED for c in v) / len(v) if v else np.nan)
    sig = np.array(sig); ok = ~np.isnan(sig)
    rho = spearmanr(sig[ok], y[ok]).correlation
    return len(hot), rho, len(th)


def main():
    rows = []
    for gh, pf, fold in FAMS:
        df = load_family(gh)
        print(f"[{gh}] n={len(df)}, thermo={int((df['OGT']>=55).sum())}", flush=True)
        X = embed(gh, df["seq"].tolist())
        groups = cluster_groups(gh, df)
        r2, rmse, rho = regress(X, y := df["OGT"].values, groups)
        nhot, sig_rho, nth = hotspots(gh, pf, df)
        rows.append({"family": gh, "fold": fold, "n": len(df), "n_thermo": nth,
                     "R2": round(r2, 3), "RMSE": round(rmse, 1), "Spearman": round(rho, 3),
                     "n_hotspots": nhot, "signature_rho": round(sig_rho, 3) if sig_rho == sig_rho else np.nan})
        print(f"  {gh}: R2={r2:.2f} | hotspots={nhot} sig_rho={sig_rho}")
    T = pd.DataFrame(rows); T.to_csv("ai_training/analysis/generalization.csv", index=False)
    print("\n" + T.to_string(index=False))

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fams = T["family"]; col = ["#2166ac" if "GH-A" in f else "#b2182b" for f in T["fold"]]
    ax[0].bar(fams, T["R2"], color=col); ax[0].set_ylim(0, 1); ax[0].set_ylabel("R² (OOF, GroupKFold)")
    ax[0].set_title("a  OGT prediction per family", fontweight="bold", loc="left")
    for i, v in enumerate(T["R2"]): ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].bar(fams, T["n_hotspots"], color=col); ax[1].set_ylabel("# charge hotspots (FDR<0.05)")
    ax[1].set_title("b  Position-specific adaptation", fontweight="bold", loc="left")
    for i, v in enumerate(T["n_hotspots"]): ax[1].text(i, v + 1, f"{int(v)}", ha="center", fontsize=9)
    ax[2].bar(fams, T["signature_rho"], color=col); ax[2].set_ylim(0, 1); ax[2].set_ylabel("signature–OGT Spearman")
    ax[2].set_title("c  Charge signature explains OGT", fontweight="bold", loc="left")
    for i, v in enumerate(T["signature_rho"]):
        if v == v: ax[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    import matplotlib.patches as mp
    ax[0].legend(handles=[mp.Patch(color="#2166ac", label="clan GH-A (TIM-barrel)"),
                          mp.Patch(color="#b2182b", label="jelly-roll (control)")], fontsize=8)
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig17_generalization.png", dpi=200)
    print("→ Fig17_generalization.png")


if __name__ == "__main__":
    main()
