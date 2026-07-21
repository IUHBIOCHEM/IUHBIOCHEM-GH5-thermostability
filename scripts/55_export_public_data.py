"""55_export_public_data.py — build clean, public-facing dataset CSVs in data_public/.

Four self-describing tables with standardized column names, assembled (no new analysis) from the
curated CSVs in ai_training/. Run from anywhere; paths resolve to the repository root.

  data_public/metadata.csv       one row per sequence (accession, species, OGT, class, length, cluster, predicted OGT)
  data_public/hotspots.csv       66 charge-adaptation hotspot positions (position, residue, p, FDR, enrichment, surface/buried)
  data_public/structures.csv     82 GH5 PDB structures (salt bridges, charged / surface residues)
  data_public/FoldX_results.csv  FoldX ddG for single substitutions and their combinations
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, "ai_training")
A = os.path.join(AI, "analysis")
OUT = os.path.join(ROOT, "data_public")
os.makedirs(OUT, exist_ok=True)

OGT_THRESHOLD = 55.0  # >= 55 C -> thermophile (manuscript definition)


def _base(acc):  # NCBI accession without the version suffix (WP_x.1 -> WP_x)
    return acc.str.replace(r"\.\d+$", "", regex=True)


def build_metadata():
    lab = pd.read_csv(f"{AI}/gh5_ogt_labeled.csv")
    pred = pd.read_csv(f"{A}/all757_predictions.csv")[["Accession", "OGT_pred", "P_thermostable"]]
    clus = pd.read_csv(f"{AI}/groups_map_cdhit70.csv").rename(columns={"ClusterID": "cluster"})
    clus["_k"] = _base(clus["Accession"])
    lab["_k"] = _base(lab["Accession"])
    df = (lab.merge(pred, on="Accession", how="left")
             .merge(clus[["_k", "cluster"]], on="_k", how="left"))

    def cls(o):
        if pd.isna(o):
            return ""
        return "thermophile" if o >= OGT_THRESHOLD else "mesophile"

    out = pd.DataFrame({
        "accession": df["Accession"],
        "species": df["Organism"],
        "OGT": df["OGT"].round(2),
        "class": df["OGT"].apply(cls),
        "sequence_length": df["Sequence"].str.len(),
        "cluster": df["cluster"],
        "predicted_OGT": df["OGT_pred"].round(2),
        "P_thermostable": df["P_thermostable"].round(4),
    })
    out.to_csv(f"{OUT}/metadata.csv", index=False)
    n_t = (out["class"] == "thermophile").sum()
    n_m = (out["class"] == "mesophile").sum()
    print(f"metadata.csv: {len(out)} sequences ({n_t} thermophile, {n_m} mesophile, "
          f"{out['cluster'].notna().sum()} with cluster)")


def build_hotspots():
    supp = pd.read_csv(f"{A}/hotspot_supptable.csv")
    stat = pd.read_csv(f"{A}/thermal_hotspots.csv")
    # FDR ties (5 of 66) make it non-unique; join on the (thermo%, meso%) charge pair instead,
    # which is unique per hotspot in both tables, to recover the raw p-value.
    stat["_k"] = list(zip((stat["thermo_charged_frac"] * 100).round(1),
                          (stat["meso_charged_frac"] * 100).round(1)))
    supp["_k"] = list(zip(supp["thermo_charged_%"].round(1), supp["meso_charged_%"].round(1)))
    df = supp.merge(stat[["_k", "p"]], on="_k", how="left")
    loc_map = {"surface": "surface", "core": "buried"}
    out = pd.DataFrame({
        "alignment_position": df["PF00150_col"].astype("Int64"),
        "residue_number_3AMC": df["res3AMC"].astype("Int64"),
        "residue_3AMC": df["res3AMC_aa"],
        "thermophile_preferred_residue": df["thermo_pref_residue"],
        "thermo_charged_pct": df["thermo_charged_%"],
        "meso_charged_pct": df["meso_charged_%"],
        "enrichment_delta_pct": df["delta_%"],
        "p_value": df["p"],
        "FDR": df["fdr"],
        "location": df["location"].map(loc_map).fillna("unmapped"),   # surface / buried / unmapped
        "relative_solvent_accessibility": df["RSA"],
        "distance_to_catalytic_Glu_A": df["dist_catGlu_A"],
    }).sort_values("FDR")
    out.to_csv(f"{OUT}/hotspots.csv", index=False)
    vc = out["location"].value_counts()
    print(f"hotspots.csv: {len(out)} positions "
          f"({vc.get('surface', 0)} surface, {vc.get('buried', 0)} buried, {vc.get('unmapped', 0)} unmapped)")


def build_structures():
    s = pd.read_csv(f"{A}/saltbridge_survey.csv")
    out = pd.DataFrame({
        "pdb": s["pdb"],
        "organism": s["organism"],
        "thermal_class": s["group"],
        "n_residues": s["n_res"],
        "salt_bridges": s["salt_bridges"],
        "salt_bridges_per_100res": s["sb_per100"].round(2),
        "surface_charged_residues": (s["surf_charged_per100"] * s["n_res"] / 100).round().astype("Int64"),
        "surface_charged_per_100res": s["surf_charged_per100"].round(2),
    }).sort_values(["thermal_class", "pdb"])
    out.to_csv(f"{OUT}/structures.csv", index=False)
    print(f"structures.csv: {len(out)} PDB structures "
          f"({(out['thermal_class']=='thermophile').sum()} thermophile, "
          f"{(out['thermal_class']=='mesophile').sum()} mesophile)")


def build_foldx():
    singles = pd.read_csv(f"{A}/ddg_combined.csv")
    combos = pd.read_csv(f"{A}/combo_ddg.csv")
    s = pd.DataFrame({
        "mutation": singles["mutation"],
        "n_mutations": 1,
        "foldx_ddG_kcal_per_mol": singles["FoldX_ddG"],
        "foldx_SD": singles["FoldX_SD"],
        "additive_expectation": np.nan,
        "coupling_energy": np.nan,
        "verdict": singles["FoldX_verdict"],
        "ESM2_log_likelihood_ratio": singles["ESM2_LLR"],
        "family_signal_delta_pct": singles["family_delta_%"],
    })
    c = pd.DataFrame({
        "mutation": combos["combo"],
        "n_mutations": combos["n"],
        "foldx_ddG_kcal_per_mol": combos["obs"],
        "foldx_SD": np.nan,
        "additive_expectation": combos["add"],
        "coupling_energy": combos["epi"],
        "verdict": np.nan,
        "ESM2_log_likelihood_ratio": np.nan,
        "family_signal_delta_pct": np.nan,
    })
    out = pd.concat([s, c], ignore_index=True)
    out.to_csv(f"{OUT}/FoldX_results.csv", index=False)
    print(f"FoldX_results.csv: {len(s)} single mutations + {len(c)} combinations")


def main():
    build_metadata()
    build_hotspots()
    build_structures()
    build_foldx()
    print(f"\nwrote 4 public CSVs to {OUT}")


if __name__ == "__main__":
    main()
