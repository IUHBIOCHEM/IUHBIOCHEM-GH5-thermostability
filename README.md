# Protein language models reveal a conserved evolutionary mechanism of thermal adaptation in TIM-barrel glycoside hydrolases

Code and curated data for the manuscript. Rather than only *predicting* thermostability, this work
uses protein language models to *reveal* the evolutionary mechanism by which the (β/α)₈ TIM-barrel
fold gains heat stability. The central principle: **evolution preserves the catalytic core while
concentrating adaptive mutations at evolutionarily permissive peripheral positions**, where
temperature-coupled acquisition of glutamate/lysine builds a stabilizing peripheral electrostatic
(salt-bridge) network.

The pipeline (i) builds a **domain-verified** dataset of glycoside hydrolase sequences (Pfam profile
HMMs), (ii) predicts **optimal growth temperature (OGT)** and **experimental melting temperature**
from **ESM2** embeddings under **leakage-controlled (homology-aware) cross-validation**, (iii)
identifies a **position-specific charge-adaptation signature** and maps it to real structures
(PyMOL/FreeSASA, FoldX ΔΔG), and (iv) shows the mechanism generalizes across GH1/GH5/GH10
(TIM-barrel clan GH-A) but not the GH11 jelly-roll fold.

## Repository layout

```
GH5-thermostability/
├── scripts/          # all 60 pipeline scripts (01–55) — run from the repository root
├── data_public/      # clean public dataset: metadata / hotspots / structures / FoldX_results CSVs
├── ai_training/      # curated datasets, per-analysis result CSVs, key PDB structures
├── data_family/      # GH1/GH10/GH11 sets for the cross-family generalization test
├── data_raw/         # TEMPURA organism growth-temperature table
├── pfam/             # Pfam HMMs (PF00150, PF00232, PF00331, PF00457)
├── foldx_run/        # FoldX inputs (individual_list*.txt, shell scripts)
├── manuscript/       # final figures, Source_Data.xlsx, Data_All_for_Review.xlsx
├── environment.yml   # conda environment (pinned)
├── requirements.txt  # pip requirements (pinned)
├── README.md
└── LICENSE
```

## Pipeline (`scripts/`, in order)

All scripts live in `scripts/` and read/write data by paths relative to the repository root, so
run them from the repo root (e.g. `python scripts/26_esm2_ogt_regression.py`).

| Stage | Scripts |
|-------|---------|
| Dataset build + domain verification (PF00150, HMMER gathering cutoff) | `01_*`, `01b_verify_pf00150_gh5.py`, `01c_fetch_thermophilic_gh5.py` |
| Thermostability labels (TEMPURA OGT, BRENDA Topt) | `01d_attach_ogt_labels.py`, `24*`, `27_fetch_brenda_topt_api.py`, `28_build_topt_labels.py` |
| Features, clustering (CD-HIT), non-leaking CV | `02_*`, `13_*`, `14_*`, `15_*`, `16_*` |
| ESM2 regression / classification | `25_train_ogt_regression.py`, `26_esm2_ogt_regression.py`, `29_topt_classifier_esm2.py`, `30_embed_all_757.py` |
| Explainability, UMAP, error analysis, ranking | `31_seq_feature_comparison.py`, `32_xai_shap.py`, `33_umap_ranking_error.py` |
| Structural mapping + salt-bridge survey (PDB, FreeSASA) | `34_structural_mapping.py`, `35_activesite_domain.py`, `38_saltbridge_survey.py` |
| External validation, PLM benchmark, SOTA baselines, ablation | `36_external_validation.py`, `37_plm_benchmark.py`, `39_sota_baselines.py`, `40_ablation.py` |
| Position-specific discovery + engineering targets | `41_biological_discovery.py`, `42_hotspot_engineering.py`, `44_figure14_nature.py` |
| ΔΔG (ESM zero-shot + FoldX single & combinatorial) | `43_ddg_zeroshot.py`, `45_generate_combo_mutations.py`, `foldx_run/` |
| Cross-family generalization (GH1/GH10/GH11) | `46_family_pipeline.py`, `46b_enrich_families.py`, `47_generalization.py` |
| External Tm (Meltome/FLIP), evolutionary constraint | `48_meltome_benchmark.py`, `49_evolutionary_constraint.py` |
| Insert-aware structural correction (canonical) + manuscript/figures | `50_corrected_structural.py`, `51_update_manuscript.py`, `52_restyle_figures.py`, `53_fig18_signature.py` |
| Consolidate all curated data + reported results into one reviewer workbook | `54_export_data_workbook.py` |
| Clean public dataset (metadata, hotspots, structures, FoldX) | `55_export_public_data.py` → `data_public/` |

> **Note on the mapping correction.** `50_corrected_structural.py` supersedes the alignment→structure
> mapping used in scripts 41/42/44/45: it is *insert-aware* (`hmmalign(trim=False)`, advancing the
> residue pointer on every aligned letter and mapping only at match columns). All structural numbers,
> hotspot positions, engineering mutations and FoldX results in the manuscript use this corrected map.
> Catalytic glutamates were verified geometrically (retaining pair, Oε–Oε ≈ 5 Å): 3AMC Glu136/Glu253,
> 3PZT Glu143/Glu186.

## Data included

- `ai_training/gh5_all_verified.csv` — 757 domain-verified GH5 sequences.
- `ai_training/gh5_ogt_labeled.csv`, `gh5_topt_labeled.csv` — OGT (TEMPURA) and enzyme Topt (BRENDA) labels.
- `ai_training/analysis/` — all reported statistics (hotspots, salt bridges, ΔΔG single/combo, benchmarks, ablation, generalization, constraint).
- `data_family/` — GH1/GH10/GH11 labeled sets for generalization; `data_raw/tempura.csv` — TEMPURA OGT.
- `pfam/` — Pfam HMMs (PF00150, PF00232, PF00331, PF00457).
- `ai_training/structures/3AMC.pdb`, `3PZT.pdb`, `5X3A.pdb` — key structures for the figures.
- `manuscript/figures/` — final manuscript figures (PNG).
- `data_public/` — **clean public dataset** (`metadata.csv`, `hotspots.csv`, `structures.csv`, `FoldX_results.csv`) with a schema README; regenerated by `55_export_public_data.py`.
- `manuscript/Data_All_for_Review.xlsx` — **all curated data and reported results in one workbook** (34 sheets + an index), regenerated by `54_export_data_workbook.py` for reviewer access.
- `ai_training/structures/pdb_org.json` — PDB→organism map for the 82-structure salt-bridge survey (`38_saltbridge_survey.py`).

**Not shipped (regenerated by the scripts; excluded via `.gitignore`):** ESM2 embedding caches
(`*.npy`), trained model pickles, the full 82-structure PDB survey set, FoldX repaired PDBs and the
FoldX binary, and raw Meltome/TemStaPro downloads. See each script header for how to regenerate.

## Reproduce

Set up the environment (conda or pip), then run scripts **from the repository root**:

```bash
conda env create -f environment.yml && conda activate gh5-thermostability   # or: pip install -r requirements.txt
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS: pip-torch + conda libomp clash

python scripts/01b_verify_pf00150_gh5.py        # domain verification
python scripts/26_esm2_ogt_regression.py --model 650M   # OGT regression (ESM2-650M)
python scripts/41_biological_discovery.py       # 66-position charge signature
python scripts/50_corrected_structural.py       # insert-aware hotspot mapping + engineering targets
python scripts/47_generalization.py             # GH1/GH10/GH11 generalization
```

FoldX 5.1 (proprietary) and a local PyMOL install are required for the ΔΔG and structural-render steps
(`foldx_run/`, `53_fig18_signature.py`); the FoldX inputs (`individual_list*.txt`) and shell scripts
are provided.

## External resources
NCBI Protein & UniProt (sequence retrieval); Pfam/InterPro (domain HMMs); TEMPURA (organism growth
temperatures); BRENDA (enzyme temperature optima, via authenticated SOAP — credentials read from
`BRENDA_EMAIL`/`BRENDA_PASSWORD` env vars, never stored); RCSB PDB (structures); Meltome Atlas / FLIP
(experimental Tm); FoldX 5.1 (ΔΔG); ESM2 (Lin et al. 2023); PyMOL (renders).

## License
Code: MIT (see `LICENSE`). Curated data derived from public databases retain their original terms.
