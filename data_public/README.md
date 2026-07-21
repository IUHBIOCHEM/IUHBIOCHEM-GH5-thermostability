# Public dataset — GH5 thermal adaptation

Four clean, self-describing tables underlying the study. Regenerate with
`python scripts/55_export_public_data.py` (run from the repository root). Raw sequences and
structures remain available from their primary databases (NCBI/UniProt, RCSB PDB); organism
growth temperatures from TEMPURA; enzyme optima from BRENDA.

### `metadata.csv` — one row per sequence (757)
| column | description |
|--------|-------------|
| `accession` | NCBI protein accession |
| `species` | source organism |
| `OGT` | organism optimal growth temperature (°C; blank if unlabeled) |
| `class` | `thermophile` (OGT ≥ 55 °C) / `mesophile` (< 55 °C); blank if unlabeled |
| `sequence_length` | length of the protein sequence (aa) |
| `cluster` | CD-HIT (70 % identity) cluster id used for homology-aware cross-validation |
| `predicted_OGT` | ESM2-predicted OGT (°C) |
| `P_thermostable` | predicted probability of thermostability |

### `hotspots.csv` — 66 charge-adaptation hotspot positions
| column | description |
|--------|-------------|
| `alignment_position` | column in the PF00150 profile alignment |
| `residue_number_3AMC` / `residue_3AMC` | residue number and amino acid in the reference thermophile 3AMC (blank if unmapped) |
| `thermophile_preferred_residue` | residue most enriched in thermophiles |
| `thermo_charged_pct` / `meso_charged_pct` | % of thermophiles / mesophiles with a charged residue here |
| `enrichment_delta_pct` | thermophile − mesophile charged-residue difference (percentage points) |
| `p_value` / `FDR` | Fisher exact test p-value and Benjamini–Hochberg FDR |
| `location` | `surface` / `buried` / `unmapped` |
| `relative_solvent_accessibility` | RSA in 3AMC |
| `distance_to_catalytic_Glu_A` | minimum distance to the catalytic glutamates (Å) |

### `structures.csv` — 82 experimentally solved GH5 structures
| column | description |
|--------|-------------|
| `pdb` | PDB identifier |
| `organism` / `thermal_class` | source organism and thermophile/mesophile class |
| `n_residues` | number of resolved residues |
| `salt_bridges` / `salt_bridges_per_100res` | salt bridges (Barlow–Thornton ion pairs ≤ 4 Å), absolute and normalized |
| `surface_charged_residues` / `surface_charged_per_100res` | surface-exposed charged residues, absolute and normalized |

### `FoldX_results.csv` — FoldX ΔΔG for engineered substitutions
| column | description |
|--------|-------------|
| `mutation` | single substitution or comma-separated combination (in the mesophile 3PZT) |
| `n_mutations` | number of substitutions |
| `foldx_ddG_kcal_per_mol` / `foldx_SD` | FoldX ΔΔG (mean over 5 runs) and its standard deviation |
| `additive_expectation` / `coupling_energy` | sum of single-mutation effects and observed − additive (epistasis), for combinations |
| `verdict` | stabilizing / neutral / destabilizing (single mutations) |
| `ESM2_log_likelihood_ratio` / `family_signal_delta_pct` | ESM-2 zero-shot score and family charge-enrichment signal (single mutations) |
