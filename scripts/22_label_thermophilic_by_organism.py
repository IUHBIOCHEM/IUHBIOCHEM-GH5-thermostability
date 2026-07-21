#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
22_label_thermophilic_by_organism.py

Option A (Q1-friendly):
- Label thermophilic GH5 sequences by organism genus whitelist (thermophile/hyperthermophile genera).
- Adds two columns:
    Is_Thermophilic  (0/1)
    Thermo_Evidence  ("organism_whitelist" or "none")
- Writes:
    ai_training/gh5_training_data_labeled.csv
    ai_training/thermophile_taxa_whitelist.txt

Input:
    ai_training/gh5_training_data_with_features.csv
Expected columns:
    Accession, Organism, Is_Thermophilic_Guess (optional), plus features...

Notes:
- Accessions are normalized by removing version suffix like ".1" to match CD-HIT groups if needed.
- This step does NOT change sequences/features; it only assigns labels from organism metadata.
"""

from pathlib import Path
import re
import pandas as pd


# ------------------ Thermophile genera whitelist (curated, expandable) ------------------
THERMOPHILE_GENERA = [
    # classic bacterial thermophiles / hyperthermophiles frequently used in enzyme studies
    "Thermotoga", "Thermus", "Geobacillus", "Anoxybacillus", "Caldicellulosiruptor",
    "Thermoanaerobacter", "Thermoanaerobacterium", "Thermoanaerobaculum",
    "Thermobifida", "Thermobacillus", "Thermomonospora", "Thermocrispum",
    "Thermococcus", "Pyrococcus", "Sulfolobus", "Thermoproteus", "Pyrobaculum",
    "Aquifex", "Hydrogenobacter", "Thermodesulfobacterium", "Thermodesulfatator",
    "Caldanaerobacter", "Caldanaerobius", "Caldimicrobium", "Caldilinea",
    "Thermodesulfovibrio", "Thermoleophilum", "Thermus", "Meiothermus",
    "Thermoflavimicrobium", "Thermoflavifilum",
    "Thermohalobacter", "Thermohydrogenium",
    "Caldicoprobacter", "Thermovenabulum", "Thermoclostridium",
    "Thermoactinomyces", "Thermogemmatispora",
    "Thermosipho", "Fervidobacterium", "Kosmotoga",
    "Dictyoglomus",
    "Caldimonas", "Thermicanus", "Thermococcus",
    "Thermopetra", "Thermotomaculum",
    # archaeal / extreme thermophiles (may appear in GH contexts less often, but keep for completeness)
    "Methanothermobacter", "Methanobacterium", "Methanocaldococcus", "Methanopyrus",
    "Ignicoccus", "Staphylothermus", "Thermoplasma", "Picrophilus",
    # additional common thermo-associated genera
    "Thermodesulforhabdus", "Thermodesulfobium", "Thermodesulfomicrobium",
    "Caldithrix", "Thermoflexus", "Thermofractor",
]

# Some genera may appear with prefixes/suffixes or strain formatting; we match genus token.
# You can expand this list later if you find missed thermophile genera in your dataset.


def norm_acc(acc: str) -> str:
    """Remove version suffix like '.1' from accessions."""
    return re.sub(r"\.\d+$", "", str(acc).strip())


def extract_genus(organism: str) -> str:
    """
    Extract genus from organism string.
    Examples:
        "Thermotoga maritima" -> "Thermotoga"
        "Geobacillus sp. WSUCF1" -> "Geobacillus"
        "Caldicellulosiruptor bescii DSM 6725" -> "Caldicellulosiruptor"
    """
    if pd.isna(organism):
        return ""
    s = str(organism).strip()
    if not s:
        return ""
    # split on whitespace, first token is usually genus
    genus = s.split()[0]
    # remove punctuation artifacts
    genus = re.sub(r"[^A-Za-z]", "", genus)
    return genus


def main():
    BASE = Path(__file__).resolve().parent.parent
    AI = BASE / "ai_training"
    INP = AI / "gh5_training_data_with_features.csv"
    OUT = AI / "gh5_training_data_labeled.csv"
    WHITELIST_TXT = AI / "thermophile_taxa_whitelist.txt"

    if not INP.exists():
        raise FileNotFoundError(f"Missing input: {INP}")

    df = pd.read_csv(INP)

    if "Accession" not in df.columns or "Organism" not in df.columns:
        raise KeyError("Input CSV must contain columns: Accession, Organism")

    # Normalize accession for consistency downstream
    df["Accession"] = df["Accession"].apply(norm_acc)

    # Extract genus
    df["_Genus"] = df["Organism"].apply(extract_genus)

    # Build whitelist set (case-sensitive genus, but we also compare case-insensitively)
    wl = sorted(set(THERMOPHILE_GENERA))
    wl_set = set(wl)
    wl_lower = set([g.lower() for g in wl])

    # Label by whitelist
    def is_thermo(genus: str) -> int:
        if not genus:
            return 0
        if genus in wl_set:
            return 1
        if genus.lower() in wl_lower:
            return 1
        return 0

    df["Is_Thermophilic"] = df["_Genus"].apply(is_thermo).astype(int)
    df["Thermo_Evidence"] = df["Is_Thermophilic"].apply(lambda v: "organism_whitelist" if v == 1 else "none")

    # Save whitelist for transparency / Methods
    WHITELIST_TXT.write_text("\n".join(wl) + "\n", encoding="utf-8")

    # Save labeled dataset
    df.to_csv(OUT, index=False)

    # Summary
    total = len(df)
    thermo_n = int(df["Is_Thermophilic"].sum())
    meso_n = total - thermo_n

    print("✅ Saved labeled dataset ->", OUT)
    print("✅ Saved whitelist ->", WHITELIST_TXT)
    print("\n=== LABEL SUMMARY ===")
    print("Total:", total)
    print("Thermophilic (1):", thermo_n)
    print("Mesophilic (0):", meso_n)

    if thermo_n > 0:
        print("\nTop thermophile genera in dataset:")
        top = df.loc[df["Is_Thermophilic"] == 1, "_Genus"].value_counts().head(15)
        print(top.to_string())
    else:
        print("\n[WARN] No thermophilic labels assigned by whitelist.")
        print("-> This means none of your organism genera matched the thermophile list.")
        print("-> Next step: expand whitelist based on your dataset genera.")


if __name__ == "__main__":
    main()

