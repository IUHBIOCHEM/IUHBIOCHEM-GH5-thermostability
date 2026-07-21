#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
23_fetch_uniprot_keywords_for_wp.py  (FINAL, SIMPLIFIED, ROBUST)

Purpose:
- Map RefSeq protein accessions (WP_/NP_/YP_/XP_) → UniProtKB
- Flag thermophilic / thermostable keywords from UniProt annotations

Input:
- ai_training/gh5_training_data_with_features.csv   (column: Accession)

Output:
- ai_training/uniprot_mapped_keywords.csv
"""

from pathlib import Path
import time
import re
from io import StringIO
import requests
import pandas as pd

# ================= PATHS =================
BASE = Path(__file__).resolve().parent.parent
AI = BASE / "ai_training"
INP = AI / "gh5_training_data_with_features.csv"
OUT = AI / "uniprot_mapped_keywords.csv"

THERMO_PAT = re.compile(
    r"\b(thermostable|thermophil|thermophile|hyperthermophil|heat[- ]stable)\b",
    re.I
)

def norm_acc(x):
    return re.sub(r"\.\d+$", "", str(x).strip())

def run_idmapping(ids):
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": "RefSeq_Protein",
        "to": "UniProtKB",
        "ids": ",".join(ids)
    }
    r = requests.post(url, data=data, timeout=120)
    r.raise_for_status()
    return r.json()["jobId"]

def wait_job(job_id):
    url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("jobStatus") == "FINISHED":
            return
        time.sleep(3)

def fetch_results(job_id):
    url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
    params = {
        "format": "tsv",
        "fields": "accession,protein_name,organism_name,cc_function"
    }
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    return r.text.strip()

def main():
    print("[INFO] Loading accessions ...")
    df = pd.read_csv(INP)
    df["Accession"] = df["Accession"].apply(norm_acc)

    refseq_ids = sorted([
        x for x in df["Accession"].unique()
        if x.startswith(("WP_", "NP_", "YP_", "XP_"))
    ])

    print("[INFO] RefSeq IDs:", len(refseq_ids))
    print("[INFO] Example:", refseq_ids[:5])

    job = run_idmapping(refseq_ids)
    print("[INFO] UniProt jobId:", job)

    wait_job(job)
    txt = fetch_results(job)

    if not txt:
        print("[WARN] UniProt returned EMPTY result file")
        pd.DataFrame(
            columns=["Accession_norm", "UniProt_ThermoKeyword"]
        ).to_csv(OUT, index=False)
        return

    m = pd.read_csv(StringIO(txt), sep="\t")

    if "From" not in m.columns:
        print("[ERROR] UniProt mapping TSV missing 'From' column")
        print("Columns:", m.columns.tolist())
        return

    m.rename(columns={"From": "Accession_norm"}, inplace=True)
    m["Accession_norm"] = m["Accession_norm"].apply(norm_acc)

    scan_cols = [c for c in m.columns if c != "Accession_norm"]

    def thermo_hit(row):
        text = " ".join(str(row[c]) for c in scan_cols)
        return 1 if THERMO_PAT.search(text) else 0

    m["UniProt_ThermoKeyword"] = m.apply(thermo_hit, axis=1)

    out = (
        m.groupby("Accession_norm", as_index=False)
         .agg(UniProt_ThermoKeyword=("UniProt_ThermoKeyword", "max"))
    )

    out.to_csv(OUT, index=False)

    print("✅ Saved:", OUT)
    print("[INFO] Rows:", len(out))
    print("[INFO] Thermo keyword hits:", out["UniProt_ThermoKeyword"].sum())

if __name__ == "__main__":
    main()
