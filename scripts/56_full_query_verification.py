"""56_full_query_verification.py — audit the complete NCBI query by domain verification.

Downloads every record returned by the GH5 retrieval query (no retrieval cap) and applies the
PF00150 profile HMM at the family gathering cutoff, quantifying how many keyword hits are genuine
GH5 domains. Reproduces the full-query verification reported in the manuscript and Supplementary
Fig. 15. Run from the repository root.

Output: ai_training/analysis/full_query_verification.csv
"""
import os
import re
import time
import csv
import pyhmmer
from Bio import Entrez

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "_scratch"))
os.makedirs(SCR, exist_ok=True)
Entrez.email = os.environ.get("NCBI_EMAIL", "")  # set NCBI_EMAIL before running

QUERY = ("(cellulase[Protein Name] OR endoglucanase[Protein Name] OR "
         "\"glycoside hydrolase family 5\"[Protein Name] OR "
         "\"glycosyl hydrolase family 5\"[Protein Name]) AND ("
         "3.2.1.4[ECNO] OR \"glycoside hydrolase family 5\"[Title] OR "
         "\"glycosyl hydrolase family 5\"[Title] OR GH5[Title])")

FASTA = os.path.join(SCR, "full_gh5_query.fasta")


def fetch_all():
    h = Entrez.esearch(db="protein", term=QUERY, retmax=0, usehistory="y")
    r = Entrez.read(h)
    n, webenv, qk = int(r["Count"]), r["WebEnv"], r["QueryKey"]
    print(f"query returns {n} records; downloading...", flush=True)
    with open(FASTA, "w") as out:
        for start in range(0, n, 500):
            for _ in range(3):
                try:
                    fh = Entrez.efetch(db="protein", rettype="fasta", retmode="text",
                                       retstart=start, retmax=500, webenv=webenv, query_key=qk)
                    out.write(fh.read()); fh.close(); break
                except Exception:
                    time.sleep(3)
            time.sleep(0.4)
    return n


def verify():
    alpha = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(os.path.join(ROOT, "pfam", "PF00150.hmm")) as hf:
        hmm = hf.read()
    with pyhmmer.easel.SequenceFile(FASTA, digital=True, alphabet=alpha) as sf:
        seqs = list(sf)
    dec = lambda x: x.decode() if isinstance(x, bytes) else x
    passed = set()
    for top in pyhmmer.hmmsearch([hmm], seqs, bit_cutoffs="gathering"):
        for hit in top:
            if hit.included:
                passed.add(dec(hit.name))
    desc = {}
    for rec in open(FASTA).read().split(">"):
        if rec.strip():
            hdr = rec.split("\n", 1)[0]
            desc[hdr.split()[0]] = hdr
    tot, npass = len(seqs), len(passed)
    ann = [a for a, h in desc.items() if re.search(r"glycoside hydrolase family 5|GH5", h, re.I)]
    ann_pass = sum(1 for a in ann if a in passed)
    rows = [["stage", "count"],
            ["full query", tot],
            ["passed PF00150 (true GH5)", npass],
            ["failed (not GH5)", tot - npass],
            ["annotated 'glycoside hydrolase family 5'", len(ann)],
            ["  of these passed", ann_pass],
            ["  of these failed", len(ann) - ann_pass]]
    with open(os.path.join(ROOT, "ai_training", "analysis", "full_query_verification.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"{tot} candidates -> {npass} verified GH5 ({100*npass/tot:.1f}%); "
          f"annotated-GH5 {ann_pass}/{len(ann)} pass ({100*ann_pass/max(len(ann),1):.0f}%)")


if __name__ == "__main__":
    if not os.path.exists(FASTA):
        fetch_all()
    verify()
