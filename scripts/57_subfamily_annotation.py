"""57_subfamily_annotation.py — assign GH5 subfamilies to the 757 sequences and test the
charge signature within subfamilies.

Parses the CAZy GH5 "characterized" page for accession->subfamily labels, fetches those reference
sequences, and assigns each of the 757 domain-verified sequences to a subfamily by best DIAMOND
match. Reproduces Supplementary Fig. 16. Run from the repository root. Requires DIAMOND on PATH.

Outputs: ai_training/analysis/cazy_gh5_characterized.csv, ai_training/analysis/gh5_757_subfamily.csv
"""
import os
import re
import csv
import time
import subprocess
import shutil
import pandas as pd
from collections import Counter
from Bio import Entrez
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("SCRATCH_DIR", os.path.join(ROOT, "_scratch"))
os.makedirs(SCR, exist_ok=True)
A = os.path.join(ROOT, "ai_training", "analysis")
Entrez.email = os.environ.get("NCBI_EMAIL", "")
DIAMOND = os.environ.get("DIAMOND") or shutil.which("diamond") or "diamond"


def parse_cazy():
    path = os.path.join(SCR, "GH5_characterized.html")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve("http://www.cazy.org/GH5_characterized.html", path)
    html = open(path).read()
    recs = []
    for r in (x for x in re.findall(r"<tr\b(.*?)</tr>", html, re.S) if "onmouseover" in x):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 8:
            continue
        txt = lambda c: re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()
        gb = re.findall(r"protein/([A-Za-z0-9_.]+)", cells[4]) or re.findall(r">([A-Z][A-Z0-9_.]{4,})<", cells[4])
        sf = txt(cells[7])
        if gb:
            recs.append((gb[0], "GH5_" + sf if sf else "unassigned", txt(cells[6]), txt(cells[3])))
    with open(os.path.join(A, "cazy_gh5_characterized.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["genbank", "subfamily", "pdb", "organism"])
        csv.writer(f).writerows(recs)
    return recs


def fetch(accs, path):
    with open(path, "w") as out:
        for i in range(0, len(accs), 200):
            for _ in range(3):
                try:
                    h = Entrez.efetch(db="protein", id=",".join(accs[i:i+200]), rettype="fasta", retmode="text")
                    out.write(h.read()); h.close(); break
                except Exception:
                    time.sleep(3)
            time.sleep(0.5)


def main():
    recs = parse_cazy()
    sub = {a: s for a, s, _, _ in recs}
    # reference FASTA (subfamily in header)
    ref = os.path.join(SCR, "cazy_char_ref.fasta")
    if not os.path.exists(ref):
        fetch(list(sub), os.path.join(SCR, "_char.fasta"))
        with open(ref, "w") as out:
            for block in open(os.path.join(SCR, "_char.fasta")).read().split(">"):
                if not block.strip():
                    continue
                acc = block.split("\n", 1)[0].split()[0]
                base = acc.split(".")[0]
                sf = sub.get(acc) or next((sub[k] for k in sub if k.split(".")[0] == base), "unassigned")
                out.write(f">{acc}|{sf}\n" + "".join(block.split('\n')[1:]) + "\n")
    df = pd.read_csv(os.path.join(ROOT, "ai_training", "gh5_all_verified.csv"))
    q = os.path.join(SCR, "our757.fasta")
    with open(q, "w") as f:
        for r in df.itertuples():
            f.write(f">{r.Accession}\n{r.Sequence}\n")
    subprocess.run([DIAMOND, "makedb", "--in", ref, "-d", os.path.join(SCR, "cazref"), "--quiet"], check=True)
    tsv = os.path.join(SCR, "sub757.tsv")
    subprocess.run([DIAMOND, "blastp", "-q", q, "-d", os.path.join(SCR, "cazref"), "-o", tsv,
                    "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
                    "--max-target-seqs", "1", "--quiet"], check=True)
    d = pd.read_csv(tsv, sep="\t", names=["q", "s", "pident", "evalue", "bits"])
    best = d.sort_values("bits", ascending=False).drop_duplicates("q")
    best["subf"] = best["s"].str.split("|").str[-1]
    best[["q", "subf", "pident", "evalue"]].rename(columns={"q": "Accession"}).to_csv(
        os.path.join(A, "gh5_757_subfamily.csv"), index=False)
    print("subfamilies:", Counter(best.subf).most_common(12))
    # within-subfamily signal
    sig = pd.read_csv(os.path.join(A, "signature_scores.csv")).merge(
        best[["q", "subf"]].rename(columns={"q": "Accession"}), on="Accession", how="left").dropna(subset=["OGT", "subf"])
    for sf, g in sig.groupby("subf"):
        if len(g) >= 15 and g.OGT.std() > 5:
            r = spearmanr(g.signature, g.OGT)
            print(f"  {sf}: rho={r.correlation:+.2f}, P={r.pvalue:.1e}, n={len(g)}")


if __name__ == "__main__":
    main()
