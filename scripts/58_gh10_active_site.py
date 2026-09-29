"""58_gh10_active_site.py — screen the GH10 dataset for catalytic-site integrity.

Aligns the GH10 sequences to the PF00331 profile, identifies the two catalytic glutamate columns
(the two most conserved Glu match states) and flags sequences that lack either, addressing whether
active-site-disrupted members drive the charge pattern. Reproduces Supplementary Table 7.
Run from the repository root.

Output: ai_training/analysis/gh10_disrupted_activesite.csv
"""
import os
import pyhmmer
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "ai_training", "analysis")


def main():
    df = pd.read_csv(os.path.join(ROOT, "data_family", "GH10_labeled.csv"))
    alpha = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(os.path.join(ROOT, "pfam", "PF00331.hmm")) as hf:
        hmm = hf.read()
    seqs = []
    for r in df.itertuples():
        s = str(r.Sequence).replace("*", "").replace("-", "")
        if len(s) >= 50:
            seqs.append((r.Accession, r.Organism,
                         pyhmmer.easel.TextSequence(name=str(r.Accession).encode(), sequence=s).digitize(alpha)))
    msa = pyhmmer.hmmalign(hmm, [s[2] for s in seqs])
    arr = np.array([list(str(a)) for a in msa.alignment])
    L = arr.shape[1]
    mcols = [j for j in range(L) if not (arr[0][j].islower() or arr[0][j] == ".")]
    glu = sorted(((j, np.mean(arr[:, j] == "E")) for j in mcols), key=lambda x: -x[1])
    cat = [glu[0][0], glu[1][0]]  # two most conserved Glu columns = catalytic pair
    disrupted = [(acc, org) for i, (acc, org, _) in enumerate(seqs)
                 if not all(arr[i][c] == "E" for c in cat)]
    intact = len(seqs) - len(disrupted)
    pd.DataFrame(disrupted, columns=["Accession", "Organism"]).to_csv(
        os.path.join(A, "gh10_disrupted_activesite.csv"), index=False)
    print(f"GH10: {intact}/{len(seqs)} intact ({100*intact/len(seqs):.1f}%), "
          f"{len(disrupted)} disrupted; catalytic Glu columns conserved "
          f"{glu[0][1]*100:.0f}% / {glu[1][1]*100:.0f}%")


if __name__ == "__main__":
    main()
