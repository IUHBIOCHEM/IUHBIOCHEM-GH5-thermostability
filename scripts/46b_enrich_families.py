"""
46b_enrich_families.py <PF> <GH>
Làm giàu thermophile cho họ CAZy: query UniProt theo Pfam AND các chi ưa nhiệt,
verify domain, gán OGT, gộp vào data_family/<GH>_labeled.csv (khử trùng lặp).
"""
import os, sys, urllib.request, urllib.parse, io, re
import pandas as pd
import pyhmmer

alphabet = pyhmmer.easel.Alphabet.amino()
THERMO = ["Thermotoga", "Pseudothermotoga", "Fervidobacterium", "Caldicellulosiruptor", "Acetivibrio",
          "Thermoanaerobacter", "Thermoanaerobacterium", "Geobacillus", "Parageobacillus", "Anoxybacillus",
          "Thermus", "Rhodothermus", "Dictyoglomus", "Caldanaerobacter", "Thermobifida", "Thermomonospora",
          "Caldibacillus", "Thermoclostridium", "Pyrococcus", "Thermococcus", "Sulfolobus", "Caldicoprobacter",
          "Herbinix", "Caldalkalibacillus", "Thermosipho", "Cohnella"]


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def fetch(pf):
    orgq = " OR ".join(f'organism_name:{g}' for g in THERMO)
    q = f"(xref:pfam-{pf}) AND ({orgq})"
    params = {"query": q, "format": "tsv", "fields": "accession,organism_name,protein_name,sequence", "size": "500"}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    rows = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "research"})
        resp = urllib.request.urlopen(req, timeout=60)
        rows.append(pd.read_csv(io.StringIO(resp.read().decode()), sep="\t"))
        m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
        url = m.group(1) if m else None
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.rename(columns={"Entry": "Accession", "Protein names": "Description"}).dropna(subset=["Sequence"]).drop_duplicates("Accession")


def verify(df, pf):
    with pyhmmer.plan7.HMMFile(f"pfam/{pf}.hmm") as f:
        hmm = f.read()
    seqs, keep = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 60:
            seqs.append(pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet)); keep.append(i)
    pipe = pyhmmer.plan7.Pipeline(alphabet, background=pyhmmer.plan7.Background(alphabet), bit_cutoffs="gathering")
    hits = pipe.search_hmm(hmm, pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs))
    ok = set(int(h.name.decode() if isinstance(h.name, bytes) else h.name) for h in hits)
    return df.reset_index(drop=True).loc[sorted(ok)].copy()


def label(df):
    t = pd.read_csv("data_raw/tempura.csv")
    sp2 = lambda x: " ".join(str(x).replace("[", "").replace("]", "").split()[:2])
    gen = lambda x: (str(x).replace("[", "").split() or [""])[0]
    t["sp"] = t["genus_and_species"].map(sp2); t["g"] = t["genus"].astype(str)
    spm = t.dropna(subset=["Topt_ave"]).groupby("sp")["Topt_ave"].mean()
    gm = t.dropna(subset=["Topt_ave"]).groupby("g")["Topt_ave"].mean()
    df["sp"] = df["Organism"].map(sp2); df["g"] = df["Organism"].map(gen)
    df["OGT"] = df["sp"].map(spm).fillna(df["g"].map(gm))
    return df.drop(columns=["sp", "g"])


def main():
    pf, gh = sys.argv[1], sys.argv[2]
    raw = fetch(pf)
    print(f"[{gh}] thermophile fetch: {len(raw)}")
    ver = label(verify(raw, pf))
    base = pd.read_csv(f"data_family/{gh}_labeled.csv")
    for c in base.columns:
        if c not in ver.columns:
            ver[c] = None
    comb = pd.concat([base, ver[base.columns]], ignore_index=True).drop_duplicates("Accession")
    comb.to_csv(f"data_family/{gh}_labeled.csv", index=False)
    d = comb.dropna(subset=["OGT"])
    print(f"[{gh}] merged: total={len(comb)}, OGT={len(d)}, thermo(>=55)={(d['OGT']>=55).sum()}")


if __name__ == "__main__":
    main()
