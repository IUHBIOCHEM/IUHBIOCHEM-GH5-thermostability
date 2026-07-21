"""
46_family_pipeline.py <PF> <GH>

Tổng quát hoá khung sang họ CAZy khác. Nguồn: UniProt REST (query theo Pfam, kèm organism)
-> xác minh domain Pfam (HMMER gathering) -> gán nhãn OGT (TEMPURA theo loài).
Lưu data_family/<GH>_labeled.csv

Ví dụ:
  python 46_family_pipeline.py PF00232 GH1
  python 46_family_pipeline.py PF00331 GH10
  python 46_family_pipeline.py PF00457 GH11
"""
import os, sys, urllib.request, urllib.parse, io
import numpy as np
import pandas as pd
import pyhmmer

alphabet = pyhmmer.easel.Alphabet.amino()
os.makedirs("data_family", exist_ok=True)


def clean(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def fetch_uniprot(pf, reviewed_only=True, cap=2000):
    q = f"(xref:pfam-{pf})"
    if reviewed_only:
        q += " AND (reviewed:true)"
    params = {"query": q, "format": "tsv",
              "fields": "accession,organism_name,protein_name,sequence", "size": "500"}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    rows = []
    while url and sum(len(r) for r in rows) < cap:
        req = urllib.request.Request(url, headers={"User-Agent": "research"})
        resp = urllib.request.urlopen(req, timeout=60)
        txt = resp.read().decode()
        df = pd.read_csv(io.StringIO(txt), sep="\t")
        rows.append(df)
        # pagination via Link header (URL chứa dấu phẩy trong 'fields' -> dùng regex)
        import re as _re
        link = resp.headers.get("Link", "")
        m = _re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = out.rename(columns={"Entry": "Accession", "Organism": "Organism",
                              "Protein names": "Description", "Sequence": "Sequence"})
    return out.dropna(subset=["Sequence"]).drop_duplicates("Accession")


def verify(df, pf):
    with pyhmmer.plan7.HMMFile(f"pfam/{pf}.hmm") as f:
        hmm = f.read()
    seqs, keep = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        s = clean(r["Sequence"])
        if len(s) >= 60:
            seqs.append(pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet)); keep.append(i)
    bg = pyhmmer.plan7.Background(alphabet)
    pipe = pyhmmer.plan7.Pipeline(alphabet, background=bg, bit_cutoffs="gathering")
    hits = pipe.search_hmm(hmm, pyhmmer.easel.DigitalSequenceBlock(alphabet, seqs))
    ok = set(int(h.name.decode() if isinstance(h.name, bytes) else h.name) for h in hits)
    return df.reset_index(drop=True).loc[sorted(ok)].copy()


def label_ogt(df):
    t = pd.read_csv("data_raw/tempura.csv")
    def sp2(x):
        p = str(x).replace("[", "").replace("]", "").replace('"', "").split(); return " ".join(p[:2]) if len(p) >= 2 else str(x)
    def gen(x):
        c = str(x).replace("[", "").replace("]", "").strip(); return c.split()[0] if c else ""
    t["sp"] = t["genus_and_species"].map(sp2); t["g"] = t["genus"].astype(str)
    sp = t.dropna(subset=["Topt_ave"]).groupby("sp")["Topt_ave"].mean()
    gm = t.dropna(subset=["Topt_ave"]).groupby("g")["Topt_ave"].mean()
    df["sp"] = df["Organism"].map(sp2); df["g"] = df["Organism"].map(gen)
    df["OGT"] = df["sp"].map(sp).fillna(df["g"].map(gm))
    return df.drop(columns=["sp", "g"])


def main():
    pf, gh = sys.argv[1], sys.argv[2]
    print(f"[{gh}] UniProt fetch (Pfam {pf})...", flush=True)
    raw = fetch_uniprot(pf, reviewed_only=True)
    if len(raw) < 150:  # nếu Swiss-Prot quá ít, thêm TrEMBL
        print(f"[{gh}] reviewed only {len(raw)}, thêm unreviewed...", flush=True)
        raw = fetch_uniprot(pf, reviewed_only=False, cap=1000)
    print(f"[{gh}] fetched {len(raw)}")
    ver = verify(raw, pf)
    lab = label_ogt(ver)
    n = lab["OGT"].notna().sum()
    lab.to_csv(f"data_family/{gh}_labeled.csv", index=False)
    print(f"[{gh}] DONE: verified={len(ver)}, with OGT={n} -> data_family/{gh}_labeled.csv")


if __name__ == "__main__":
    main()
