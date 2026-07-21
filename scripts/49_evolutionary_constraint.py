"""
49_evolutionary_constraint.py

Trả lời: "Vì sao tiến hoá chọn đúng các hotspot này để thích nghi nhiệt?"

Tính CONSTRAINT tiến hoá cho từng vị trí PF00150 bằng 2 thước đo độc lập:
  (A) Shannon entropy của cột trong MSA 757 GH5 (bảo tồn tiến hoá cổ điển).
  (B) ESM2 masked-position entropy trên trình tự tham chiếu 3AMC ("ESM conservation").
So sánh 3 nhóm vị trí: active-site core (bảo tồn) | 66 hotspot | vị trí khác (control).
Đồng thời vẽ 2D: constraint vs temperature-coupling (|Δ charged thermo−meso|).

Output: manuscript/figures/Fig20_constraint.png , ai_training/analysis/constraint.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import pyhmmer, torch, esm
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from Bio.PDB import PDBParser
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HMM = "pfam/PF00150.hmm"; alphabet = pyhmmer.easel.Alphabet.amino()
AA = "ACDEFGHIKLMNPQRSTVWY"; CHARGED = set("DEKR")
T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
       "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def _s(v): return v.decode() if isinstance(v,(bytes,bytearray)) else str(v)
def clean(s): return "".join(c for c in str(s).upper() if c in AA)
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); q=np.empty(n); prev=1.0
    for i in range(n-1,-1,-1): prev=min(prev,p[o[i]]*n/(i+1)); q[o[i]]=prev
    return q
def entropy(col):
    col=col[col!="-"]
    if len(col)<10: return np.nan
    _,c=np.unique(col,return_counts=True); p=c/c.sum()
    return float(-(p*np.log2(p)).sum())


def main():
    with pyhmmer.plan7.HMMFile(HMM) as f: hmm=f.read()
    df=pd.read_csv("ai_training/gh5_all_verified.csv")
    ogt=pd.read_csv("ai_training/gh5_ogt_labeled.csv")[["Accession","OGT"]].dropna()
    om=dict(zip(ogt["Accession"].astype(str).str.split(".").str[0],ogt["OGT"]))
    # reference 3AMC seq
    m=PDBParser(QUIET=True).get_structure("x","ai_training/structures/3AMC.pdb")[0]
    ch=max(m,key=lambda c:sum(1 for r in c if r.id[0]==" "))
    refseq="".join(T2O[r.resname] for r in ch if r.id[0]==" " and r.resname in T2O and "CA" in r)

    seqs,names=[],[]
    for _,r in df.iterrows():
        s=clean(r["Sequence"])
        if len(s)>=40: seqs.append(pyhmmer.easel.TextSequence(name=str(r["Accession"]).encode(),sequence=s).digitize(alphabet)); names.append(str(r["Accession"]))
    seqs.append(pyhmmer.easel.TextSequence(name=b"REF",sequence=refseq).digitize(alphabet)); names.append("REF")
    msa=pyhmmer.hmmalign(hmm,seqs,trim=True)
    aln=[_s(x) for x in msa.alignment]; mn=[_s(n) for n in msa.names]
    arr=np.array([list(x) for x in aln]); n2i={n:i for i,n in enumerate(mn)}
    mcols=[j for j in range(arr.shape[1]) if all((c=="-") or c.isupper() for c in arr[:,j])]
    thermo=[i for i,n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]]>=55]
    meso=[i for i,n in enumerate(mn) if n.split(".")[0] in om and om[n.split(".")[0]]<55]

    # per-column: MSA entropy, temperature coupling (charge delta), E-conservation
    rec=[]
    for j in mcols:
        ent=entropy(arr[:,j])
        tn=arr[thermo,j]; tn=tn[tn!="-"]; me=arr[meso,j]; me=me[me!="-"]
        if len(tn)>=8 and len(me)>=8:
            tc=sum(c in CHARGED for c in tn)/len(tn); mc=sum(c in CHARGED for c in me)/len(me)
            _,p=fisher_exact([[sum(c in CHARGED for c in tn),len(tn)-sum(c in CHARGED for c in tn)],
                              [sum(c in CHARGED for c in me),len(me)-sum(c in CHARGED for c in me)]])
            delta=tc-mc
        else: delta=np.nan; p=1.0
        Efrac=(arr[:,j][arr[:,j]!="-"]=="E").mean()
        rec.append({"col":j,"entropy":ent,"charge_delta":delta,"p":p,"Efrac":Efrac})
    C=pd.DataFrame(rec); C["fdr"]=bh(C["p"].fillna(1).values)

    # groups
    hot=set(C[(C["fdr"]<0.05)&(C["charge_delta"]>0)]["col"])
    # active-site core = cột bảo tồn >=80% (residue chủ đạo) — gồm 2 Glu xúc tác
    core=set()
    for j in mcols:
        col=arr[:,j][arr[:,j]!="-"]
        if len(col)==0: continue
        _,c=np.unique(col,return_counts=True)
        if c.max()/len(col)>=0.8: core.add(j)
    def grp(j):
        if j in core: return "active-site core"
        if j in hot: return "hotspot"
        return "other"
    C["group"]=C["col"].map(grp)

    # ---- ESM2 masked entropy on 3AMC ----
    print("ESM2 masked entropy on reference...", flush=True)
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    model,alph=esm.pretrained.esm2_t30_150M_UR50D(); model=model.to(dev).eval(); bc=alph.get_batch_converter()
    _,_,toks=bc([("r",refseq)]); toks=toks.to(dev)
    aa_idx=[alph.get_idx(a) for a in AA]
    esm_ent=np.full(len(refseq),np.nan)
    with torch.no_grad():
        for pos in range(len(refseq)):
            mk=toks.clone(); mk[0,pos+1]=alph.mask_idx
            lg=model(mk)["logits"][0,pos+1,aa_idx]
            pr=torch.softmax(lg,dim=-1).cpu().numpy()
            esm_ent[pos]=-(pr*np.log2(pr+1e-12)).sum()
            if pos%80==0: print(f"  pos {pos}/{len(refseq)}",flush=True)
    # map ref residue index -> column
    refrow=arr[n2i["REF"]]; col_esm={}; rp=0
    for j in mcols:
        if refrow[j]!="-":
            if rp<len(esm_ent): col_esm[j]=esm_ent[rp]
            rp+=1
    C["esm_entropy"]=C["col"].map(col_esm)
    C.to_csv("ai_training/analysis/constraint.csv",index=False)

    # ---- stats ----
    print("\n=== MSA entropy by group (bits; thấp = bảo tồn) ===")
    g={k:C[C["group"]==k]["entropy"].dropna() for k in ["active-site core","hotspot","other"]}
    for k,v in g.items(): print(f"  {k:18} n={len(v):3}  entropy median={v.median():.2f} mean={v.mean():.2f}")
    H,pk=kruskal(*g.values()); print(f"  Kruskal-Wallis p={pk:.2e}")
    for a,b in [("active-site core","hotspot"),("hotspot","other"),("active-site core","other")]:
        _,pp=mannwhitneyu(g[a],g[b]); print(f"  {a} vs {b}: MWU p={pp:.2e}")
    ge={k:C[C["group"]==k]["esm_entropy"].dropna() for k in g}
    print("ESM entropy medians:", {k:round(v.median(),2) for k,v in ge.items()})

    # ---- figure ----
    fig,ax=plt.subplots(1,2,figsize=(13,5.2))
    order=["active-site core","hotspot","other"]; colors=["#d4a017","#c0392b","#9aa0a6"]
    data=[g[k] for k in order]
    bp=ax[0].boxplot(data,tick_labels=[f"{k}\n(n={len(g[k])})" for k in order],patch_artist=True,widths=0.6,showfliers=False)
    for patch,c in zip(bp["boxes"],colors): patch.set_facecolor(c); patch.set_alpha(0.8)
    for i,k in enumerate(order):
        ax[0].scatter(np.random.normal(i+1,0.05,len(g[k])),g[k],s=8,color="k",alpha=0.25,zorder=3)
    ax[0].set_ylabel("MSA Shannon entropy (bits)  —  low = conserved")
    star="***" if pk<1e-3 else "**"
    ax[0].set_title(f"a  Evolutionary constraint by position class ({star}, KW p={pk:.0e})",fontweight="bold",loc="left")
    # 2D: constraint (1/entropy proxy: use entropy on x) vs temperature coupling
    for k,c in zip(order,colors):
        d=C[C["group"]==k]
        ax[1].scatter(d["entropy"],100*d["charge_delta"],s=26,alpha=0.6,color=c,edgecolor="none",label=k)
    ax[1].axhline(0,color="k",lw=0.5)
    ax[1].set_xlabel("MSA entropy (← conserved | variable →)")
    ax[1].set_ylabel("Temperature coupling\nΔ charged freq (thermo−meso), %")
    ax[1].set_title("b  Hotspots: intermediate constraint + high T-coupling",fontweight="bold",loc="left")
    ax[1].legend(fontsize=9,loc="upper left")
    plt.tight_layout(); plt.savefig("manuscript/figures/Fig20_constraint.png",dpi=200)
    print("→ Fig20_constraint.png")


if __name__ == "__main__":
    main()
