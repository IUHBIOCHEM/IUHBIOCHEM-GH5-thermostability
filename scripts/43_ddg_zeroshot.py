"""
43_ddg_zeroshot.py

Ước lượng tác động ổn định (ΔΔG proxy) của 4 đột biến đề xuất bằng ESM2 zero-shot
(masked-marginal log-likelihood ratio; Meier et al. 2021, NeurIPS/Nature Biotech) —
FoldX/Rosetta không có sẵn ở môi trường này (proprietary).

Quy ước: LLR = log P(mut) − log P(wt) tại vị trí (đã mask). LLR > 0 => mô hình ưu ái
đột biến (dự đoán dung nạp/ổn định); LLR < 0 => bất lợi. Đồng thời tính điểm ủng hộ từ
CHÍNH họ GH5 (tần suất residue đích ở thermophile tại vị trí đó).

Cũng xuất file individual_list.txt định dạng FoldX để người dùng tự chạy BuildModel nếu có license.

Output: ai_training/analysis/ddg_zeroshot.csv , ai_training/analysis/foldx_individual_list.txt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
import esm
from Bio.PDB import PDBParser

T2O = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
       "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def pzt_seq():
    m = PDBParser(QUIET=True).get_structure("x", "ai_training/structures/3PZT.pdb")[0]
    chain = max(m, key=lambda c: sum(1 for r in c if r.id[0] == " "))
    seq, nums = "", []
    for r in chain:
        if r.id[0] == " " and r.resname in T2O and "CA" in r:
            seq += T2O[r.resname]; nums.append(r.id[1])
    return seq, nums, chain.id


def main():
    muts = pd.read_csv("ai_training/analysis/engineering_mutations.csv")
    seq, nums, chain_id = pzt_seq()
    num2idx = {n: i for i, n in enumerate(nums)}

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([("wt", seq)])
    toks = toks.to(dev)

    rows = []
    with torch.no_grad():
        for _, mrow in muts.iterrows():
            mut = mrow["mutation"]  # e.g. Q58K
            wt, resnum, to = mut[0], int(mut[1:-1]), mut[-1]
            if resnum not in num2idx:
                rows.append({"mutation": mut, "note": "resnum not in structure"}); continue
            idx = num2idx[resnum]
            assert seq[idx] == wt, f"{mut}: seq has {seq[idx]} at {resnum}"
            tpos = idx + 1  # +1 cho token BOS
            masked = toks.clone(); masked[0, tpos] = alphabet.mask_idx
            logits = model(masked)["logits"][0, tpos]
            logp = torch.log_softmax(logits, dim=-1)
            llr = (logp[alphabet.get_idx(to)] - logp[alphabet.get_idx(wt)]).item()
            rows.append({"mutation": mut, "wt": wt, "pos_3PZT": resnum, "to": to,
                         "ESM2_LLR": round(llr, 3),
                         "prediction": "favorable" if llr > 0 else "unfavorable",
                         "thermo_charged_%": mrow["thermo_charged_%"], "delta_%": mrow["delta_%"]})
    out = pd.DataFrame(rows)
    out.to_csv("ai_training/analysis/ddg_zeroshot.csv", index=False)
    print(out.to_string(index=False))

    # FoldX individual_list.txt (để người dùng tự chạy nếu có FoldX)
    with open("ai_training/analysis/foldx_individual_list.txt", "w") as f:
        for _, mrow in muts.iterrows():
            mut = mrow["mutation"]; wt, resnum, to = mut[0], mut[1:-1], mut[-1]
            f.write(f"{wt}{chain_id}{resnum}{to};\n")
    print("\nĐã ghi FoldX individual_list.txt (chain", chain_id, ") — chạy: "
          "foldx --command=BuildModel --pdb=3PZT.pdb --mutant-file=individual_list.txt")


if __name__ == "__main__":
    main()
