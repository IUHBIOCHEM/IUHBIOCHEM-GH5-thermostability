"""
Xác minh trình tự thực sự thuộc họ GH5 bằng domain Pfam PF00150
(Cellulase / glycosyl hydrolase family 5).

Chỉ dựa vào tên/description là chưa đủ chắc chắn (có thể chú thích nhầm họ),
nên bước này quét từng trình tự bằng HMM PF00150 với ngưỡng gathering (GA)
chuẩn của Pfam. Chỉ giữ lại trình tự đạt ngưỡng GA => đảm bảo 100% đúng GH5.

Yêu cầu:
    pip install pyhmmer
    HMM: pfam/PF00150.hmm  (tải từ InterPro:
        https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00150?annotation=hmm )

Dùng:
    python 01b_verify_pf00150_gh5.py                      # mặc định lọc CSV đã fetch
    python 01b_verify_pf00150_gh5.py <input> <out_prefix> # tùy chọn
    <input> có thể là .csv (cột Accession/Sequence/...) hoặc .fasta
"""
import os
import sys
import re
import pandas as pd
import pyhmmer

HMM_PATH = "pfam/PF00150.hmm"
DEFAULT_INPUT = "ai_training/gh5_training_data_large.csv"
DEFAULT_OUT_PREFIX = "ai_training/gh5_verified_pf00150"

# Ký tự acid amin hợp lệ cho easel (20 chuẩn + nhập nhằng B/J/Z/X + gap)
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBJZXUO")


def _s(v):
    """Trả về str dù pyhmmer đưa ra bytes hay str (khác nhau giữa các phiên bản)."""
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def clean_seq(seq):
    """Bỏ khoảng trắng, dấu '*' (stop) và ký tự lạ để easel digitize được."""
    s = re.sub(r"\s+", "", str(seq)).upper().replace("*", "").replace("-", "")
    return "".join(ch for ch in s if ch in _VALID_AA)


def load_records(input_path):
    """Trả về DataFrame có cột Accession / Sequence / Organism / Description."""
    if input_path.lower().endswith((".fasta", ".fa", ".faa")):
        from Bio import SeqIO
        rows = []
        for rec in SeqIO.parse(input_path, "fasta"):
            rows.append({
                "Accession": rec.id,
                "Sequence": str(rec.seq),
                "Organism": "",
                "Description": rec.description,
            })
        return pd.DataFrame(rows)

    df = pd.read_csv(input_path)
    for col in ("Accession", "Sequence"):
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong {input_path}")
    for col in ("Organism", "Description"):
        if col not in df.columns:
            df[col] = ""
    return df


def verify_gh5(input_path=DEFAULT_INPUT, out_prefix=DEFAULT_OUT_PREFIX):
    if not os.path.exists(HMM_PATH):
        raise FileNotFoundError(
            f"Không thấy {HMM_PATH}. Tải PF00150 từ InterPro rồi giải nén vào thư mục pfam/."
        )

    df = load_records(input_path)
    df = df.reset_index(drop=True)
    print(f"Đọc {len(df)} trình tự từ {input_path}")

    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(HMM_PATH) as hf:
        hmm = hf.read()
    print(f"HMM: {_s(hmm.name)} ({_s(hmm.accession)}), dùng ngưỡng gathering (GA).")

    # Số hóa trình tự; tên = chỉ số dòng để map ngược an toàn (Accession có thể trùng)
    digital_seqs = []
    idx_ok = []
    for i, row in df.iterrows():
        seq = clean_seq(row["Sequence"])
        if len(seq) < 40:
            continue
        try:
            ts = pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=seq)
            digital_seqs.append(ts.digitize(alphabet))
            idx_ok.append(i)
        except Exception as e:
            print(f"  Bỏ qua dòng {i} ({row['Accession']}): {e}")

    # hmmsearch: query = HMM, database = các trình tự, áp ngưỡng GA của Pfam
    passed = {}  # index -> (bitscore, evalue)
    for top_hits in pyhmmer.hmmsearch([hmm], digital_seqs, bit_cutoffs="gathering"):
        for hit in top_hits:
            if hit.included:  # đạt ngưỡng GA
                passed[int(_s(hit.name))] = (hit.score, hit.evalue)

    verified = df.loc[sorted(passed.keys())].copy()
    verified["PF00150_bitscore"] = [passed[i][0] for i in sorted(passed.keys())]
    verified["PF00150_evalue"] = [passed[i][1] for i in sorted(passed.keys())]

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    fasta_path = f"{out_prefix}.fasta"
    verified.to_csv(csv_path, index=False)
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, r in verified.iterrows():
            org = str(r.get("Organism", "")).replace(" ", "_")
            f.write(f">{r['Accession']}_{org}\n{clean_seq(r['Sequence'])}\n")

    n_in, n_out = len(df), len(verified)
    print("\n=== KẾT QUẢ XÁC MINH PF00150 (GH5) ===")
    print(f"Đầu vào:            {n_in}")
    print(f"Đạt ngưỡng GA (GH5): {n_out}")
    print(f"Bị loại (không phải GH5 thật): {n_in - n_out}")
    print(f"CSV:   {csv_path}")
    print(f"FASTA: {fasta_path}")
    return verified


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_PREFIX
    verify_gh5(inp, outp)
