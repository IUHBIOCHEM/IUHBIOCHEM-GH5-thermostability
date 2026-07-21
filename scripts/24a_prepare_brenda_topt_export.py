#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
24a_prepare_brenda_topt_export.py

Chuyển bảng BRENDA (Temperature Optimum) thành file sạch dạng dài cho hồi quy Topt:

Output:
- ai_training/brenda_topt_export.csv  (Accession, Organism, Topt)

Hỗ trợ 2 dạng đầu vào:
  (A) BRENDA "flat file" xuất trực tiếp: TSV KHÔNG header, các cột theo vị trí
      EC | Topt | (max) | Commentary | Organism | UniProt | Literature
      -> cột UniProt có thể chứa nhiều ID phân tách bởi dấu phẩy (đã lặp) -> tách & khử trùng lặp.
  (B) Bảng CÓ header (csv/tsv/xlsx) -> tự dò cột Accession và Temperature Optimum.

Usage:
  python 24a_prepare_brenda_topt_export.py --in_file "ai_training/brenda_topt_export.csv"
"""

from pathlib import Path
from io import StringIO
import re
import argparse
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
AI = BASE / "ai_training"
OUT = AI / "brenda_topt_export.csv"

EC_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
# UniProt accession (Swiss-Prot/TrEMBL), gồm cả dạng A0A...
UNIPROT_RE = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}")


def norm_acc(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)  # bỏ version .1
    return s


def extract_temp_c(val):
    """Lấy nhiệt độ °C từ chuỗi như '70 °C', '343 K', '60-70'."""
    if pd.isna(val):
        return None
    s = str(val)
    mK = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*K\b", s, re.I)
    if mK:
        return float(mK.group(1)) - 273.15
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    x = float(m.group(1))
    if x > 150:  # heuristic Kelvin
        return x - 273.15
    return x


def read_raw_lines(path: Path):
    """Đọc file text với encoding an toàn (BRENDA hay có ký tự ° latin-1)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc).splitlines(), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace").splitlines(), "latin-1(replace)"


def is_brenda_flatfile(lines) -> bool:
    for ln in lines:
        if ln.strip():
            first = ln.split("\t")[0].strip()
            return bool(EC_RE.match(first))
    return False


def parse_brenda_flatfile(lines) -> pd.DataFrame:
    """
    Parse BRENDA flat TSV không header theo vị trí cột.
    Trả về long-format: mỗi UniProt ID một dòng; nếu không có ID, giữ dòng với Accession rỗng
    để vẫn có thể join theo Organism.
    """
    rows = []
    for ln in lines:
        if not ln.strip():
            continue
        cols = ln.split("\t")
        if len(cols) < 5 or not EC_RE.match(cols[0].strip()):
            continue
        topt = extract_temp_c(cols[1])
        organism = cols[4].strip() if len(cols) > 4 else ""
        uni_field = cols[5] if len(cols) > 5 else ""
        # Tách UniProt ID, khử trùng lặp, bỏ '-'
        ids = sorted({m.group(0) for m in UNIPROT_RE.finditer(uni_field)})
        if ids:
            for acc in ids:
                rows.append({"Accession": acc, "Organism": organism, "Topt": topt})
        else:
            rows.append({"Accession": "", "Organism": organism, "Topt": topt})
    return pd.DataFrame(rows)


def read_headered_table(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    seps = [",", "\t", ";"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    raw = path.read_bytes().decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(raw), sep=None, engine="python")


def find_col(df, patterns):
    for p in patterns:
        for c in df.columns:
            if re.search(p, str(c), re.I):
                return c
    return None


def parse_headered(df: pd.DataFrame) -> pd.DataFrame:
    acc_col = find_col(df, [r"\baccession\b", r"\buniprot\b", r"\bprotein\s*id\b", r"\bentry\b", r"^id$"])
    org_col = find_col(df, [r"\borganism\b", r"\bspecies\b", r"\bsource\b"])
    temp_col = find_col(df, [r"temperature.*opt", r"temp.*opt", r"\btopt\b", r"optimum\s*temperature"])
    if temp_col is None:
        raise RuntimeError("Không dò được cột Temperature Optimum. Cho biết tên cột chính xác trong header.")
    out = pd.DataFrame()
    out["Accession"] = df[acc_col].apply(norm_acc) if acc_col else ""
    out["Organism"] = df[org_col].astype(str).str.strip() if org_col else ""
    out["Topt"] = df[temp_col].apply(extract_temp_c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="BRENDA table (flat TSV không header hoặc csv/tsv/xlsx có header)")
    args = ap.parse_args()

    in_file = Path(args.in_file)
    if not in_file.exists():
        raise FileNotFoundError(in_file)

    if in_file.suffix.lower() in (".xlsx", ".xls"):
        out = parse_headered(pd.read_excel(in_file))
    else:
        lines, enc = read_raw_lines(in_file)
        print(f"[INFO] Đọc {in_file.name} (encoding={enc}), {len(lines)} dòng")
        if is_brenda_flatfile(lines):
            print("[INFO] Nhận dạng: BRENDA flat file (TSV không header) -> parse theo vị trí cột")
            out = parse_brenda_flatfile(lines)
        else:
            print("[INFO] Nhận dạng: bảng có header -> dò cột")
            out = parse_headered(read_headered_table(in_file))

    out["Accession"] = out["Accession"].apply(lambda x: norm_acc(x) if str(x).strip() else "")
    out["Topt"] = pd.to_numeric(out["Topt"], errors="coerce")
    out = out.dropna(subset=["Topt"])
    out = out[(out["Topt"] > 0) & (out["Topt"] < 130)]
    out = out.drop_duplicates().reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    n_acc = (out["Accession"].astype(str).str.len() > 0).sum()
    print("✅ Saved:", OUT)
    print(f"[INFO] Rows: {len(out)} | có Accession: {n_acc} | có Organism: {(out['Organism'].astype(str).str.len()>0).sum()}")
    print(f"[INFO] Topt: min={out['Topt'].min()}, max={out['Topt'].max()}, "
          f"n_unique={out['Topt'].nunique()}, std={out['Topt'].std():.2f}")
    if out["Topt"].nunique() <= 1:
        print("\n[CẢNH BÁO] Topt chỉ có 1 giá trị duy nhất -> KHÔNG có phương sai, không thể hồi quy.")
        print("           Cần export lại BRENDA với ĐẦY ĐỦ dải nhiệt độ (không lọc theo 1 mức).")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
