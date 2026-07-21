"""
27_fetch_brenda_topt_api.py

Kéo Temperature Optimum (Topt enzyme thật) từ BRENDA qua SOAP API cho EC 3.2.1.4
(và các EC GH5 liên quan tùy chọn). Dùng chính tài khoản BRENDA của bạn.

BẢO MẬT: script KHÔNG lưu mật khẩu. Lấy thông tin đăng nhập theo thứ tự:
  1) biến môi trường BRENDA_EMAIL / BRENDA_PASSWORD, nếu thiếu thì
  2) hỏi trực tiếp trên terminal (getpass, không hiện mật khẩu).
Mật khẩu được băm SHA256 theo yêu cầu của BRENDA trước khi gửi.

Cài đặt:  pip install zeep
Chạy:     python 27_fetch_brenda_topt_api.py
          hoặc:  python 27_fetch_brenda_topt_api.py --ec 3.2.1.4 3.2.1.73 3.2.1.78

Output:   ai_training/brenda_topt_api.csv   (EC, Organism, UniProt, Topt, Commentary)
"""
import os
import re
import sys
import hashlib
import getpass
import argparse
import pandas as pd
from zeep import Client

WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
OUT = "ai_training/brenda_topt_api.csv"
# EC mặc định: endoglucanase (cellulase). Có thể thêm EC GH5 khác qua --ec
DEFAULT_ECS = ["3.2.1.4"]


def get_credentials():
    email = os.environ.get("BRENDA_EMAIL") or input("BRENDA email: ").strip()
    pw = os.environ.get("BRENDA_PASSWORD") or getpass.getpass("BRENDA password (ẩn): ")
    pw_hash = hashlib.sha256(pw.encode("utf-8")).hexdigest()
    return email, pw_hash


def parse_temp(val):
    if val is None:
        return None
    s = str(val)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec", nargs="+", default=DEFAULT_ECS, help="Danh sách EC (mặc định 3.2.1.4)")
    args = ap.parse_args()

    email, pw_hash = get_credentials()
    print(f"Kết nối BRENDA API ({email})...", flush=True)
    client = Client(WSDL)

    rows = []
    for ec in args.ec:
        # Đúng chữ ký WSDL: email, password, ecNumber, organism,
        # temperatureOptimum, temperatureOptimumMaximum, commentary, literature
        params = (email, pw_hash, f"ecNumber*{ec}", "organism*",
                  "temperatureOptimum*", "temperatureOptimumMaximum*",
                  "commentary*", "literature*")
        try:
            result = client.service.getTemperatureOptimum(*params)
        except Exception as e:
            print(f"  [LỖI] EC {ec}: {e}")
            continue
        n = len(result) if result else 0
        print(f"  EC {ec}: {n} bản ghi Temperature Optimum")
        for r in (result or []):
            # zeep trả object (không phải dict) -> dùng getattr
            def fld(name):
                return getattr(r, name, None)
            rows.append({
                "EC": ec,
                "Organism": (fld("organism") or "").strip(),
                "Topt": parse_temp(fld("temperatureOptimum")),
                "Topt_max": parse_temp(fld("temperatureOptimumMaximum")),
                "Commentary": (fld("commentary") or "").strip(),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("Không lấy được bản ghi nào. Kiểm tra lại đăng nhập/EC.")
        sys.exit(1)

    df = df.dropna(subset=["Topt"])
    df = df[(df["Topt"] > 0) & (df["Topt"] < 130)]
    os.makedirs("ai_training", exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\n✅ Lưu {len(df)} bản ghi -> {OUT}")
    print(f"Topt: min={df['Topt'].min():.0f}, max={df['Topt'].max():.0f}, "
          f"mean={df['Topt'].mean():.1f}, std={df['Topt'].std():.1f}, "
          f"n_organism={df['Organism'].nunique()}")
    print("\nMẫu:")
    print(df[["Organism", "Topt"]].drop_duplicates().head(12).to_string(index=False))


if __name__ == "__main__":
    main()
