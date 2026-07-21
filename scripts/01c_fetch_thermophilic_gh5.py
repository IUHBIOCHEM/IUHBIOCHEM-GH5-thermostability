"""
01c_fetch_thermophilic_gh5.py

Bổ sung GH5 cellulase từ các CHI ƯA NHIỆT (thermophilic/hyperthermophilic) để tập
huấn luyện có dải nhiệt độ rộng (mục tiêu hồi quy Topt/OGT).

Cùng logic lọc như 01 (giới hạn [Protein Name], loại regulator/hypothetical...),
nhưng thêm ràng buộc [Organism] vào danh sách chi ưa nhiệt đã biết có cellulase
và có mặt trong TEMPURA với OGT cao.

Output: ai_training/gh5_thermophilic_raw.csv
"""
from Bio import Entrez, SeqIO
import pandas as pd
import os
import time

Entrez.email = "hothienhoang@iuh.edu.vn"

# Chi ưa nhiệt/siêu ưa nhiệt có cellulase (GH5), OGT ~55-90°C
THERMOPHILIC_GENERA = [
    "Thermotoga", "Pseudothermotoga", "Fervidobacterium", "Thermosipho",
    "Caldicellulosiruptor", "Anaerocellum",
    "Thermobifida", "Thermomonospora", "Thermopolyspora",
    "Acetivibrio", "Ruminiclostridium", "Hungateiclostridium", "Clostridium thermocellum",
    "Thermoanaerobacter", "Thermoanaerobacterium", "Caldanaerobius", "Caldanaerobacter",
    "Geobacillus", "Parageobacillus", "Anoxybacillus", "Caldibacillus",
    "Thermus", "Rhodothermus", "Dictyoglomus",
    "Herbinix", "Cohnella", "Caldalkalibacillus",
    "Thermoclostridium", "Thermosediminibacter",
]

EXCLUDE_KEYWORDS = [
    "regulator", "transcription", "hypothetical", "binding protein",
    "binding-protein", "uncharacterized", "transporter",
]
INCLUDE_KEYWORDS = [
    "cellulase", "endoglucanase", "endo-1,4-beta-glucanase", "glucanase",
    "glycoside hydrolase family 5", "glycosyl hydrolase family 5",
    "glycoside hydrolase 5", "gh5",
]


def is_valid_gh5_cellulase(description):
    d = (description or "").lower()
    if any(bad in d for bad in EXCLUDE_KEYWORDS):
        return False
    return any(good in d for good in INCLUDE_KEYWORDS)


def fetch_thermophilic(limit_per_query=1000):
    if not os.path.exists('ai_training'):
        os.makedirs('ai_training')

    organism_clause = " OR ".join(f'"{g}"[Organism]' for g in THERMOPHILIC_GENERA)
    query = (
        '('
        'cellulase[Protein Name] OR endoglucanase[Protein Name] OR '
        '"glycoside hydrolase family 5"[Protein Name] OR '
        '"glycosyl hydrolase family 5"[Protein Name]'
        f') AND ({organism_clause})'
    )
    print("Query:", query[:180], "...")

    search_handle = Entrez.esearch(db="protein", term=query, retmax=limit_per_query)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]
    print(f"Tìm thấy {len(id_list)} mã tiềm năng (thermophilic).")

    all_records, skipped = [], 0
    batch_size = 50
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        try:
            fetch_handle = Entrez.efetch(db="protein", id=batch_ids, rettype="gp", retmode="text")
            records = list(SeqIO.parse(fetch_handle, "genbank"))
            fetch_handle.close()
            for rec in records:
                if not is_valid_gh5_cellulase(rec.description):
                    skipped += 1
                    continue
                all_records.append({
                    "Accession": rec.id,
                    "Organism": rec.annotations.get("source", "Unknown"),
                    "Sequence": str(rec.seq),
                    "Is_Thermophilic_Guess": 1,  # theo taxa ưa nhiệt
                    "Description": rec.description,
                })
            print(f"Tiến độ: giữ {len(all_records)} | loại {skipped} / duyệt {i + len(batch_ids)}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Bỏ qua batch {i}: {e}")
            continue

    df = pd.DataFrame(all_records).drop_duplicates(subset=["Accession"])
    df.to_csv("ai_training/gh5_thermophilic_raw.csv", index=False)
    print(f"✅ Lưu {len(df)} trình tự GH5 ưa nhiệt (loại {skipped}).")
    print("Top chi:")
    print(df['Organism'].str.split().str[0].value_counts().head(12).to_string())
    return df


if __name__ == "__main__":
    fetch_thermophilic(limit_per_query=2000)
