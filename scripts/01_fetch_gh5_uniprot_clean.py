from Bio import Entrez, SeqIO
import pandas as pd
import os
import time

# NCBI yêu cầu email thật để dùng Entrez
Entrez.email = "hothienhoang@iuh.edu.vn"

# Từ khóa loại bỏ: những protein KHÔNG phải enzyme cellulase GH5
# (protein điều hòa phiên mã, protein giả định, protein chỉ có domain bám...)
EXCLUDE_KEYWORDS = [
    "regulator",
    "transcription",
    "hypothetical",
    "binding protein",
    "binding-protein",
    "uncharacterized",
    "putative uncharacterized",
    "transporter",
]

# Từ khóa xác nhận đúng là cellulase / GH5 (phải khớp ít nhất một)
INCLUDE_KEYWORDS = [
    "cellulase",
    "endoglucanase",
    "endo-1,4-beta-glucanase",
    "glucanase",
    "glycoside hydrolase family 5",
    "glycosyl hydrolase family 5",
    "glycoside hydrolase 5",
    "gh5",
]


def is_valid_gh5_cellulase(description):
    """Giữ lại nếu description có từ khóa cellulase/GH5 và không có từ khóa loại bỏ."""
    d = (description or "").lower()
    if any(bad in d for bad in EXCLUDE_KEYWORDS):
        return False
    return any(good in d for good in INCLUDE_KEYWORDS)


def fetch_large_gh5_dataset(limit=2000):
    if not os.path.exists('ai_training'):
        os.makedirs('ai_training')

    # Truy vấn giới hạn ở trường [Protein Name] (tên protein), dùng nháy kép cho cụm từ,
    # và bắt buộc thuộc GH5 hoặc mang EC 3.2.1.4 (endoglucanase) để đảm bảo đúng họ/chức năng.
    query = (
        '('
        'cellulase[Protein Name] OR '
        'endoglucanase[Protein Name] OR '
        '"glycoside hydrolase family 5"[Protein Name] OR '
        '"glycosyl hydrolase family 5"[Protein Name]'
        ') AND ('
        '3.2.1.4[ECNO] OR '
        '"glycoside hydrolase family 5"[Title] OR '
        '"glycosyl hydrolase family 5"[Title] OR '
        'GH5[Title]'
        ')'
    )

    print(f"--- Đang tìm kiếm trình tự mục tiêu: {limit} ---")
    print(f"Query: {query}")

    try:
        search_handle = Entrez.esearch(db="protein", term=query, retmax=limit)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]
        print(f"Đã tìm thấy {len(id_list)} mã định danh tiềm năng.")

        all_records = []
        skipped = 0
        batch_size = 50  # Giảm batch size để tránh lỗi timeout khi tải nhiều

        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            try:
                fetch_handle = Entrez.efetch(db="protein", id=batch_ids, rettype="gp", retmode="text")
                records = list(SeqIO.parse(fetch_handle, "genbank"))
                fetch_handle.close()

                for rec in records:
                    # BỘ LỌC HẬU KIỂM: loại bỏ protein không phải cellulase GH5
                    # dựa trên description (regulator / hypothetical / binding protein...)
                    if not is_valid_gh5_cellulase(rec.description):
                        skipped += 1
                        continue

                    # Trích xuất thông tin nhiệt độ nếu có trong phần note
                    features_text = str(rec.features)
                    is_thermo = 1 if any(word in features_text.lower() for word in ['thermo', 'hot', 'stable']) else 0

                    all_records.append({
                        "Accession": rec.id,
                        "Organism": rec.annotations.get("source", "Unknown"),
                        "Sequence": str(rec.seq),
                        "Is_Thermophilic_Guess": is_thermo,  # Dự đoán chịu nhiệt dựa trên từ khóa
                        "Description": rec.description
                    })
                print(f"Tiến độ: giữ {len(all_records)} | loại {skipped} / đã duyệt {i + len(batch_ids)}...")
                time.sleep(0.5)
            except Exception as batch_err:
                print(f"Bỏ qua batch {i} do lỗi kết nối... ({batch_err})")
                continue

        df = pd.DataFrame(all_records)
        df.to_csv("ai_training/gh5_training_data_large.csv", index=False)
        print(f"✅ Hoàn tất! Đã lưu {len(df)} trình tự cellulase GH5 hợp lệ "
              f"(đã loại {skipped} trình tự không phù hợp).")
        return df

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    df_gh5 = fetch_large_gh5_dataset(2000)
