import pandas as pd
import os

def extract_top_sequences(input_csv, top_n=50):
    if not os.path.exists('ai_training'):
        os.makedirs('ai_training')

    # 1. Đọc dữ liệu đã làm giàu đặc trưng
    df = pd.read_csv(input_csv)
    
    # 2. Sắp xếp theo Aliphatic_Index giảm dần để lấy các chuỗi bền nhiệt nhất
    # Đồng thời lọc bỏ các chuỗi quá ngắn (< 200 AA) để đảm bảo lấy được vùng domain hoàn chỉnh
    top_df = df[df['Sequence'].str.len() > 200].sort_values(by='Aliphatic_Index', ascending=False).head(top_n)
    
    # 3. Xuất file FASTA
    fasta_path = f"ai_training/top_{top_n}_thermostable_gh5.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in top_df.iterrows():
            # Header định dạng: >ID_Loài_ChỉSốBềnNhiệt
            header = f">{row['Accession']}_{row['Organism'].replace(' ', '_')}_AI{row['Aliphatic_Index']}"
            f.write(f"{header}\n{row['Sequence']}\n")
            
    print(f"✅ Đã trích xuất thành công {len(top_df)} trình tự bền nhiệt nhất.")
    print(f"📍 File FASTA lưu tại: {fasta_path}")
    print("\n--- Top 5 trình tự bền nhiệt nhất trong Dataset ---")
    print(top_df[['Accession', 'Organism', 'Aliphatic_Index']].head())

if __name__ == "__main__":
    extract_top_sequences("ai_training/gh5_features_enriched.csv")
