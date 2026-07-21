import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os

def calculate_protein_features(input_csv):
    df = pd.read_csv(input_csv)
    # Loại bỏ các dòng có chuỗi chứa ký tự lạ (X, U, Z...) để tránh lỗi Biopython
    df = df[df['Sequence'].str.contains('^[ACDEFGHIKLMNPQRSTVWY]+$', regex=True)]
    
    results = []
    for index, row in df.iterrows():
        seq = row['Sequence']
        analysed = ProteinAnalysis(seq)
        
        # AA Composition (%)
        aa_perc = analysed.get_amino_acids_percent()
        
        # Chỉ số Aliphatic (Độ bền nhiệt)
        ali_index = (aa_perc['A']*100) + 2.9*(aa_perc['V']*100) + 3.9*((aa_perc['I']*100) + (aa_perc['L']*100))
        
        # Các chỉ số quan trọng khác cho AI
        results.append({
            'Accession': row['Accession'],
            'Aliphatic_Index': round(ali_index, 2),
            'Instability_Index': round(analysed.instability_index(), 2),
            'Aromaticity': round(analysed.aromaticity(), 2),
            'Isoelectric_Point': round(analysed.isoelectric_point(), 2),
            'GRAVY': round(analysed.gravy(), 2),
            'Length': row['Length'] if 'Length' in df.columns else len(seq)
        })
    
    features_df = pd.DataFrame(results)
    final_df = pd.merge(df, features_df, on='Accession')
    final_df.to_csv("ai_training/gh5_features_enriched.csv", index=False)
    print("✅ Đã hoàn thành file: gh5_features_enriched.csv")

if __name__ == "__main__":
    # Dùng tập đã xác minh domain GH5 (PF00150) thay cho dữ liệu thô chưa lọc
    calculate_protein_features("ai_training/gh5_verified_pf00150.csv")
