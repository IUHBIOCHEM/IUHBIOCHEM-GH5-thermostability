import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_gh5_landscape(enriched_csv):
    df = pd.read_csv(enriched_csv)
    
    # Lọc các cột số quan trọng
    features = ['Aliphatic_Index', 'Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    x = df[features]
    
    # Chuẩn hóa dữ liệu để các chỉ số có cùng trọng số
    x_scaled = StandardScaler().fit_transform(x)
    
    # Chạy PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_scaled)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    
    # Phân nhóm dựa trên Aliphatic Index (Ngưỡng lý thuyết cho tính chịu nhiệt thường > 90-95)
    pca_df['Thermostability'] = df['Aliphatic_Index'].apply(
        lambda x: 'High (Potential Thermostable)' if x > 95 else 'Medium/Low'
    )

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Thermostability', style='Thermostability', 
                    data=pca_df, palette='coolwarm', s=60, alpha=0.7)
    
    plt.title('Figure 1: Physicochemical Landscape of 2000 GH5 Cellulases', fontsize=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) - Stability Axis')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%) - Charge/Hydrophobicity Axis')
    plt.legend(title='Predicted Nature')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("ai_training/figure1_pca_landscape.png", dpi=300, bbox_inches='tight')
    print("✅ Đã tạo Figure 1: figure1_pca_landscape.png")

if __name__ == "__main__":
    plot_gh5_landscape("ai_training/gh5_features_enriched.csv")
