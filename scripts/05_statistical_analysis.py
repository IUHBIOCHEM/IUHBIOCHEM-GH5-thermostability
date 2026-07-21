import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def perform_statistical_testing(enriched_csv):
    df = pd.read_csv(enriched_csv)
    
    # Phân nhóm dựa trên ngưỡng Aliphatic Index (giống Figure 1)
    threshold = 95
    group_high = df[df['Aliphatic_Index'] > threshold]
    group_low = df[df['Aliphatic_Index'] <= threshold]
    
    features = ['Aliphatic_Index', 'Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    stats_results = []

    print(f"--- Phân tích thống kê (Ngưỡng Aliphatic: {threshold}) ---")
    print(f"Số mẫu nhóm High: {len(group_high)}")
    print(f"Số mẫu nhóm Low: {len(group_low)}\n")

    for feat in features:
        # Kiểm định t-test độc lập
        t_stat, p_val = stats.ttest_ind(group_high[feat], group_low[feat], equal_var=False)
        
        # Tính giá trị trung bình mỗi nhóm
        mean_high = group_high[feat].mean()
        mean_low = group_low[feat].mean()
        
        stats_results.append({
            'Feature': feat,
            'Mean_High': round(mean_high, 3),
            'Mean_Low': round(mean_low, 3),
            'P_value': f"{p_val:.2e}",
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

    # Lưu bảng kết quả thống kê (Bảng này đưa vào phần Results của paper)
    results_df = pd.DataFrame(stats_results)
    results_df.to_csv("ai_training/statistical_results.csv", index=False)
    print(results_df)

    # Vẽ biểu đồ Boxplot để trực quan hóa sự khác biệt (Figure 3 cho paper)
    plt.figure(figsize=(15, 5))
    df['Group'] = df['Aliphatic_Index'].apply(lambda x: 'High' if x > threshold else 'Low')
    
    for i, feat in enumerate(['Aliphatic_Index', 'Instability_Index', 'GRAVY'], 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='Group', y=feat, data=df, palette='Set2')
        plt.title(f'{feat} Comparison')
    
    plt.tight_layout()
    plt.savefig("ai_training/figure3_boxplots.png", dpi=300)
    print("\n✅ Đã tạo bảng thống kê và Figure 3 (Boxplots).")

if __name__ == "__main__":
    perform_statistical_testing("ai_training/gh5_features_enriched.csv")
