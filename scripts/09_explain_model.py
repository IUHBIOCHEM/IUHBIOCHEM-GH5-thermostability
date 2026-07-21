import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explain_thermo_model(model_path, features_csv):
    # 1. Load mô hình và dữ liệu
    model = joblib.load(model_path)
    df = pd.read_csv(features_csv)
    
    features = ['Aliphatic_Index', 'Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    
    # 2. Lấy tầm quan trọng của các đặc trưng
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

    # 3. Vẽ biểu đồ Feature Importance (Figure 5 cho Paper)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Figure 5: Contribution of Structural Parameters to Thermostability Prediction')
    plt.xlabel('Gini Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("ai_training/figure5_feature_importance.png", dpi=300)
    
    print("--- Phân tích tầm quan trọng của đặc trưng ---")
    print(feat_imp)
    print("\n✅ Đã lưu Figure 5. Hãy dùng dữ liệu này để giải thích cơ chế bền nhiệt.")

if __name__ == "__main__":
    explain_thermo_model('ai_training/gh5_thermo_model.pkl', 'ai_training/gh5_features_enriched.csv')
