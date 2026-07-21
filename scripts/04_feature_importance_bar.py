import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def plot_feature_importance(enriched_csv):
    df = pd.read_csv(enriched_csv)
    features = ['Aliphatic_Index', 'Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    X = df[features]
    # Gán nhãn giống như cách bạn làm trong PCA
    y = df['Aliphatic_Index'].apply(lambda x: 1 if x > 95 else 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='magma')
    plt.title('Figure 2: Contribution of Physicochemical Properties to GH5 Thermostability')
    plt.savefig("ai_training/figure2_feature_importance.png", dpi=300)
    print("✅ Đã tạo Figure 2: feature_importance.png")

if __name__ == "__main__":
    plot_feature_importance("ai_training/gh5_features_enriched.csv")
