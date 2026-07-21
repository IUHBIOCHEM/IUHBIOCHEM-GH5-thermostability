import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             roc_curve, auc, classification_report)
from sklearn.model_selection import train_test_split

def detailed_evaluation(model_path, data_csv):
    # 1. Load model và dữ liệu
    model = joblib.load(model_path)
    df = pd.read_csv(data_csv)
    
    # 2. Chuẩn bị lại tập Test - PHẢI KHỚP VỚI FILE 08 V3
    # Chúng ta loại bỏ Aliphatic_Index khỏi features
    features = ['Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    X = df[features]
    y = df['Aliphatic_Index'].apply(lambda x: 1 if x > 85 else 0)
    
    # Chia tập test 20% giống file 08
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Tính toán các chỉ số
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] 

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("--- BẢNG CHỈ SỐ ĐÁNH GIÁ MODEL V3 (DỮ LIỆU CHUẨN Q1) ---")
    print(f"1. Accuracy: {acc:.4f}")
    print(f"2. F1-Score: {f1:.4f}")
    print(f"3. ROC-AUC:  {roc_auc:.4f}")
    print("\nChi tiết Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4. Vẽ đường cong ROC (Figure 7 mới)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Figure 7: ROC Curve - Model Validation (Non-Leaking)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("ai_training/figure7_roc_curve_v3.png", dpi=300)
    
    print("\n✅ Đã cập nhật Figure 7. Đường cong ROC lúc này sẽ phản ánh đúng năng lực thực tế của AI.")

if __name__ == "__main__":
    detailed_evaluation('ai_training/gh5_thermo_model.pkl', 'ai_training/gh5_features_enriched.csv')