import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE 
import joblib
import os

def train_balanced_model_v3(input_csv):
    if not os.path.exists('ai_training'):
        os.makedirs('ai_training')

    df = pd.read_csv(input_csv)
    
    # 1. TẠO NHÃN (LABEL) - Dựa trên ngưỡng Aliphatic Index
    y = df['Aliphatic_Index'].apply(lambda x: 1 if x > 85 else 0)
    
    # 2. CHỌN ĐẶC TRƯNG (FEATURES) - QUAN TRỌNG: 
    # Loại bỏ 'Aliphatic_Index' khỏi X để máy không bị "lộ đề"
    features = ['Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    X = df[features]
    
    print(f"Thống kê ban đầu: Nhóm thường: {sum(y==0)}, Nhóm bền nhiệt: {sum(y==1)}")

    # 3. Cân bằng dữ liệu (SMOTE)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # 4. Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # 5. Huấn luyện mô hình (Giới hạn độ sâu max_depth để máy không học vẹt)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # 6. Đánh giá và vẽ Confusion Matrix
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix (Non-Leaking Model)')
    plt.xlabel('AI Predicted')
    plt.ylabel('Actual Nature')
    plt.savefig("ai_training/model_confusion_matrix_v3.png", dpi=300)

    # 7. Vẽ Feature Importance (Figure 4 thực tế)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', color='teal')
    plt.title('Figure 4: Real Structural Drivers (Excluding Aliphatic Index)')
    plt.savefig("ai_training/figure4_feature_importance_real.png", dpi=300)

    # 8. Lưu model
    joblib.dump(model, 'ai_training/gh5_thermo_model.pkl')
    
    print(f"\n✅ XONG! Độ chính xác thực tế: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("📍 Kiểm tra file 'model_confusion_matrix_v3.png'. Bây giờ kết quả sẽ thực tế hơn.")

if __name__ == "__main__":
    train_balanced_model_v3("ai_training/gh5_features_enriched.csv")