import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

def generate_performance_table(model_path, data_csv, output_csv):
    # 1. Tải mô hình và dữ liệu
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file mô hình tại {model_path}")
        return

    model = joblib.load(model_path)
    df = pd.read_csv(data_csv)

    # 2. Chuẩn bị tập dữ liệu kiểm tra (Khớp hoàn toàn với logic File 08 v3)
    # Loại bỏ Aliphatic_Index để đảm bảo tính khách quan (Non-Leaking)
    features = ['Instability_Index', 'Aromaticity', 'Isoelectric_Point', 'GRAVY']
    X = df[features]
    y = df['Aliphatic_Index'].apply(lambda x: 1 if x > 85 else 0)

    # Chia tập test 20% với random_state=42 để kết quả nhất quán
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Thực hiện dự đoán
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. Tính toán bộ chỉ số Q1
    metrics = {
        'Metric': ['Accuracy', 'Precision (Positive)', 'Recall (Sensitivity)', 'F1-Score', 'ROC-AUC'],
        'Value': [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_prob)
        ],
        'Description': [
            'Tổng độ chính xác dự đoán',
            'Độ tin cậy khi máy báo "Bền nhiệt"',
            'Khả năng không bỏ sót mẫu bền nhiệt',
            'Chỉ số cân bằng giữa Precision và Recall',
            'Khả năng phân loại tổng quát (AUC)'
        ]
    }

    # 5. Tạo DataFrame và định dạng số thập phân
    performance_df = pd.DataFrame(metrics)
    performance_df['Value'] = performance_df['Value'].apply(lambda x: f"{x:.4f}")

    # 6. Xuất file CSV để copy vào Word/Excel
    performance_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*50)
    print("      TABLE 1: MODEL PERFORMANCE METRICS")
    print("="*50)
    print(performance_df[['Metric', 'Value']])
    print("="*50)
    print(f"\n✅ Đã lưu bảng kết quả tại: {output_csv}")
    print("👉 Bạn có thể mở file này bằng Excel để copy vào Paper.")

if __name__ == "__main__":
    generate_performance_table(
        model_path='ai_training/gh5_thermo_model.pkl',
        data_csv='ai_training/gh5_features_enriched.csv',
        output_csv='ai_training/model_performance_table_v3.csv'
    )
