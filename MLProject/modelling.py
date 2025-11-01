import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# === Konfigurasi direktori kerja ===
# Pastikan semua file berada di dalam folder MLProject
base_dir = os.path.dirname(os.path.abspath(__file__))  # direktori tempat modelling.py berada
os.chdir(base_dir)  # ubah ke direktori tersebut agar relatif path sesuai

# === Baca dataset hasil preprocessing ===
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# === Pastikan folder mlruns sudah ada ===
mlruns_path = os.path.join(base_dir, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# === Konfigurasi MLflow Tracking lokal ===
mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
mlflow.set_experiment("Experiment_XGBoost_Baseline")

# === Inisialisasi model XGBoost ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

# === Jalankan eksperimen MLflow ===
with mlflow.start_run(run_name="XGBoost_Baseline"):
    # 🔹 Logging parameter model
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("reg_lambda", 1.0)
    mlflow.log_param("random_state", 42)

    # Latih model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi metrik
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging metrik
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Logging model ke MLflow artifacts
    mlflow.xgboost.log_model(model, artifact_path="model_boosting")

    # Cetak hasil evaluasi di konsol
    print("=== Hasil Evaluasi Model Boosting (XGBoost) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
