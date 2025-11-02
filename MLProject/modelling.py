import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load Dataset
X_train = pd.read_csv("MLProject/X_train.csv")
y_train = pd.read_csv("MLProject/y_train.csv").squeeze()
X_test = pd.read_csv("MLProject/X_test.csv")
y_test = pd.read_csv("MLProject/y_test.csv").squeeze()

# Konfigurasi MLflow Tracking Lokal
mlruns_path = os.path.abspath("mlruns")
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
mlflow.set_experiment("Experiment_XGBoost_AutoLog")

# Mengaktifkan Autolog untuk XGBoost
mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

# Inisialisasi dan Melatih Model
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

with mlflow.start_run(run_name="XGBoost_AutoLog"):
    # Latih model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi manual 
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=== Hasil Evaluasi Model XGBoost (Autolog) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

