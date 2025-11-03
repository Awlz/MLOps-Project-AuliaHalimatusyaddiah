import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="Experiment_XGBoost_AutoLog")
    args = parser.parse_args()

    # Load Dataset
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    # HILANGKAN setting tracking URI di sini
    # Biarkan MLflow menggunakan MLFLOW_TRACKING_URI dari environment
    print(f"Current MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment Name: {args.experiment_name}")

    # Set experiment (tanpa set tracking URI)
    mlflow.set_experiment(args.experiment_name)

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

if __name__ == "__main__":
    main()
