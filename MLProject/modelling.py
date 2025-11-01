import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv")).squeeze()
X_test = pd.read_csv(os.path.join(base_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(base_dir, "y_test.csv")).squeeze()

mlruns_path = os.path.join(base_dir, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
mlflow.set_experiment("Experiment_XGBoost_Baseline")

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

with mlflow.start_run(run_name="XGBoost_Baseline"):
    mlflow.log_params(model.get_params())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.xgboost.log_model(model, artifact_path="model")

    print("=== Hasil Evaluasi Model (XGBoost) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
