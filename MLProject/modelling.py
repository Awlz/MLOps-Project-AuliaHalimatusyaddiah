import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

with mlflow.start_run():
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE: {rmse}")

    mlflow.log_metric("rmse", rmse)
    mlflow.xgboost.log_model(model, artifact_path="model")

print("Model retraining completed successfully.")
