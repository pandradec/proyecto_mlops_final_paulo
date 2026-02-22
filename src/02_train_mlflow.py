import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_model_mlflow():

    print("Starting MLflow training...")

    # Set experiment
    #mlflow.set_experiment("Credit_Risk_Experiment")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():

        # 1. Load data
        X_train = pd.read_csv("data/training/X_train.csv")
        X_test = pd.read_csv("data/training/X_test.csv")
        y_train = pd.read_csv("data/training/y_train.csv").values.ravel()
        y_test = pd.read_csv("data/training/y_test.csv").values.ravel()

        print("Data loaded successfully.")

        # 2. Model parameters
        n_estimators = 200
        random_state = 42

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        # 3. Train
        model.fit(X_train, y_train)
        print("Model training completed.")

        # 4. Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 5. Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print("\nModel Performance (MLflow Run):")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")

        # 6. Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # 7. Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # 8. Log model
        #mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(
            model,
            name="credit_risk_model",
            registered_model_name="Credit_Risk_Model"
        )
        

        # 9. Save columns as artifact
        os.makedirs("models", exist_ok=True)
        columns_path = "models/model_columns.pkl"
        pd.Series(X_train.columns).to_pickle(columns_path)
        mlflow.log_artifact(columns_path)

        print("\nModel and artifacts logged to MLflow successfully.")


if __name__ == "__main__":
    train_model_mlflow()