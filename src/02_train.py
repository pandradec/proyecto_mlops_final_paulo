import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_model():

    print("Starting production model training...")

    # 1. Load data
    X_train = pd.read_csv("data/training/X_train.csv")
    X_test = pd.read_csv("data/training/X_test.csv")
    y_train = pd.read_csv("data/training/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/training/y_test.csv").values.ravel()

    print("Data loaded successfully.")

    # 2. Initialize Random Forest (Champion Model)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # 3. Train model
    model.fit(X_train, y_train)

    print("Model training completed.")

    # 4. Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5. Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("\nFinal Model Performance (Random Forest):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    # 6. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/credit_risk_model.pkl")
    # Save training columns
    joblib.dump(X_train.columns.tolist(), "models/model_columns.pkl")

    print("\nProduction model saved as models/credit_risk_model.pkl")
    print("Model columns saved as models/model_columns.pkl")

    return model


if __name__ == "__main__":
    train_model()