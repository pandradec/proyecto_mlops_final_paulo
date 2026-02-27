import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def main():

    print("Loading training data...")

    X_train = pd.read_csv("data/training/X_train.csv")
    X_test = pd.read_csv("data/training/X_test.csv")
    y_train = pd.read_csv("data/training/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/training/y_test.csv").values.ravel()

    results = {}

    # --------------------------------------------------
    # 1️⃣ Logistic Regression + StandardScaler
    # --------------------------------------------------
    print("\nTraining Logistic Regression with StandardScaler...")

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    lr_pipeline.fit(X_train, y_train)
    results["Logistic_Regression"] = evaluate_model(lr_pipeline, X_test, y_test)

    # --------------------------------------------------
    # 2️⃣ Random Forest
    # --------------------------------------------------
    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    results["Random_Forest"] = evaluate_model(rf_model, X_test, y_test)

    # --------------------------------------------------
    # Show results
    # --------------------------------------------------
    print("\nModel Comparison Results:\n")

    for model_name, metrics in results.items():
        print(f"{model_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("-" * 30)
        
    # --------------------------------------------------
    # Save results to CSV
    # --------------------------------------------------
    print("\nSaving experiment results...")

    results_df = pd.DataFrame(results).T
    results_df.reset_index(inplace=True)
    results_df.rename(columns={"index": "model"}, inplace=True)

    os.makedirs("experiments", exist_ok=True)
    results_df.to_csv("experiments/results.csv", index=False)

    print("Results saved in experiments/results.csv")


    # --------------------------------------------------
    # Select best model (based on ROC-AUC)
    # --------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
    print(f"\nBest model based on ROC-AUC: {best_model_name}")

    if best_model_name == "Logistic_Regression":
        best_model = lr_pipeline
    else:
        best_model = rf_model

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_credit_risk_model.pkl")

    print("\nBest model saved as models/best_credit_risk_model.pkl")


if __name__ == "__main__":
    main()