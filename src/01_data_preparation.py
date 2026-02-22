import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():

    print("Starting data preparation...")

    # 1. Load dataset
    df = pd.read_csv("data/raw/german_credit_data.csv")

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    # 2. Handle missing values
    df["Saving accounts"] = df["Saving accounts"].fillna("No_info")
    df["Checking account"] = df["Checking account"].fillna("No_info")

    # 3. Convert target variable
    df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})

    # 4. Convert Job to categorical
    df["Job"] = df["Job"].astype("category")

    # 5. One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    print("Encoding completed.")

    # 6. Split features and target
    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # importante por desbalance
    )

    print("Train-test split completed.")

    # 7. Create training folder if not exists
    os.makedirs("data/training", exist_ok=True)

    # 8. Save datasets
    X_train.to_csv("data/training/X_train.csv", index=False)
    X_test.to_csv("data/training/X_test.csv", index=False)
    y_train.to_csv("data/training/y_train.csv", index=False)
    y_test.to_csv("data/training/y_test.csv", index=False)

    print("Data saved successfully in data/training/")
    print("Data preparation finished.")


if __name__ == "__main__":
    main()